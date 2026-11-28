"""vLLM patch serving train_vector.py's MODE project hook exactly: at the output of each decoder layer of the window, every position,
h <- h - ((h - mu_i).u) u. RHABLATION_PROJ is a safetensors file with u [d_model] or [k, d_model] (fp32 unit vectors: one shared, or one per window layer), c [n_layers] (mu_i.u per decoder layer) and
layers [k] (decoder layer indices), and optionally gen_only, which hooks model-generated tokens only: a token past its request's prompt length, or inside a finished assistant turn of the
prompt (generated_mask), so a multi-turn history gets the hook it had when it was written. The mask is refilled each step. Unset, nothing is patched. Installed by box_proj_setup.sh next to Will's resid-bias plugin, whose register() calls register() here.

Models (sweep/mission_second_model.md: the model is a parameter, one plugin): the decoder layer class of every supported model is patched when its module imports.
- Qwen3.5 / Qwen3.8 (qwen3_5.Qwen3_5DecoderLayer, forward from qwen3_next): a layer returns (mlp output, residual) and its output is their sum, so the correction goes on the mlp output. The MTP draft layers are left alone.
- Gemma 4 (gemma4.Gemma4DecoderLayer): a layer returns (hidden_states, None) with hidden_states the full residual after the layer's scalar, so the correction goes on hidden_states directly; the layer's index is its layer_idx.
gen_only's generated_mask knows the Qwen template's tokens only (IM_START / IM_END / <think>); no second-model row uses gen_only."""
import os

import numpy as np
import torch
from safetensors.torch import load_file

IM_START, IM_END, ASSISTANT, THINK_OPEN = 248045, 248046, 74455, 248068


def generated_mask(ids: list[int]) -> np.ndarray:
    """Which tokens the model wrote in the finished assistant turns of ids: after the template's `<|im_start|>assistant\\n` and `<think>\\n`, through `<|im_end|>`. An open turn at the end is the generation prompt, so it has none."""
    mask = np.zeros(len(ids), dtype=bool)
    ends = np.flatnonzero(np.array(ids) == IM_END)
    for start in np.flatnonzero(np.array(ids) == IM_START):
        if start + 1 >= len(ids) or ids[start + 1] != ASSISTANT: continue
        first = start + 3 + 2 * (start + 3 < len(ids) and ids[start + 3] == THINK_OPEN)
        after = ends[ends > start]
        if len(after): mask[first:after[0] + 1] = True
    return mask


def register() -> None:
    path = os.environ.get("RHABLATION_PROJ", "")
    if not path: return
    saved = load_file(path)
    window = saved["layers"].tolist()
    gen_only = "gen_only" in saved
    skip_first = "skip_first" in saved
    shift = float(saved["shift"].item()) if "shift" in saved else None   # sweep/mission_stage9_envs.md: steering, a constant shift h <- h - shift u at every position of every window layer (u, layers as above; c unused)
    scale = float(saved["scale"].item()) if "scale" in saved else 1.0   # the KL-matched random controls (box G update 1): h <- h - scale ((h.u) - c) u   # the sequence's first token (Gemma's <bos>, a massive-activation sink) is left alone (sweep/mission_second_model.md, box G: every-position hooks of random directions destroyed the model through it)
    mask = []   # the one generated-token mask, made by the first hooked layer on the worker

    def patch_runner():
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.states import RequestState
        orig_prepare = GPUModelRunner.prepare_inputs
        orig_add = RequestState.add_request

        def patched_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
            orig_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens)
            if not hasattr(self, "proj_generated"): self.proj_generated = torch.zeros(self.max_num_reqs, self.max_model_len, dtype=torch.bool, device=self.device)
            self.proj_generated[self.req_id_to_index[req_id], :prompt_len] = torch.from_numpy(generated_mask(list(all_token_ids[:prompt_len]))).to(self.device)

        def patched_prepare(self, *args, **kwargs):
            batch = orig_prepare(self, *args, **kwargs)
            n = batch.num_tokens
            req = torch.searchsorted(batch.query_start_loc[1:batch.num_reqs + 1], torch.arange(n, device=batch.positions.device, dtype=batch.query_start_loc.dtype), right=True)
            slot = batch.idx_mapping[req]
            positions = batch.positions[:n]
            mask[0][:n] = ((positions >= self.req_states.prompt_len.gpu[slot]) | self.req_states.proj_generated[slot, positions]).float()
            return batch

        RequestState.add_request = patched_add
        GPUModelRunner.prepare_inputs = patched_prepare

    def attach(self, vllm_config, layer_idx: int, prefix: str) -> None:
        """The hook's buffers on a decoder layer whose index is in the window (not on an MTP draft layer)."""
        self.proj_on = layer_idx in window and "mtp" not in prefix
        if self.proj_on:
            device = torch.empty(0).device
            self.register_buffer("proj_u", (saved["u"] if saved["u"].ndim == 1 else saved["u"][window.index(layer_idx)]).float().to(device), persistent=False)
            self.register_buffer("proj_c", saved["c"][layer_idx].float().to(device), persistent=False)
            if gen_only:
                assert vllm_config is None or vllm_config.use_v2_model_runner, "the mask is filled in the v2 runner's prepare_inputs"
                if not mask:
                    mask.append(torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens + 1024, device=device))
                    patch_runner()
                self.register_buffer("proj_gen", mask[0], persistent=False)
            print(f"vllm_proj: hook on {prefix}, c {self.proj_c.item():+.3f}", flush=True)

    def correction(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """((h.u) - c) u for the residual h [tokens, d_model], in h's dtype; what the hook subtracts. positions are the tokens' positions in their sequences."""
        if shift is not None: return (shift * self.proj_u).to(h.dtype).expand(h.shape[0], -1)
        s = h.float() @ self.proj_u - self.proj_c
        if gen_only: s = s * self.proj_gen[:s.shape[0]]
        if skip_first: s = s * (positions[:s.shape[0]] != 0).to(s.dtype)
        if scale != 1.0: s = s * scale
        return (s[:, None] * self.proj_u).to(h.dtype)

    # Qwen3.5 / Qwen3.8: the layer's output is hidden_states + residual, so the correction goes on hidden_states
    try:
        from vllm.model_executor.models import qwen3_5, qwen3_next
    except ImportError:
        qwen3_5 = None
    if qwen3_5 is not None:
        orig_init = qwen3_5.Qwen3_5DecoderLayer.__init__
        orig_forward = qwen3_next.Qwen3NextDecoderLayer.forward

        def patched_init(self, vllm_config, layer_type, prefix=""):
            orig_init(self, vllm_config, layer_type, prefix=prefix)
            attach(self, vllm_config, self.layer_idx, prefix)

        def patched_forward(self, hidden_states, residual, positions, **kwargs):
            hidden_states, residual = orig_forward(self, hidden_states, residual, positions, **kwargs)
            if getattr(self, "proj_on", False):
                hidden_states = hidden_states - correction(self, hidden_states + residual, positions)
            return hidden_states, residual

        qwen3_5.Qwen3_5DecoderLayer.__init__ = patched_init
        qwen3_next.Qwen3NextDecoderLayer.forward = patched_forward

    # Gemma 4: the layer returns (hidden_states, None), hidden_states the full residual after the layer scalar
    try:
        from vllm.model_executor.models import gemma4
    except ImportError:
        gemma4 = None
    if gemma4 is not None:
        g_init = gemma4.Gemma4DecoderLayer.__init__
        g_forward = gemma4.Gemma4DecoderLayer.forward

        def g_patched_init(self, config, cache_config=None, quant_config=None, prefix=""):
            g_init(self, config, cache_config=cache_config, quant_config=quant_config, prefix=prefix)
            attach(self, None, self.layer_idx, prefix)

        def g_patched_forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
            hidden_states, residual = g_forward(self, positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs)
            if getattr(self, "proj_on", False):
                hidden_states = hidden_states - correction(self, hidden_states, positions)
            return hidden_states, residual

        gemma4.Gemma4DecoderLayer.__init__ = g_patched_init
        gemma4.Gemma4DecoderLayer.forward = g_patched_forward
