"""vLLM patch serving train_vector.py's MODE project hook exactly: at the output of each decoder layer of the window, every position,
h <- h - ((h - mu_i).u) u. RHABLATION_PROJ is a safetensors file with u [d_model] or [k, d_model] (fp32 unit vectors: one shared, or one per window layer), c [n_layers] (mu_i.u per decoder layer) and
layers [k] (decoder layer indices), and optionally gen_only, which hooks model-generated tokens only: a token past its request's prompt length, or inside a finished assistant turn of the
prompt (generated_mask), so a multi-turn history gets the hook it had when it was written. The mask is refilled each step. Unset, nothing is patched. A decoder layer returns (mlp output, residual) and its output is their sum, so the
correction goes on the mlp output. The MTP draft layers are left alone. Installed by box_proj_setup.sh next to Will's resid-bias plugin, whose register() calls register() here.

RHABLATION_PROJ_ARCH picks the model family (sweep/mission_second_model.md): qwen (default; Qwen3.5 / Qwen3.8, the classes above), glm4_moe_lite (GLM-4.7-Flash: Glm4MoeLiteDecoderLayer, which
sets self.layer_idx from its prefix and returns (mlp output, residual) like Qwen3-Next) or nemotron_h (NVIDIA Nemotron 3.5 Lightning:
the four NemotronH*DecoderLayer classes, Mamba / MoE / MLP / attention, every one a layer of config.layers_block_type; each returns (mixer output, residual) and the next norm adds
them, as Qwen3-Next's layers do, so the same correction on the mixer output edits the layer's output residual). gen_only is Qwen-only (its mask reads Qwen's template tokens).
RHABLATION_PROJ_CHECK=1 (nemotron_h, eager serving) prints, for the first forward passes, the largest |component along u minus c| after the edit at every hooked layer: the mechanical check."""
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
    arch = os.environ.get("RHABLATION_PROJ_ARCH", "qwen")
    if arch == "nemotron_h": return register_nemotron_h(path)
    if arch == "glm4_moe_lite": return register_glm4_moe_lite(path)
    assert arch == "qwen", arch
    from vllm.model_executor.models import qwen3_5, qwen3_next
    saved = load_file(path)
    window = saved["layers"].tolist()
    gen_only = "gen_only" in saved
    mask = []   # the one generated-token mask, made by the first hooked layer on the worker
    orig_init = qwen3_5.Qwen3_5DecoderLayer.__init__
    orig_forward = qwen3_next.Qwen3NextDecoderLayer.forward

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

    def patched_init(self, vllm_config, layer_type, prefix=""):
        orig_init(self, vllm_config, layer_type, prefix=prefix)
        self.proj_on = self.layer_idx in window and "mtp" not in prefix
        if self.proj_on:
            device = torch.empty(0).device
            self.register_buffer("proj_u", (saved["u"] if saved["u"].ndim == 1 else saved["u"][window.index(self.layer_idx)]).float().to(device), persistent=False)
            self.register_buffer("proj_c", saved["c"][self.layer_idx].float().to(device), persistent=False)
            if gen_only:
                assert vllm_config.use_v2_model_runner, "the mask is filled in the v2 runner's prepare_inputs"
                if not mask:
                    mask.append(torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens + 1024, device=device))
                    patch_runner()
                self.register_buffer("proj_gen", mask[0], persistent=False)
            print(f"vllm_proj: hook on {prefix}, c {self.proj_c.item():+.3f}", flush=True)

    def patched_forward(self, hidden_states, residual, positions, **kwargs):
        hidden_states, residual = orig_forward(self, hidden_states, residual, positions, **kwargs)
        if getattr(self, "proj_on", False):
            s = (hidden_states + residual).float() @ self.proj_u - self.proj_c
            if gen_only: s = s * self.proj_gen[:s.shape[0]]
            hidden_states = hidden_states - (s[:, None] * self.proj_u).to(hidden_states.dtype)
        return hidden_states, residual

    qwen3_5.Qwen3_5DecoderLayer.__init__ = patched_init
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_forward


def register_nemotron_h(path: str) -> None:
    """The same hook on vLLM's Nemotron-H decoder layers (nemotron_h.py: NemotronHMambaDecoderLayer, NemotronHMoEDecoderLayer, NemotronHMLPDecoderLayer, NemotronHAttentionDecoderLayer).
    Every class takes config, layer_idx, ..., prefix as keyword arguments from NemotronHModel.get_layer and its forward returns (mixer output, residual); the layer's output residual is their sum."""
    from vllm.model_executor.models import nemotron_h
    saved = load_file(path)
    assert "gen_only" not in saved, "gen_only is implemented for the Qwen template only"
    window = saved["layers"].tolist()
    check = os.environ.get("RHABLATION_PROJ_CHECK", "0") == "1"
    counts = {}   # prefix -> forward passes printed under the check

    def patch(cls):
        orig_init, orig_forward = cls.__init__, cls.forward

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            layer_idx = kwargs["layer_idx"] if "layer_idx" in kwargs else args[1]
            prefix = kwargs.get("prefix", "")
            self.proj_layer_idx, self.proj_prefix = layer_idx, prefix
            self.proj_on = layer_idx in window and "mtp" not in prefix
            if self.proj_on:
                device = torch.empty(0).device
                self.register_buffer("proj_u", (saved["u"] if saved["u"].ndim == 1 else saved["u"][window.index(layer_idx)]).float().to(device), persistent=False)
                self.register_buffer("proj_c", saved["c"][layer_idx].float().to(device), persistent=False)
                print(f"vllm_proj: hook on {prefix} ({cls.__name__}), c {self.proj_c.item():+.3f}", flush=True)

        def patched_forward(self, *args, **kwargs):
            hidden_states, residual = orig_forward(self, *args, **kwargs)
            if getattr(self, "proj_on", False):
                s = (hidden_states + residual).float() @ self.proj_u - self.proj_c
                hidden_states = hidden_states - (s[:, None] * self.proj_u).to(hidden_states.dtype)
                if check and counts.get(self.proj_prefix, 0) < 3:
                    counts[self.proj_prefix] = counts.get(self.proj_prefix, 0) + 1
                    after = (hidden_states + residual).float() @ self.proj_u - self.proj_c
                    print(f"vllm_proj check {self.proj_prefix}: {s.shape[0]} tokens, before |comp - c| max {s.abs().max().item():.3f} mean {s.abs().mean().item():.3f}, after max {after.abs().max().item():.4f} mean {after.abs().mean().item():.5f}, |u| {self.proj_u.norm().item():.4f}", flush=True)
            return hidden_states, residual

        cls.__init__, cls.forward = patched_init, patched_forward

    for cls in (nemotron_h.NemotronHMambaDecoderLayer, nemotron_h.NemotronHMoEDecoderLayer, nemotron_h.NemotronHMLPDecoderLayer, nemotron_h.NemotronHAttentionDecoderLayer):
        patch(cls)
    print(f"vllm_proj: nemotron_h patch installed, window {window[0]}-{window[-1]} ({len(window)} layers), u {tuple(saved['u'].shape)}", flush=True)


def register_glm4_moe_lite(path: str) -> None:
    """The same hook on vLLM's GLM-4.7 decoder layer (glm4_moe_lite.py: Glm4MoeLiteDecoderLayer(vllm_config, prefix, ...), self.layer_idx from the prefix, forward(positions, hidden_states, residual) -> (mlp output, residual))."""
    from vllm.model_executor.models import glm4_moe_lite
    saved = load_file(path)
    assert "gen_only" not in saved, "gen_only is implemented for the Qwen template only"
    window = saved["layers"].tolist()
    check = os.environ.get("RHABLATION_PROJ_CHECK", "0") == "1"
    counts = {}
    cls = glm4_moe_lite.Glm4MoeLiteDecoderLayer
    orig_init, orig_forward = cls.__init__, cls.forward

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        self.proj_prefix = prefix
        self.proj_on = self.layer_idx in window and "mtp" not in prefix
        if self.proj_on:
            device = torch.empty(0).device
            self.register_buffer("proj_u", (saved["u"] if saved["u"].ndim == 1 else saved["u"][window.index(self.layer_idx)]).float().to(device), persistent=False)
            self.register_buffer("proj_c", saved["c"][self.layer_idx].float().to(device), persistent=False)
            print(f"vllm_proj: hook on {prefix} ({cls.__name__}), c {self.proj_c.item():+.3f}", flush=True)

    def patched_forward(self, *args, **kwargs):
        hidden_states, residual = orig_forward(self, *args, **kwargs)
        if getattr(self, "proj_on", False):
            s = (hidden_states + residual).float() @ self.proj_u - self.proj_c
            hidden_states = hidden_states - (s[:, None] * self.proj_u).to(hidden_states.dtype)
            if check and counts.get(self.proj_prefix, 0) < 3:
                counts[self.proj_prefix] = counts.get(self.proj_prefix, 0) + 1
                after = (hidden_states + residual).float() @ self.proj_u - self.proj_c
                print(f"vllm_proj check {self.proj_prefix}: {s.shape[0]} tokens, before |comp - c| max {s.abs().max().item():.3f} mean {s.abs().mean().item():.3f}, after max {after.abs().max().item():.4f} mean {after.abs().mean().item():.5f}, |u| {self.proj_u.norm().item():.4f}", flush=True)
        return hidden_states, residual

    cls.__init__, cls.forward = patched_init, patched_forward
    print(f"vllm_proj: glm4_moe_lite patch installed, window {window[0]}-{window[-1]} ({len(window)} layers), u {tuple(saved['u'].shape)}", flush=True)
