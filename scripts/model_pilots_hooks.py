#!/usr/bin/env python3
"""sweep/mission_model_pilots.md: the hook files of a pilot model, for scripts/second_model_serve_row.sh (SM_MODEL=<tag>, hooks in ~/hooks/<tag>/).
  base      zero hook (u = 0, c = 0, every decoder layer): the same patched forward as any hooked row
  hookstrong the same u at one layer round(0.75 N) with scale 30 (the plugin's h <- h - s((h.u) - c) u), a known-large edit
  hookrand  the hook check's random unit direction (seed 0) projected out (c = 0) at every layer of a late window, round(0.6 N)..round(0.94 N)
Run with any python that has torch, safetensors and transformers (root: /root/vllm-env/bin/python). n_layers / d_model come from the model's config.
    model_pilots_hooks.py <tag> <hf id> <revision>"""
import json, sys
from pathlib import Path
import torch as t
from safetensors.torch import save_file
from transformers import AutoConfig

tag, mid, rev = sys.argv[1:4]
c = AutoConfig.from_pretrained(mid, revision=rev, trust_remote_code=True)
c = getattr(c, "text_config", None) or c
N, D = c.num_hidden_layers, c.hidden_size
out = Path.home() / "hooks" / tag
out.mkdir(parents=True, exist_ok=True)
save_file(dict(u=t.zeros(D), c=t.zeros(N), layers=t.arange(N)), out / "base.safetensors")
g = t.Generator().manual_seed(0)
u = t.randn(D, generator=g); u = u / u.norm()
window = list(range(round(0.6 * N), round(0.94 * N) + 1))
save_file(dict(u=u, c=t.zeros(N), layers=t.tensor(window)), out / "hookrand.safetensors")
# hookstrong: the same u at one layer, round(0.75 N), with the plugin's scale 30 (h <- h - 30 (h.u) u, the component flipped and amplified 29x): a random unit
# direction projected out at s = 1 moves a d_model-wide residual by ~1/sqrt(d_model) of its norm, near the bf16 graph noise, so this row shows the hook path acts.
# (Scale 100 on the whole window compounds across layers and gave NaN logits on Nanbeige.)
save_file(dict(u=u, c=t.zeros(N), layers=t.tensor([round(0.75 * N)]), scale=t.tensor(30.0)), out / "hookstrong.safetensors")
info = dict(tag=tag, model=mid, revision=rev, n_layers=N, d_model=D, hookrand_window=[window[0], window[-1]], hookrand_seed=0)
(out / "hooks.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info))
