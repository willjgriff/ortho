"""Diagnostic 2 (Gemma 4 port): why does a ~1-unit shift along a random direction at the R window's layers destroy the model (prand first-token KL 19.5) while refdom's ~40-unit
shifts cost 0.7? First-token KL on heretic's 100 harmless test prompts under hook variants: prand at one layer, prand scaled by 0.1, prand at the last position only, prand at every
position but the last, Gaussian noise of norm 1 at the same layers, zrand at one layer, and the per-position residual norms and components of one prompt. Run from /root/ortho with vLLM
stopped: .venv/bin/python scripts/second_model_bos_diag2.py -> acts/second_model/<tag>/bos_diag2.json"""
import json
import os
from pathlib import Path

import torch as t
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from rhenvs.common import CONFIGS, load_config

TAG = os.environ.get("SM_MODEL", "gemma4_31b")
MODEL_ID, REVISION, N, D = "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", 60, 5376
OUT = Path(f"acts/second_model/{TAG}")
HOOKS = Path.home() / "hooks"
PROMPTS = json.loads((CONFIGS / "refusal" / "prompts.json").read_text())
SYSTEM = load_config("refusal", f"{TAG}_off")["system_prompt"]
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="right")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
model.requires_grad_(False)
layers = model.model.language_model.layers

def render(p: str) -> list[int]:
    return tok(tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False).input_ids

# the hook in force: mode in proj (pinned projection), noise (add a fixed random vector of norm `scale`), none; positions: all, last, notlast
H = dict(mode="none", u=None, c=None, on=[False] * N, scale=1.0, positions="all", noise=None)
collect = None
def layer_hook(L):
    def f(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if collect is not None: collect(L, h)
        if H["on"][L]:
            if H["mode"] == "proj":
                s = (h.float() @ H["u"][L] - H["c"][L]) * H["scale"]
                if H["positions"] == "last": s[:, :-1] = 0
                if H["positions"] == "notlast": s[:, -1] = 0
                h = h - (s[..., None] * H["u"][L]).to(h.dtype)
            elif H["mode"] == "noise":
                d = H["noise"][L] * H["scale"]
                if H["positions"] == "last": h = t.cat([h[:, :-1], (h[:, -1:] + d).to(h.dtype)], 1)
                else: h = (h + d).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return f
for L in range(N): layers[L].register_forward_hook(layer_hook(L))

def load_row(name):
    hf = load_file(HOOKS / f"{name}.safetensors")
    u = t.zeros(N, D)
    for k, L in enumerate(hf["layers"].tolist()): u[L] = hf["u"].float() if hf["u"].ndim == 1 else hf["u"][k].float()
    return u.cuda(), hf["c"].float().cuda(), hf["layers"].tolist()

def set_hook(mode="none", row=None, layers_on=None, scale=1.0, positions="all"):
    H.update(mode=mode, scale=scale, positions=positions)
    if row is not None:
        u, c, window = load_row(row); H.update(u=u, c=c)
    else: window = list(range(N))
    H["on"] = [L in (layers_on if layers_on is not None else window) for L in range(N)]

kl_items = [render(p) for p in PROMPTS["harmless"]["test"]]   # one prompt per forward: no padding anywhere in this diagnostic
def first_logps():
    out = []
    with t.no_grad():
        for ids in kl_items: out.append(t.log_softmax(model(input_ids=t.tensor([ids], device="cuda")).logits[0, -1].float(), -1).cpu())
    return t.stack(out)
def kl(lp): return (base.exp() * (base - lp)).sum(-1).mean().item()

set_hook("none"); base = first_logps()
res = {}
def run(name, **kw):
    set_hook(**kw); v = kl(first_logps()); res[name] = v; print(f"{name}: first-token KL {v:.4f}", flush=True); set_hook("none")

STEP = [28, 31, 34, 37, 40, 43, 46, 49, 52, 55]
H["noise"] = t.nn.functional.normalize(t.randn(N, D, generator=t.Generator().manual_seed(11)), dim=1).cuda()
run("prand_all_unpadded", mode="proj", row="prand")
run("prand_L28_only", mode="proj", row="prand", layers_on=[28])
run("prand_L55_only", mode="proj", row="prand", layers_on=[55])
run("prand_scale0.1", mode="proj", row="prand", scale=0.1)
run("prand_last_position_only", mode="proj", row="prand", positions="last")
run("prand_every_position_but_last", mode="proj", row="prand", positions="notlast")
run("noise_norm1_R_window", mode="noise", layers_on=STEP, scale=1.0)
run("noise_norm1_L28_only", mode="noise", layers_on=[28], scale=1.0)
run("noise_norm10_L28_only", mode="noise", layers_on=[28], scale=10.0)
run("zrand_L28_only", mode="proj", row="zrand", layers_on=[28])
run("zrand_L0_only", mode="proj", row="zrand", layers_on=[0])
run("zrand_L0-9", mode="proj", row="zrand", layers_on=list(range(10)))
run("refdom_all_unpadded", mode="proj", row="refdom")
run("refdom_L37_only", mode="proj", row="refdom", layers_on=[37])
# per-position residual norms and prand components of one prompt, window layers
u, c, window = load_row("prand")
ids = kl_items[0]
per = {}
def grab(L, h):
    if L in window:
        x = h[0].float(); s = x @ u[L]
        per[L] = dict(norms=[round(v, 1) for v in x.norm(dim=-1).tolist()], comp_minus_pin=[round(v, 2) for v in (s - c[L]).tolist()], max_channel=[int(v) for v in x.abs().argmax(-1).tolist()], max_abs=[round(v, 1) for v in x.abs().max(-1).values.tolist()])
collect = grab
with t.no_grad(): model(input_ids=t.tensor([ids], device="cuda"))
collect = None
res["positions"] = dict(tokens=[tok.decode([i]) for i in ids], per_layer=per)
for L in (28, 37, 55): print(f"L{L} norms {per[L]['norms']}\n   max|h_i| {per[L]['max_abs']} channels {per[L]['max_channel']}\n   prand comp-pin {per[L]['comp_minus_pin']}")
(OUT / "bos_diag2.json").write_text(json.dumps(res, indent=1))
print(f"saved {OUT / 'bos_diag2.json'}")
