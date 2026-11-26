"""Diagnostic (sweep/mission_second_model.md, Gemma 4 port): why do the every-position hooks of random directions destroy the model (prand first-token KL 19.5, zrand 17.5, every Arditi
candidate KL >= 0.14)? Per layer of the R window, on heretic's harmless test prompts: the residual norm at position 0 (<bos>) against the other positions, the component along each row's
direction at position 0 against its pin, and the first-token KL of each hook with and without position 0 exempted. Run from /root/ortho with vLLM stopped:
    .venv/bin/python scripts/second_model_bos_diag.py   -> acts/second_model/<tag>/bos_diag.json"""
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

hook = dict(u=None, c=None, on=[False] * N, skip_first=False)
collect = None
def layer_hook(L):
    def f(module, args, output):
        global collect
        h = output[0] if isinstance(output, tuple) else output
        if collect is not None: collect(L, h)
        if hook["on"][L]:
            s = h.float() @ hook["u"][L] - hook["c"][L]
            if hook["skip_first"]: s[:, 0] = 0
            h = h - (s[..., None] * hook["u"][L]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return f
for L in range(N): layers[L].register_forward_hook(layer_hook(L))

def use(name: str | None, skip_first: bool = False):
    if name is None: hook.update(u=None, c=None, on=[False] * N, skip_first=False); return
    hf = load_file(HOOKS / f"{name}.safetensors")
    u = t.zeros(N, D)
    for k, L in enumerate(hf["layers"].tolist()): u[L] = hf["u"].float() if hf["u"].ndim == 1 else hf["u"][k].float()
    hook.update(u=u.cuda(), c=hf["c"].float().cuda(), on=[L in hf["layers"].tolist() for L in range(N)], skip_first=skip_first)

res = {}
# 1. residual norms per position and the components along each row at position 0 vs the pin (base model, 8 prompts)
items = [render(p) for p in PROMPTS["harmless"]["test"][:8]]
stats = {}
def grab(L, h):
    x = h[0].float()
    stats.setdefault(L, []).append(dict(norm_pos0=x[0].norm().item(), norm_pos1=x[1].norm().item(), norm_median_rest=x[1:].norm(dim=-1).median().item(), norm_last=x[-1].norm().item(), max_abs_pos0=x[0].abs().max().item(), argmax_pos0=x[0].abs().argmax().item(), max_abs_last=x[-1].abs().max().item()))
use(None)
collect = grab
with t.no_grad():
    for ids in items: model(input_ids=t.tensor([ids], device="cuda"))
collect = None
res["residual"] = {L: {k: sum(d[k] for d in v) / len(v) for k in v[0] if k != "argmax_pos0"} | dict(argmax_pos0=[d["argmax_pos0"] for d in v][:4]) for L, v in stats.items() if L in (0, 6, 12, 18, 24, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 59)}
for L in (0, 28, 37, 46, 55): print(f"L{L}: |h| pos0 {res['residual'][L]['norm_pos0']:.0f}, pos1 {res['residual'][L]['norm_pos1']:.0f}, median rest {res['residual'][L]['norm_median_rest']:.0f}, last {res['residual'][L]['norm_last']:.0f}; max|h_i| pos0 {res['residual'][L]['max_abs_pos0']:.0f} (channel {res['residual'][L]['argmax_pos0']}), last {res['residual'][L]['max_abs_last']:.0f}")
comp = {}
for row in ("prand", "refdom", "gpdom", "zrand"):
    hf = load_file(HOOKS / f"{row}.safetensors"); window = hf["layers"].tolist()
    u = t.zeros(N, D)
    for k, L in enumerate(window): u[L] = hf["u"].float() if hf["u"].ndim == 1 else hf["u"][k].float()
    u = u.cuda(); c = hf["c"].float()
    per = {}
    def grab2(L, h):
        if L in window:
            s = h[0].float() @ u[L]
            per.setdefault(L, []).append(dict(pos0=s[0].item(), rest_mean=s[1:].mean().item(), rest_sd=s[1:].std().item(), last=s[-1].item()))
    use(None); collect = grab2
    with t.no_grad():
        for ids in items: model(input_ids=t.tensor([ids], device="cuda"))
    collect = None
    comp[row] = {L: dict(pin=c[L].item(), pos0=sum(d["pos0"] for d in v) / len(v), rest_mean=sum(d["rest_mean"] for d in v) / len(v), rest_sd=sum(d["rest_sd"] for d in v) / len(v), last=sum(d["last"] for d in v) / len(v)) for L, v in per.items()}
    show = [L for L in window if L in (28, 37, 46, 55, 0, 30, 59)][:4]
    print(f"{row}: " + "; ".join(f"L{L} pin {comp[row][L]['pin']:+.1f}, comp pos0 {comp[row][L]['pos0']:+.1f}, rest {comp[row][L]['rest_mean']:+.1f} +- {comp[row][L]['rest_sd']:.1f}, last {comp[row][L]['last']:+.1f}" for L in show))
res["components"] = comp
# 2. first-token KL on the 100 harmless test prompts, each hook with every position vs position 0 exempt
kl_items = [render(p) for p in PROMPTS["harmless"]["test"]]
def first_logps():
    out = []
    with t.no_grad():
        for ids in kl_items: out.append(t.log_softmax(model(input_ids=t.tensor([ids], device="cuda")).logits[0, -1].float(), -1).cpu())
    return t.stack(out)
use(None); base = first_logps()
kls = {}
for row in ("prand", "zrand", "refdom", "gpdom"):
    for skip in (False, True):
        use(row, skip); lp = first_logps()
        kls[f"{row}_{'skip_bos' if skip else 'all'}"] = (base.exp() * (base - lp)).sum(-1).mean().item()
        print(f"{row} {'position 0 exempt' if skip else 'every position'}: first-token KL {kls[f'{row}_{chr(115) + chr(107) + chr(105) + chr(112) + chr(95) + chr(98) + chr(111) + chr(115)}' if skip else f'{row}_all']:.4f}")
use(None)
res["first_token_kl"] = kls
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "bos_diag.json").write_text(json.dumps(res, indent=1))
print(f"saved {OUT / 'bos_diag.json'}")
