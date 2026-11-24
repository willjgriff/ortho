"""HF numerics diagnostic: TV rollout 0 of the served check, under the gprand hook, through HF in bf16 (sdpa), bf16 eager attention, and fp32; per-token log-probs compared with vLLM plain and at the suspicious positions."""
import json, sys, torch as t, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText
sys.path.insert(0, "/root/ortho/scripts"); sys.argv = ["x", "list"]
exec(open("/root/ortho/scripts/prompt_dom_served_check.py").read().split("if sys.argv[1] ==")[0])
xs = items(); x = xs[0]; ids = x["ids"]
served = json.loads(open("/root/ortho/acts/prompt_dom/served_plain.json").read())["logps"][0]
hf_prev = json.loads(open("/root/ortho/acts/prompt_dom/served_hf_gprand.json").read())["logps"][0]
h = load_file("/root/hooks/gprand.safetensors"); window = h["layers"].tolist(); U = h["u"].float(); C = h["c"].float()
MODEL_ID, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
def run(dtype, attn, hook=True):
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REV, dtype=dtype, device_map="cuda", attn_implementation=attn).eval()
    layers = model.model.language_model.layers
    u, c = U.cuda(), C.cuda()
    def mk(L):
        def hk(m, a, out):
            hh = out[0] if isinstance(out, tuple) else out
            if hook and L in window:
                s = hh.float() @ u[window.index(L)] - c[L]
                hh = hh - (s[..., None] * u[window.index(L)]).to(hh.dtype)
            return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
        return hk
    hs = [layers[L].register_forward_hook(mk(L)) for L in range(64)]
    with t.no_grad(): logits = model(input_ids=t.tensor([ids], device="cuda")).logits[0, :-1].float()
    lp = t.log_softmax(logits, -1).gather(-1, t.tensor(ids[1:], device="cuda")[:, None])[:, 0].cpu()
    for hh in hs: hh.remove()
    del model, logits; t.cuda.empty_cache()
    return lp
res = {}
for name, dtype, attn in [("bf16_sdpa", t.bfloat16, "sdpa"), ("bf16_eager", t.bfloat16, "eager"), ("fp32_sdpa", t.float32, "sdpa")]:
    try:
        res[name] = run(dtype, attn); print(name, "done", flush=True)
    except Exception as e:
        print(name, "FAILED", type(e).__name__, str(e)[:200], flush=True); t.cuda.empty_cache()
vl = t.tensor(served); prev = t.tensor(hf_prev)
print(f"prev HF(bf16 batched) vs vLLM plain: {(prev - vl).abs().mean():.4f}")
for k, v in res.items():
    print(f"{k}: vs vLLM plain {(v - vl).abs().mean():.4f} (max {(v - vl).abs().max():.2f}); vs prev HF {(v - prev).abs().mean():.4f}")
for pos in (3113, 3282, 3362, 2908, 3299):
    j = pos - 1
    print(f"pos {pos}: vLLM {vl[j]:+.2f} prevHF {prev[j]:+.2f} " + " ".join(f"{k} {v[j]:+.2f}" for k, v in res.items()))
if "fp32_sdpa" in res and "bf16_sdpa" in res:
    d = (res["bf16_sdpa"] - res["fp32_sdpa"]).abs(); print(f"bf16 vs fp32 HF: mean {d.mean():.4f}, max {d.max():.2f}, frac>0.5 {(d > 0.5).float().mean():.3f}; fp32 vs vLLM frac>0.5 {((res['fp32_sdpa'] - vl).abs() > 0.5).float().mean():.3f}")
