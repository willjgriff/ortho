"""Diagnostic 2: (1) fp32 vs fp32+TF32 on TV rollout 0 (speed/accuracy trade); (2) the TV fit's last-token residuals in fp32 vs the bf16 acts for the first 60 train pairs: per-layer cosine of the
60-pair difference-of-means direction and per-prompt residual cosines; (3) the Grader Arditi base and chosen-candidate bypass scores in fp32 vs bf16."""
import json, sys, time, torch as t, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText
sys.path.insert(0, "/root/ortho/scripts"); sys.argv = ["x", "list"]
exec(open("/root/ortho/scripts/prompt_dom_served_check.py").read().split("if sys.argv[1] ==")[0])
MODEL_ID, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
xs = items(); ids0 = xs[0]["ids"]
served = t.tensor(json.loads(open("/root/ortho/acts/prompt_dom/served_plain.json").read())["logps"][0])
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REV, dtype=t.float32, device_map="cuda").eval()
layers = model.model.language_model.layers
outs = [None] * 64
def mk(L):
    def hk(m, a, out): outs[L] = (out[0] if isinstance(out, tuple) else out)
    return hk
for L in range(64): layers[L].register_forward_hook(mk(L))
def lp(ids):
    with t.no_grad(): logits = model(input_ids=t.tensor([ids], device="cuda")).logits[0, :-1].float()
    return t.log_softmax(logits, -1).gather(-1, t.tensor(ids[1:], device="cuda")[:, None])[:, 0].cpu()
def last(ids):
    with t.no_grad(): model(input_ids=t.tensor([ids], device="cuda"))
    return t.stack([o[0, -1].float() for o in outs]).cpu()   # [64, d]
# (1) done in the first run: fp32 6.5 s vs tf32 1.2 s per 4,698-token forward, |fp32 - tf32| mean 0.004, max 0.30 -> TF32 on
use_tf32 = True
t.backends.cuda.matmul.allow_tf32 = True; t.backends.cudnn.allow_tf32 = True
print('using tf32:', use_tf32, flush=True)
# (2) TV fit check on 60 pairs
acts = t.load("/root/ortho/acts/prompt_dom/acts_tv.pt")
pairs = t.load("/root/ortho/acts/prompt_dom/tv_pairs.pt")["train"][:60]
t0 = time.time()
pos32 = t.stack([last(p["impossible"]) for p in pairs]); neg32 = t.stack([last(p["possible"]) for p in pairs])
print(f"(2) 120 TV prompts in {time.time()-t0:.0f} s", flush=True)
pos16, neg16 = acts["fit_pos"][:60, -1].float(), acts["fit_neg"][:60, -1].float()
d32 = F.normalize(pos32.mean(0) - neg32.mean(0), dim=1); d16 = F.normalize(pos16.mean(0) - neg16.mean(0), dim=1)
cos_dir = F.cosine_similarity(d32, d16, dim=1)
cos_prompt = F.cosine_similarity(pos32, pos16, dim=-1)   # [60, 64]
full = load_file("/root/hooks/tvpdomall.safetensors")["u"].float()   # the 400-pair bf16 direction per layer
cos_full = F.cosine_similarity(d32, full, dim=1)
print("(2) 60-pair direction, fp32 vs bf16 acts, cosine by layer: " + ", ".join(f"L{L} {cos_dir[L]:.3f}" for L in range(0, 64, 6)) + f"; min over L30-63 {cos_dir[30:].min():.3f}")
print("(2) 60-pair fp32 direction vs the 400-pair bf16 direction (tvpdomall): " + ", ".join(f"L{L} {cos_full[L]:.3f}" for L in range(30, 64, 6)) + f"; mean L30-63 {cos_full[30:].mean():.3f}")
print(f"(2) per-prompt last-token residual cosine fp32 vs bf16 (impossible members): min {cos_prompt.min():.4f}, mean {cos_prompt.mean():.5f}, by layer min: " + ", ".join(f"L{L} {cos_prompt[:, L].min():.4f}" for L in range(0, 64, 8)))
rel = (pos32 - pos16).norm(dim=-1) / pos32.norm(dim=-1)
print(f"(2) relative residual error of bf16 per prompt: median {rel.median():.4f}, 95th {rel.quantile(0.95):.4f}, max {rel.max():.4f} (over prompts and layers)", flush=True)
# (3) grader arditi scores in fp32: base and chosen candidate
sys.path.insert(0, "/root/ortho")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
g = json.load(open("/root/ortho/acts/prompt_dom/arditi_grader.json"))
gacts = t.load("/root/ortho/acts/prompt_dom/acts_grader.pt")
val = [x for x in gacts["items"]["val_pos"]]
import yaml
fams = yaml.safe_load(open("/root/ortho/prompt_dom/grader_fit_families.yaml"))
from rhenvs.grader import prompt as gprompt
def render(user):
    return tok(tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False, tokenize=False), add_special_tokens=False).input_ids
def ans_ids(a, kind): return tok(a if kind == "int" else a[:1].upper() + a[1:], add_special_tokens=False).input_ids
hook_u, hook_on = None, False
def mk2(L):
    def hk(m, a, out):
        hh = out[0] if isinstance(out, tuple) else out
        if hook_on: hh = hh - ((hh @ hook_u)[..., None] * hook_u)
        return (hh,) + tuple(out[1:]) if isinstance(out, tuple) else hh
    return hk
for L in range(64): layers[L].register_forward_hook(mk2(L))
def score(items_):
    tot = []
    for x in items_:
        ids = render(gprompt(fams[x["family"]], x["side"], x["kind"]))
        def lpsum(ans):
            seq = ids + ans
            with t.no_grad(): logits = model(input_ids=t.tensor([seq], device="cuda")).logits[0, len(ids)-1:-1].float()
            return t.log_softmax(logits, -1).gather(-1, t.tensor(ans, device="cuda")[:, None]).sum().item()
        tot.append(lpsum(ans_ids(x["other"], x["answer_kind"])) - lpsum(ans_ids(x["own"], x["answer_kind"])))
    return sum(tot) / len(tot)
base32 = score(val)
ch = g["chosen"]; diffs = gacts["fit_pos"].float().mean(0) - gacts["fit_neg"].float().mean(0)
hook_u = F.normalize(diffs[ch["position"] + 5, ch["layer"]], dim=0).cuda(); hook_on = True
chosen32 = score(val); hook_on = False
print(f"(3) grader Arditi bypass in fp32: base {base32:+.3f} (bf16 {g['base_bypass']:+.3f}), chosen pos {ch['position']} L{ch['layer']}: {chosen32:+.3f} (bf16 {ch['bypass']:+.3f})")
