#!./.venv/bin/python
"""Part 2 of the KTO refusal control (sweep/mission_kto_orthog_refusal.md): a kto_refusal.py direction as a weight edit, served with no hook. The trained hook
(kto_refusal.py, scripts/vllm_proj.py) sets h <- h - ((h - mu_L).u) u at the output of each decoder layer L, so the component of the residual along the shared unit
direction u becomes c_L = mu_L.u (the hook file's c). Two weight forms of the same u, both built from the base weights:

  zero form (_wz, Arditi et al.)   every matrix that writes into the residual stream loses its component along u: the token embedding row-wise, E <- E - (E u) u^T, and
                                   each decoder layer's attention output projection (self_attn.o_proj in full-attention layers, linear_attn.out_proj in the GatedDeltaNet
                                   layers) and mlp.down_proj, W <- (I - u u^T) W. The residual then has component 0 along u at every layer and position.
  mean-pinned form (_wm)           the zero form plus a bias (c_L - c_{L-1}) u on each layer's mlp.down_proj (c_{-1} = 0), so the component at the output of layer L is
                                   c_L, as under the hook. The only difference from the hook is inside a layer: the hook leaves the attention write along u in place until
                                   the layer output, the edit removes it.

A row's checkpoint directory (~/models/cb_<row>) symlinks the base snapshot's files, adds the edited tensors as one shard shared by the two forms of a direction
(~/models/edits/<direction>.safetensors, ~18 GB), its own bias shard (model-resid-bias.safetensors, zero for _wz and base_w, served by Will's vllm_resid_bias plugin
with RHABLATION_RESID_BIAS=1) and a rewritten model.safetensors.index.json, plus a manifest. GPU box, vLLM stopped for build.

    ./kto_orthog.py build <row> ...   rows of EDIT_ROWS: edit the writers of the loaded base model in place, write the checkpoint, then in HF: the mechanical check (component
                                      along u at every layer output on 20 harmless test prompts, against the base model and the hook), first-token KL from base on heretic's
                                      test prompts, MMLU (every fourth test question) against base, the held-out screen (gains, AUC) and the held-out harmless neutral KL,
                                      and the per-token log-probs of 50 held-out harmless answers for the served check; to acts/kto_orthog/<row>.json and <row>_held.pt
    ./kto_orthog.py mech <row> ...    the mechanical check again for built rows (the edit recomputed in memory), acts/kto_orthog/<row>.json updated in place
    ./kto_orthog.py served <row> ...  the row served on localhost:8000 as cb_<row>: teacher-forced prompt_logprobs of those 50 answers against the HF log-probs (mean |difference|
                                      per token <= 0.005 nats passes), added to acts/kto_orthog/<row>.json
    ./kto_orthog.py table             the SUMMARY tables of this mission (refusal rates from the records, the checks of every row), to acts/kto_orthog/table.md and rates.json
"""
import hashlib
import os
import sys

STAGE, ARG_ROWS = (sys.argv[1] if len(sys.argv) > 1 else "list"), sys.argv[2:]
# the rows: served name -> (the hook file of the direction, the form)
EDIT_ROWS = dict(base_w=(None, None), rkfull_wm=("rkfull", "wm"), rkfull_wz=("rkfull", "wz"), rkfullrand_wm=("rkfullrand", "wm"), rkfullrand_wz=("rkfullrand", "wz"), rktoall_wz=("rktoall", "wz"),
                 probe_w=("rkfullrand", "probe"))   # the bias-plugin probe: base weights plus rkfullrand_wm's biases times PROBE_SCALE, no edit; served vs HF must differ from base by far more than the numerics floor
assert all(r in EDIT_ROWS for r in ARG_ROWS), f"rows are {list(EDIT_ROWS)}, not {[r for r in ARG_ROWS if r not in EDIT_ROWS]}"
if STAGE == "list":
    for r, (d, f) in EDIT_ROWS.items(): print(r, d, f)
    sys.exit()
if STAGE == "table":
    import kto_orthog_table
    kto_orthog_table.main()
    sys.exit()

sys.argv = ["kto_refusal.py", "sets"]
exec(open("kto_refusal.py").read().split("#%% export")[0])   # kto_refusal.py's constants, the rollout sets (refuse/comply/neutral, train and held out) and first_person; prints the counts
from torch import nn

OUT2 = Path("acts/kto_orthog")
MODELS = Path.home() / "models"
EDITS = MODELS / "edits"
ORTHOG_SHARD, BIAS_SHARD = "model-orthog.safetensors", "model-resid-bias.safetensors"
PREFIX = "model.language_model."
N_MECH, N_SERVED = 20, 50
SERVED_TOL = 0.005   # nats per token, mean |served - HF|: the mission's tolerance. It is below the HF-vs-vLLM bf16 numerics floor measured on the unedited weights (base_w: 0.0115, signed +0.0004), so the
# rule applied is: floor = base_w's own mean |served - HF|; a row passes when its mean |served - HF_row| is within 2 x floor on the 50 harmless answers and on the 200 held-out refusing / compliant
# answers (summed per rollout, HF sum = base reference + the row's screened gain), and the probe row's served log-probs must be farther than 2 x floor from HF base_w (the biases are being served)
PROBE_SCALE = 20.0
OUT2.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def hook_file(direction: str) -> Path: return HOOKS / f"{direction}.safetensors"
def served_items() -> list[dict]: return sorted(neutral_held, key=crc)[:N_SERVED]   # 50 held-out harmless answers, fixed by key hash

#%% served: the server's teacher-forced log-probs against the stored HF ones; no model
if STAGE == "served":
    import requests
    from concurrent.futures import ThreadPoolExecutor
    base = t.load(OUT / "base.pt")
    held_by_key = {x["key"]: x for x in held}
    def score(row: str, x: dict) -> list[float]:
        r = requests.post("http://localhost:8000/v1/completions", json=dict(model=f"cb_{row}", prompt=x["ids"], max_tokens=1, temperature=0, prompt_logprobs=0), timeout=900)
        assert r.status_code == 200, r.text
        lp = r.json()["choices"][0]["prompt_logprobs"]
        return [lp[j][str(x["ids"][j])]["logprob"] for j in range(x["start"], len(x["ids"]))]
    def mean_abs(a: list[list[float]], b: list[list[float]]) -> tuple[float, float, float]:
        d = np.concatenate([np.array(p) - np.array(q) for p, q in zip(a, b)])
        return float(np.abs(d).mean()), float(d.mean()), float(np.abs(d).max())
    stored = {p.stem[:-len("_held")]: t.load(p) for p in OUT2.glob("*_held.pt")}
    infos = {r: json.loads((OUT2 / f"{r}.json").read_text()) for r in EDIT_ROWS if (OUT2 / f"{r}.json").exists()}
    floor = (infos.get("base_w", {}).get("served_vs_hf") or {}).get("harmless_mean_abs")   # base_w's own served-vs-HF: the numerics floor (None while base_w itself is checked)
    for row in ARG_ROWS:
        mine = stored[row]
        with ThreadPoolExecutor(8) as ex: served = list(ex.map(lambda x: score(row, x), mine["items"]))
        harmless = {other: mean_abs(served, s_["logps"]) for other, s_ in stored.items() if [x["key"] for x in s_["items"]] == [x["key"] for x in mine["items"]]}
        own_abs, own_signed, own_max = harmless[row]
        res = dict(n_items=len(served), n_tokens=int(sum(map(len, served))), harmless_mean_abs=own_abs, harmless_signed=own_signed, harmless_max_abs=own_max, harmless_vs_hf_rows={o: v[0] for o, v in harmless.items()}, tol_mission=SERVED_TOL)
        # the 200 held-out answers, summed per rollout, against every row that screened them (HF sum = base reference + gain)
        if "held" in infos.get(row, {}):
            keys = infos[row]["held"]["keys"]
            items = [held_by_key[k] for k in keys]
            with ThreadPoolExecutor(8) as ex: sums = np.array([sum(lp) for lp in ex.map(lambda x: score(row, x), items)])
            n_tok = sum(x["n"] for x in items)
            ref = np.array([base["ref"][k] for k in keys])
            per_row = {o: float(np.abs(sums - (ref + np.array(i["held"]["gains"]))).sum() / n_tok) for o, i in infos.items() if "held" in i and i["held"]["keys"] == keys}
            res |= dict(held_n=len(items), held_tokens=n_tok, held_mean_abs_per_token=per_row[row], held_signed_per_token=float((sums - (ref + np.array(infos[row]["held"]["gains"]))).sum() / n_tok), held_vs_hf_rows=per_row, held_served_sum=float(sums.sum()), held_hf_sum=float((ref + np.array(infos[row]["held"]["gains"])).sum()))
        f = floor if (floor is not None and row != "base_w") else own_abs
        tol = 2 * f
        checks = dict(harmless=own_abs <= tol, held=res["held_mean_abs_per_token"] <= tol if "held_mean_abs_per_token" in res else None)
        if row == "base_w": checks = dict(floor=own_abs <= 0.02 and abs(own_signed) <= 0.002, held=checks["held"])   # the floor itself: bf16 numerics, no systematic offset
        if infos.get(row, {}).get("form") == "probe": checks = dict(harmless=checks["harmless"], visible=harmless.get("base_w", (0,))[0] > tol)   # the biases must move the served log-probs well past the floor
        nearest = min(harmless, key=lambda o: harmless[o][0])
        res |= dict(floor=f, tol=tol, checks=checks, nearest_hf_row=nearest, passed=all(v for v in checks.values() if v is not None))
        path = OUT2 / f"{row}.json"
        info = infos.get(row, dict(row=row))
        info["served_vs_hf"] = res
        path.write_text(json.dumps(info, indent=1))
        held_line = f"; held-out 200 answers {res['held_mean_abs_per_token']:.4f} nats/token (signed {res['held_signed_per_token']:+.4f}; vs other HF rows " + ", ".join(f"{o} {v:.4f}" for o, v in res["held_vs_hf_rows"].items() if o != row) + ")" if "held_mean_abs_per_token" in res else ""
        print(f"{green if res['passed'] else red}{row}: served vs HF, harmless 50 answers ({res['n_tokens']} tokens): mean |diff| {own_abs:.4f} nats/token (signed {own_signed:+.4f}, max {own_max:.3f}; vs other HF rows " + ", ".join(f"{o} {v[0]:.4f}" for o, v in harmless.items() if o != row) + f"; nearest {nearest}){held_line}; floor {f:.4f}, tol {tol:.4f}, checks {checks}: {'PASS' if res['passed'] else 'FAIL'}{endc}")
    sys.exit()

#%% build: the base model once; each row edits the writers in place
assert STAGE in ("build", "mech"), STAGE   # mech: the mechanical check of built rows again (acts/kto_orthog/<row>.json updated in place), no checkpoint written, no HF checks
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
PAD = tok.pad_token_id
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda")
model.requires_grad_(False)
model.eval()
text = model.model.language_model
layers = text.layers
assert len(layers) == 64 and model.config.text_config.hidden_size == D_MODEL
tied = bool(getattr(model.config.text_config, "tie_word_embeddings", False)) or model.lm_head.weight.data_ptr() == text.embed_tokens.weight.data_ptr()
if tied:   # untie: the unembedding keeps the base weights, the embedding is edited
    model.lm_head.weight = nn.Parameter(text.embed_tokens.weight.detach().clone(), requires_grad=False)
    print(f"{yellow}tie_word_embeddings: untied, lm_head keeps the base weights{endc}")
from huggingface_hub import snapshot_download
snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION))   # the cached base snapshot the checkpoint directories link to
assert (snapshot / "model.safetensors.index.json").exists(), snapshot
base_index = json.loads((snapshot / "model.safetensors.index.json").read_text())

# down_proj with a bias (zero = the base model), the module the bias shard and the plugin put it on
for layer in layers:
    old = layer.mlp.down_proj
    new = nn.Linear(old.in_features, old.out_features, bias=True, device="cuda", dtype=t.bfloat16)
    new.weight = old.weight
    new.bias.data.zero_()
    layer.mlp.down_proj = new
# the writers into the residual stream: checkpoint tensor name -> parameter
W = {f"{PREFIX}embed_tokens.weight": text.embed_tokens.weight}
for L, layer in enumerate(layers):
    if hasattr(layer, "linear_attn"): W[f"{PREFIX}layers.{L}.linear_attn.out_proj.weight"] = layer.linear_attn.out_proj.weight
    else: W[f"{PREFIX}layers.{L}.self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
    W[f"{PREFIX}layers.{L}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight
missing = [n for n in W if n not in base_index["weight_map"]]
assert not missing, f"writer names not in the checkpoint index: {missing[:5]}"
assert all(p.shape[0] == D_MODEL for n, p in W.items() if "embed" not in n) and W[f"{PREFIX}embed_tokens.weight"].shape[1] == D_MODEL
other_writers = [n for n in base_index["weight_map"] if n.startswith(PREFIX + "layers.") and n.endswith(".weight") and any(k in n for k in ("o_proj", "out_proj", "down_proj")) and n not in W]
assert not other_writers, other_writers   # every residual writer of the decoder layers is edited; mtp.* (the draft layers) are left alone, as the hook leaves them
orig = {n: p.detach().to("cpu", copy=True) for n, p in W.items()}
print(f"{cyan}{len(W)} writer matrices ({sum(p.numel() for p in W.values()) / 1e9:.2f}B parameters), {sum(hasattr(l, 'linear_attn') for l in layers)} linear-attention and {sum(hasattr(l, 'self_attn') for l in layers)} full-attention layers; tied embeddings: {tied}{endc}")

def edited_tensors(u: Tensor) -> dict[str, Tensor]:
    """Every writer with its component along u removed, computed in float32 from the base weights, stored bf16 on the CPU."""
    out = {}
    for name, w in pbar(orig.items(), desc="edit"):
        e = w.cuda().float()
        if "embed_tokens" in name: e -= (e @ u)[:, None] * u[None]   # rows are token vectors
        else: e -= u[:, None] * (u @ e)[None]   # columns write into the residual
        out[name] = e.to(t.bfloat16).cpu()
        del e
    return out

def apply(src: dict[str, Tensor] | None, bias: Tensor | None) -> None:
    """The model's writers set to src (None = the base weights) and its down_proj biases to bias [64, d_model] (None = zero)."""
    for name, p in W.items(): p.data.copy_((orig if src is None else src)[name], non_blocking=True)
    for L, layer in enumerate(layers): layer.mlp.down_proj.bias.data.copy_(bias[L].to(t.bfloat16) if bias is not None else t.zeros(D_MODEL))
    t.cuda.synchronize()

# the trained hook in HF, for the mechanical check's bf16 reference
hook_on, hook_u, hook_c, hook_layers = False, None, None, ()
collect = None
def layer_hook(L: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if hook_on and L in hook_layers:
            s = h.float() @ hook_u - hook_c[L]
            h = h - (s[..., None] * hook_u).to(h.dtype)
        if collect: collect(L, h)   # the layer's output as the next layer sees it: after the hook's edit when it is on
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for L in range(64): layers[L].register_forward_hook(layer_hook(L))

system = load_config("refusal", "nothink")["system_prompt"]
def render(prompt: str) -> list[int]:
    text_ = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text_, add_special_tokens=False).input_ids
    assert ids[-4:] == [THINK_OPEN, BLANK, THINK_CLOSE, BLANK], tok.decode(ids[-6:])
    return ids

def forward(batch: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
    """Next-token logits of a right-padded batch [batch, seq - 1, vocab], the next ids, and which of those are generated tokens."""
    n = max(len(x["ids"]) for x in batch)
    ids = t.tensor([x["ids"] + [PAD] * (n - len(x["ids"])) for x in batch], device="cuda")
    pos = t.arange(n, device="cuda")[None]
    attn = pos < t.tensor([len(x["ids"]) for x in batch], device="cuda")[:, None]
    gen = attn & (pos >= t.tensor([x["start"] for x in batch], device="cuda")[:, None])
    logits = model(input_ids=ids, attention_mask=attn).logits
    return logits[:, :-1].float(), ids[:, 1:], gen[:, 1:]
def batches(items: list[dict], max_tokens: int) -> list[list[dict]]:
    out = []
    for x in items:
        if out and (len(out[-1]) + 1) * max(len(y["ids"]) for y in out[-1] + [x]) <= max_tokens: out[-1].append(x)
        else: out.append([x])
    return out
def completion_logps(batch: list[dict]) -> Tensor:
    logits, following, completion = forward(batch)
    return (-F.cross_entropy(logits.transpose(1, 2), following, reduction="none") * completion).sum(1)
def token_logps(x: dict) -> list[float]:
    """log p of each generated token of one rollout, positions start .. end."""
    logits, following, completion = forward([x])
    lp = -F.cross_entropy(logits[0], following[0], reduction="none")
    return lp[x["start"] - 1:].tolist()

def components(prompts: list[str], u: Tensor) -> dict:
    """At every layer output over every position of the rendered prompts: the mean component along u, its mean absolute value, and the mean residual norm; the raw components kept for the check."""
    global collect
    comps = {L: [] for L in range(64)}
    norms = {L: [] for L in range(64)}
    def add(L: int, h: Tensor):
        comps[L].append((h[0].float() @ u).cpu())
        norms[L].append(h[0].float().norm(dim=-1).cpu())
    collect = add
    with t.no_grad():
        for p in prompts: model(input_ids=t.tensor([render(p)], device="cuda"))
    collect = None
    return dict(comp={L: t.cat(comps[L]) for L in range(64)}, norm={L: t.cat(norms[L]).mean().item() for L in range(64)})

def first_logps(prompts: list[str], desc: str) -> Tensor:
    with t.no_grad(): return t.stack([t.log_softmax(model(input_ids=t.tensor([render(p)], device="cuda")).logits[0, -1].float(), -1).cpu() for p in pbar(prompts, desc=desc)])

from datasets import load_dataset
mmlu = list(load_dataset("cais/mmlu", "all", split="test"))[::MMLU_EVERY]
def mmlu_prompt(x: dict) -> str:
    choices = "".join(f"{letter}. {c}\n" for letter, c in zip("ABCD", x["choices"]))
    return f"The following are multiple choice questions (with answers) about {x['subject'].replace('_', ' ')}.\n\n{x['question'].strip()}\n{choices}Answer:"
letters = [tok(f" {c}", add_special_tokens=False).input_ids for c in "ABCD"]
assert all(len(ids) == 1 for ids in letters), letters
letters = [ids[0] for ids in letters]
order = sorted(range(len(mmlu)), key=lambda k: len(mmlu_prompt(mmlu[k])), reverse=True)
tok.padding_side = "right"
def mmlu_correct(desc: str) -> list[bool]:
    correct = [False] * len(mmlu)
    for s in pbar(range(0, len(order), MMLU_BATCH), desc=f"mmlu {desc}"):
        batch = order[s:s + MMLU_BATCH]
        enc = tok([mmlu_prompt(mmlu[k]) for k in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with t.no_grad(): picks = model.lm_head(model.model(**enc).last_hidden_state[t.arange(len(batch)), enc["attention_mask"].sum(1) - 1])[:, letters].argmax(-1).tolist()
        for k, pick in zip(batch, picks): correct[k] = pick == mmlu[k]["answer"]
    return correct
def paired(a: list[bool], b: list[bool]) -> dict:
    d = t.tensor(b).float() - t.tensor(a).float()
    half = 1.96 * d.std().item() / len(d) ** .5
    return dict(points=100 * d.mean().item(), low=100 * (d.mean().item() - half), high=100 * (d.mean().item() + half))

# the base model's references, once per box: first-token log-probs on heretic's test prompts, MMLU, the per-token log-probs of the served-check answers
base = t.load(OUT / "base.pt")
ref_path = OUT2 / "base_ref.pt"
apply(None, None)
if ref_path.exists():
    base_ref = t.load(ref_path)
    print(f"{gray}base references from {ref_path}{endc}")
else:
    base_ref = dict(first_harmless=first_logps(PROMPTS["harmless"]["test"], "first token, base, harmless"), first_harmful=first_logps(PROMPTS["harmful"]["test"], "first token, base, harmful"), mmlu=mmlu_correct("base"))
    t.save(base_ref, ref_path)
    print(f"{cyan}base MMLU {100 * sum(base_ref['mmlu']) / len(mmlu):.2f} of {len(mmlu)}; saved {ref_path}{endc}")
mech_prompts = PROMPTS["harmless"]["test"][:N_MECH]
mech_cache = {}   # direction -> base and hook components along that direction

def write_checkpoint(row: str, direction: str | None, edited: dict[str, Tensor] | None, bias: Tensor, extra: dict) -> Path:
    out = MODELS / f"cb_{row}"
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(snapshot.iterdir()):
        if p.name in (ORTHOG_SHARD, BIAS_SHARD, "model.safetensors.index.json", "manifest.json") or p.name.startswith("."): continue
        dst = out / p.name
        if dst.is_symlink() or dst.exists(): dst.unlink()
        os.symlink(p.resolve(), dst)
    index = dict(metadata=dict(base_index.get("metadata", {})), weight_map={k: v for k, v in base_index["weight_map"].items()})
    if edited is not None:
        EDITS.mkdir(parents=True, exist_ok=True)
        shard = EDITS / f"{direction}.safetensors"
        if not shard.exists():
            save_file({k: v.contiguous() for k, v in edited.items()}, shard, metadata={"format": "pt"})
            print(f"{cyan}wrote {shard} ({shard.stat().st_size / 1e9:.1f} GB){endc}")
        dst = out / ORTHOG_SHARD
        if dst.is_symlink() or dst.exists(): dst.unlink()
        os.symlink(shard.resolve(), dst)
        for k in edited: index["weight_map"][k] = ORTHOG_SHARD
    biases = {f"{PREFIX}layers.{L}.mlp.down_proj.bias": bias[L].to(t.bfloat16).contiguous() for L in range(64)}
    save_file(biases, out / BIAS_SHARD, metadata={"format": "pt"})
    for k in biases: index["weight_map"][k] = BIAS_SHARD
    index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + sum(v.numel() * v.element_size() for v in biases.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(row=row, direction=direction, base=str(snapshot), model=MODEL_ID, revision=REVISION, edited_shard=str(shard) if edited is not None else None, writers=sorted(edited) if edited is not None else [], bias_shard=BIAS_SHARD, bias_norms=[bias[L].norm().item() for L in range(64)], created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())) | extra, indent=1))
    return out

edits = {}   # direction -> edited tensors (CPU)
for row in ARG_ROWS:
    started = time.time()
    direction, form = EDIT_ROWS[row]
    info = dict(row=row, direction=direction, form=form, model=MODEL_ID, revision=REVISION)
    if direction is None:   # base_w: the base weights, zero biases
        u, c, hl = None, t.zeros(64), ()
        bias = t.zeros(64, D_MODEL)
        apply(None, bias)
        info |= dict(hook_file=None)
    else:
        hf = load_file(hook_file(direction))
        u, c, hl = hf["u"].float().cuda(), hf["c"].float(), hf["layers"].tolist()
        assert abs(u.norm().item() - 1) < 1e-4 and c.shape == (64,)
        info |= dict(hook_file=str(hook_file(direction)), direction_sha256=sha256(hook_file(direction)), hook_layers=[hl[0], hl[-1], len(hl)], c=c.tolist())
        if form == "probe":   # base weights + PROBE_SCALE x the mean-pinned biases of this direction: only the served-check log-probs (and their distance from HF base_w)
            bias = t.zeros(64, D_MODEL)
            for L in range(64): bias[L] = PROBE_SCALE * (c[L] - (c[L - 1] if L else 0.0)) * u.cpu()
            apply(None, bias)
            out = write_checkpoint(row, None, None, bias, dict(form=form, probe_scale=PROBE_SCALE, direction_sha256=info["direction_sha256"], c=c.tolist()))
            with t.no_grad():
                items = served_items()
                logps = [token_logps(x) for x in pbar(items, desc=f"served-check log-probs {row}")]
            t.save(dict(items=[dict(key=x["key"], ids=x["ids"], start=x["start"]) for x in items], logps=logps), OUT2 / f"{row}_held.pt")
            base_held = t.load(OUT2 / "base_w_held.pt")["logps"] if (OUT2 / "base_w_held.pt").exists() else None
            info |= dict(bias_norms=[bias[L].norm().item() for L in range(64)], probe_scale=PROBE_SCALE, checkpoint=str(out), hf_probe_vs_hf_base_mean_abs=float(np.abs(np.concatenate([np.array(a) - np.array(b) for a, b in zip(logps, base_held)])).mean()) if base_held else None, seconds=time.time() - started)
            (OUT2 / f"{row}.json").write_text(json.dumps(info, indent=1))
            print(f"{green}{row}: checkpoint {out}; HF probe vs HF base_w on the 50 answers: mean |diff| {info['hf_probe_vs_hf_base_mean_abs']} nats/token; saved {OUT2 / f'{row}.json'}{endc}")
            apply(None, None)
            continue
        if direction not in edits:
            for d in [d for d in edits if d != direction]: del edits[d]
            edits[direction] = edited_tensors(u)
        target = c.clone() if form == "wm" else t.zeros(64)
        bias = t.zeros(64, D_MODEL)
        if form == "wm":
            for L in range(64): bias[L] = (c[L] - (c[L - 1] if L else 0.0)) * u.cpu()
        # the mechanical check: the base model's and the hook's components along u, cached per direction
        if direction not in mech_cache:
            apply(None, None)
            base_comp = components(mech_prompts, u)
            hook_on, hook_u, hook_c, hook_layers = True, u, c.cuda(), tuple(hl)
            hook_comp = components(mech_prompts, u)
            hook_on = False
            mech_cache[direction] = (base_comp, hook_comp)
        base_comp, hook_comp = mech_cache[direction]
        apply(edits[direction], bias)
        edit_comp = components(mech_prompts, u)
        hook_dev = {L: (hook_comp["comp"][L] - c[L]).abs().mean().item() for L in hl}   # the hook's own distance from c_L: bf16 residual precision
        tol_floor = max(np.median(list(hook_dev.values())), 1e-3)
        per_layer = []
        for L in range(64):
            e, b = edit_comp["comp"][L], base_comp["comp"][L]
            dev = (e - target[L]).abs()
            tol = max(3 * hook_dev.get(L, tol_floor), 0.05)
            per_layer.append(dict(layer=L, target=target[L].item(), edit_mean=e.mean().item(), edit_mean_abs_dev=dev.mean().item(), edit_max_abs_dev=dev.max().item(), base_mean=b.mean().item(), base_mean_abs=b.abs().mean().item(), base_rms=b.pow(2).mean().sqrt().item(), hook_mean_abs_dev=hook_dev.get(L), resid_norm=edit_comp["norm"][L], tol=tol, ok=bool(dev.mean().item() <= tol)))
        info["mechanical"] = dict(prompts=N_MECH, positions=int(edit_comp["comp"][0].numel()), rule="mean |component - target| <= max(3 x the hook's own mean |component - c_L| at that layer (median over hooked layers where unhooked), 0.05)", per_layer=per_layer, passed=all(p["ok"] for p in per_layer), worst=max(p["edit_mean_abs_dev"] / p["tol"] for p in per_layer))
        print(f"{green if info['mechanical']['passed'] else red}{row} mechanical check: {'PASS' if info['mechanical']['passed'] else 'FAIL'} (worst layer at {info['mechanical']['worst']:.2f} of its tolerance); mean |comp - target| by layer: " + ", ".join(f"L{p['layer']} {p['edit_mean_abs_dev']:.3f} (base |comp| {p['base_mean_abs']:.2f}, target {p['target']:+.2f})" for p in per_layer[::9]) + endc)
    if STAGE == "mech":
        path = OUT2 / f"{row}.json"
        prev = json.loads(path.read_text()) if path.exists() else {}
        prev |= dict(mechanical=info.get("mechanical"), c=info.get("c"), hook_layers=info.get("hook_layers"))
        path.write_text(json.dumps(prev, indent=1))
        print(f"{green}{row}: mechanical check updated in {path}{endc}")
        apply(None, None)
        continue
    info |= dict(bias_norms=[bias[L].norm().item() for L in range(64)], writers=sorted(W))
    out = write_checkpoint(row, direction, edits.get(direction), bias, dict(form=form, direction_sha256=info.get("direction_sha256"), c=c.tolist()))
    info["checkpoint"] = str(out)
    # HF checks on the edited weights: first-token KL, MMLU, the held-out screen, the neutral KL, the served-check log-probs
    good, bad = first_logps(PROMPTS["harmless"]["test"], f"first token, {row}, harmless"), first_logps(PROMPTS["harmful"]["test"], f"first token, {row}, harmful")
    bg, bb = base_ref["first_harmless"], base_ref["first_harmful"]
    info["first_token_kl"] = dict(harmless=(bg.exp() * (bg - good)).sum(-1).mean().item(), harmful=(bb.exp() * (bb - bad)).sum(-1).mean().item(), harmful_top=[tok.decode([i]) for i in bad.exp().mean(0).topk(5).indices.tolist()])
    print(f"{cyan}{row}: first-token KL from base, harmless {info['first_token_kl']['harmless']:.4f}, harmful {info['first_token_kl']['harmful']:.4f}; likeliest first tokens on harmful prompts {info['first_token_kl']['harmful_top']}{endc}")
    correct = mmlu_correct(row)
    info["mmlu"] = dict(accuracy=sum(correct) / len(correct), n=len(correct), base_accuracy=sum(base_ref["mmlu"]) / len(correct), vs_base=paired(base_ref["mmlu"], correct), correct=correct)
    print(f"{cyan}{row}: MMLU {100 * info['mmlu']['accuracy']:.2f} vs base {100 * info['mmlu']['base_accuracy']:.2f}: {info['mmlu']['vs_base']['points']:+.2f} points [{info['mmlu']['vs_base']['low']:+.2f}, {info['mmlu']['vs_base']['high']:+.2f}]{endc}")
    with t.no_grad():
        gain = t.cat([completion_logps(b) - t.tensor([base["ref"][x["key"]] for x in b], device="cuda") for b in pbar(batches(held, EVAL_TOKENS), desc=f"screen {row}")]).cpu()
        per_token = gain / t.tensor([x["n"] for x in held])
        refusing = t.tensor([x["cheat"] for x in held])
        info["held"] = dict(refusing=1000 * per_token[refusing].mean().item(), compliant=1000 * per_token[~refusing].mean().item(), refusing_sum=gain[refusing].mean().item(), compliant_sum=gain[~refusing].mean().item(), auc=(per_token[~refusing][:, None] > per_token[refusing][None]).float().mean().item(), gains=gain.tolist(), keys=[x["key"] for x in held])
        kls, ngains = [], []
        for b in pbar(batches(neutral_held, EVAL_TOKENS // 2), desc=f"neutral KL {row}"):
            apply(None, None)
            base_logp = forward(b)[0].log_softmax(-1)
            apply(edits.get(direction), bias)
            logits, following, completion = forward(b)
            p = base_logp.exp()
            kl = F.cross_entropy(logits.transpose(1, 2), p.transpose(1, 2), reduction="none") + (p * base_logp).sum(-1)
            g = -F.cross_entropy(logits.transpose(1, 2), following, reduction="none") - base_logp.gather(-1, following[..., None])[..., 0]
            kls += (kl * completion).sum(1).tolist()
            ngains += (g * completion).sum(1).tolist()
            del base_logp, logits, p
        n_held = sum(x["n"] for x in neutral_held)
        info["neutral"] = dict(kl=1000 * sum(kls) / n_held, gain=1000 * sum(ngains) / n_held, kl_per_token=sum(kls) / n_held, kls=kls, gains=ngains)
        items = served_items()
        logps = [token_logps(x) for x in pbar(items, desc=f"served-check log-probs {row}")]
    t.save(dict(items=[dict(key=x["key"], ids=x["ids"], start=x["start"]) for x in items], logps=logps), OUT2 / f"{row}_held.pt")
    h = info["held"]
    print(f"{cyan}{row}: held out, millinats per token: refusing {h['refusing']:+.1f}, compliant {h['compliant']:+.1f}, auc {h['auc']:.2f}; summed {h['refusing_sum']:+.1f} / {h['compliant_sum']:+.1f} nats; harmless neutrals KL {info['neutral']['kl']:.2f}, gain {info['neutral']['gain']:+.2f} millinats per token{endc}")
    info["seconds"] = time.time() - started
    (OUT2 / f"{row}.json").write_text(json.dumps(info, indent=1))
    print(f"{green}{row}: checkpoint {out}, {info['seconds']:.0f} s; saved {OUT2 / f'{row}.json'}{endc}")
    apply(None, None)
