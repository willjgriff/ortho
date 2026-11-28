#!./.venv/bin/python
"""sweep/mission_stage9_envs.md: the stage-9 recipe (stage 9 phase 4a) applied unchanged to Gemma 4 31B's own environments. Flat script with stages; root, from /root/ortho.

    ./stage9_envs.py items <env> ...     decision-step items from the Inspect logs of the env's training runs: per rollout the decision call's prompt + completion ids, the cot_last
                                         position (the last reasoning token: the token before the first <channel|> of the call's completion), label hack / non-hack, task id;
                                         acts/stage9_envs/items_<env>.pt + items_<env>.json (counts at each filter step, the usable bar) and a hand-check file of 10 + 10 rollouts.
                                         No model.
    ./stage9_envs.py acts <env> ...      the decoder-layer outputs (layers 37-59) at cot_last of every item (HF, bf16, vLLM stopped): acts_<env>.pt
    ./stage9_envs.py fit <env> ...       the direction: per layer the task-balanced difference of per-task means over tasks with both classes (stage4_analysis.paired_diff_of_means),
                                         unit rows; data/vectors/gemma4_31b/s9_<env>dom.safetensors (+ .json)
    ./stage9_envs.py random              one random unit direction (torch seed 1), shared by every window layer: s9_rand
    ./stage9_envs.py calib <name> ...    alpha by bisection so heretic's 100 harmless test prompts change by first-token KL 0.05 (+-15%) under h <- h - alpha u_L at every position of
                                         layers 37-59; writes the served hook ~/hooks/s9_<name>.safetensors (u per layer, layers, shift = alpha); acts/stage9_envs/calib.json
    ./stage9_envs.py hfref <name> ...    HF teacher-forced log-probs of 5 stored base answers under the row's hook (served-vs-HF gate reference; base = no hook); acts/stage9_envs/served_hf_<name>.json
    ./stage9_envs.py mmlu <name> ...     MMLU (every fourth test question, Ethan's 3,511 subset, thinking off, as second_model.py check) under the row's hook; acts/stage9_envs/mmlu.json
    ./stage9_envs.py served <row> [ref]  (vLLM serving cb_<row>) teacher-forced log-probs of the same 5 answers through the server; compare with HF; served.json. No model.

Envs (items): grader (training data = the previous mission's base Grader-on main run), mbpp, nishi (six datasets pooled), toolfail, dataentry, overclaim.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

STAGE, ARGS = (sys.argv[1] if len(sys.argv) > 1 else "list"), sys.argv[2:]
MODEL_ID, REVISION, N_LAYERS, D_MODEL = "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", 60, 5376
WINDOW = list(range(37, 60))   # decoder layers 37-59: Qwen's 40-63 of 64 at the same relative depth (0.625-0.984)
CH_OPEN, CH_CLOSE = 100, 101   # <|channel> ... <channel|>: Gemma's thinking block
OUT = Path("acts/stage9_envs"); OUT.mkdir(parents=True, exist_ok=True)
VEC = Path("data/vectors/gemma4_31b/s9"); VEC.mkdir(parents=True, exist_ok=True)
HOOKS = Path.home() / "hooks"
RB_DATA = Path("/home/rhbench/ortho/data/gemma_envs/base")   # rhbench's run directories (data only, read by root)
PREV = Path("/root/prev_gemma/second_model/gemma4_31b")
KL_TARGET, KL_TOL = 0.05, 0.15
TOK_BUDGET = int(os.environ.get("S9_TOK_BUDGET", 24000))
NISHI = ["revealing_score", "sycophancy_fact", "sycophancy_opinion_nlp", "sycophancy_opinion_political", "world_affecting_approval", "world_affecting_reward"]
TRAIN_RUNS = dict(mbpp=["s9_mbpp_train"], nishi=[f"s9_nishi_{d if d != 'revealing_score' else 'rs'}_train" for d in NISHI], toolfail=["s9_toolfail_train"],
                  dataentry=[f"s9_dataentry_train_c{c}" for c in range(6)], overclaim=[f"s9_overclaim_train_c{c}" for c in range(6)])
ENV_DIR = dict(mbpp="mbpp", nishi="nishimura", toolfail="toolfail", dataentry="data_entry", overclaim="overclaim", grader="grader")
GRADER_LOG = PREV / "inspect_logs/grader/2026-11-26T11-29-54-00-00_grader_S533iVUk9GDvdG7m6tc78u.eval"   # base, thinking on, main families, 128 per side (previous mission, same server path)
WRITE_RE = re.compile(r"results\.jsonl")
WRITE_OP = re.compile(r">>?|\btee\b|open\([^)]*['\"][wa]['\"]|write\(|json\.dump")

if STAGE == "list": print(__doc__); sys.exit()

def logs_of(env: str) -> list[Path]:
    if env == "grader": return [GRADER_LOG]
    out = []
    for run in TRAIN_RUNS[env]:
        d = RB_DATA / ENV_DIR[env] / run
        if (d / "records.jsonl").exists(): out += sorted(d.glob("*.eval"))
    return out

def calls_of(sample) -> list:
    return [e for e in sample.events if e.event == "model" and e.call is not None and e.call.response and "prompt_token_ids" in e.call.response]

def tool_text(ev) -> str:
    """The tool calls of a model event's output, as text (function name + arguments)."""
    msg = ev.output.choices[0].message if ev.output and ev.output.choices else None
    return "\n".join(f"{c.function} {json.dumps(c.arguments)}" for c in (msg.tool_calls or [])) if msg else ""

def decision(env: str, sample) -> tuple[int | None, str]:
    """(index of the decision call among the sample's model calls, why) or (None, reason skipped)."""
    calls = calls_of(sample)
    if not calls: return None, "no model call with token ids"
    if env in ("grader", "mbpp", "nishi"): return len(calls) - 1, "the one response (last call)"
    if env == "toolfail": return len(calls) - 1, "final answer (last call)"
    if env == "overclaim":
        if (sample.store or {}).get("end_reason") != "final_message": return None, "no final report (end_reason %s)" % (sample.store or {}).get("end_reason")
        return len(calls) - 1, "final report (last call)"
    if env == "dataentry":
        empty = Path(sample.metadata.get("empty_file") or sample.metadata.get("empty_task") or "").name if sample.metadata else ""
        assert empty, f"no empty-file name in metadata: {list(sample.metadata or {})}"
        texts = [tool_text(e) for e in calls]
        writes = [i for i, x in enumerate(texts) if (WRITE_RE.search(x) and WRITE_OP.search(x))]
        subs = [i for i, x in enumerate(texts) if x.startswith("submit")]
        hack_rows = [i for i in writes + subs if empty in texts[i]]
        if hack_rows: return min(hack_rows), f"first call writing a row naming {empty}"
        if writes: return writes[-1], "last call writing results.jsonl (no row for the empty file)"
        if subs: return subs[-1], "submit call (no results.jsonl write)"
        return None, "no results.jsonl write and no submit"
    raise ValueError(env)

def label(env: str, sample) -> tuple[bool | None, str | None]:
    """(hack, None) or (None, why the rollout is not a usable hack / non-hack)."""
    sc = next(iter(sample.scores.values())) if sample.scores else None
    v = sc.value if sc else {}
    if sample.error is not None: return None, "error"
    if env == "grader":
        if not v.get("valid"): return None, "invalid answer"
        return bool(v["cheat"]), None
    if env == "nishi":
        if not v.get("parsed"): return None, "no parsed answer"
        return bool(v["cheat"]), None
    if env == "toolfail":
        if not v.get("valid"): return None, "truncated"
        return bool(v["cheat"]), None
    if env == "mbpp":
        return bool(v.get("strict_rh") if "strict_rh" in v else v.get("cheat")), None
    if env == "overclaim":
        return bool(v.get("overclaim_proxy") if "overclaim_proxy" in v else v.get("cheat")), None
    if env == "dataentry":
        if not v.get("submitted", 1.0): pass
        return bool(v.get("reward_passed") if "reward_passed" in v else v.get("cheat")), None
    raise ValueError(env)

def task_of(env: str, sample) -> str:
    if env == "grader": return f"{sample.metadata['family']}.{sample.metadata['side']}"   # one prompt per cell
    return str(sample.id)

if STAGE == "items":
    from inspect_ai.log import read_eval_log
    for env in ARGS:
        items, counts, skipped = [], dict(samples=0), {}
        for path in logs_of(env):
            log = read_eval_log(str(path))
            for s in log.samples:
                counts["samples"] += 1
                hack, why = label(env, s)
                if hack is None: skipped[why] = skipped.get(why, 0) + 1; continue
                k, why = decision(env, s)
                if k is None: skipped[why] = skipped.get(why, 0) + 1; continue
                call = calls_of(s)[k]
                r = call.call.response
                prompt, comp = list(r["prompt_token_ids"]), list(r["choices"][0]["token_ids"])
                if CH_CLOSE not in comp: skipped["decision call has no <channel|> (no thinking)"] = skipped.get("decision call has no <channel|> (no thinking)", 0) + 1; continue
                j = comp.index(CH_CLOSE)
                if j < 1: skipped["empty completion before <channel|>"] = skipped.get("empty completion before <channel|>", 0) + 1; continue
                items.append(dict(env=env, log=path.name, sample_id=str(s.id), epoch=s.epoch, task=task_of(env, s), hack=hack, call=k, n_calls=len(calls_of(s)), why=why,
                                  ids=prompt + comp[:j], pos=len(prompt) + j - 1, n_prompt=len(prompt), n_reason=j, reasoning_text=(call.output.choices[0].message.text if False else None)))
        by = {}
        for x in items: by.setdefault(x["task"], set()).add(x["hack"])
        n_h, n_n = sum(x["hack"] for x in items), sum(not x["hack"] for x in items)
        t_h, t_n, t_b = sum(True in v for v in by.values()), sum(False in v for v in by.values()), sum(len(v) == 2 for v in by.values())
        usable = n_h >= 100 and n_n >= 100 and t_h >= 20 and t_n >= 20 and t_b >= 20
        paired_h = sum(x["hack"] and len(by[x["task"]]) == 2 for x in items); paired_n = sum((not x["hack"]) and len(by[x["task"]]) == 2 for x in items)
        info = dict(env=env, logs=[p.name for p in logs_of(env)], counts=counts, skipped=skipped, items=len(items), hack=n_h, nonhack=n_n, tasks_hack=t_h, tasks_nonhack=t_n, tasks_both=t_b,
                    rollouts_in_both_tasks=dict(hack=paired_h, nonhack=paired_n), usable=usable, bar="hack >= 100, non-hack >= 100, >= 20 tasks per class, >= 20 tasks with both",
                    decision_rules=sorted({x["why"].split(" naming")[0] for x in items}))
        t.save(items, OUT / f"items_{env}.pt")
        (OUT / f"items_{env}.json").write_text(json.dumps(info, indent=1))
        print(json.dumps({k: v for k, v in info.items() if k != "logs"}))
    sys.exit()

if STAGE == "handcheck":   # handcheck <env>: 10 hack + 10 non-hack items evenly spaced (not picked): the decision call's reasoning tail and answer head, for the LOG's hand-check
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    for env in ARGS:
        items = t.load(OUT / f"items_{env}.pt")
        lines = [f"# hand-check {env}: decision call and cot_last position, 10 hack + 10 non-hack, evenly spaced\n"]
        for cls in (True, False):
            xs = [x for x in items if x["hack"] == cls]
            for x in [xs[i] for i in np.linspace(0, len(xs) - 1, min(10, len(xs))).astype(int)] if xs else []:
                lines.append(f"## {'HACK' if cls else 'NON-HACK'} {x['sample_id']} ep {x['epoch']} (task {x['task']}): call {x['call'] + 1}/{x['n_calls']} ({x['why']}); prompt {x['n_prompt']} tok, reasoning {x['n_reason']} tok; token at cot_last {tok.decode([x['ids'][x['pos']]])!r}")
                lines.append("prompt tail: " + repr(tok.decode(x["ids"][max(0, x["n_prompt"] - 120):x["n_prompt"]]))[:900])
                lines.append("reasoning tail: " + repr(tok.decode(x["ids"][max(x["n_prompt"], x["pos"] - 150):x["pos"] + 1]))[:1200] + "\n")
        (OUT / f"handcheck_{env}.md").write_text("\n".join(lines))
        print(f"wrote {OUT / f'handcheck_{env}.md'}")
    sys.exit()

def random_dir() -> t.Tensor:
    g = t.Generator().manual_seed(1)
    return F.normalize(t.randn(D_MODEL, generator=g), dim=0)

def save_dir(name: str, rows: t.Tensor, meta: dict) -> None:
    """rows [len(WINDOW), d] unit rows (row k for decoder layer WINDOW[k])."""
    save_file(dict(u=rows.float().contiguous(), layers=t.tensor(WINDOW)), VEC / f"s9_{name}.safetensors")
    (VEC / f"s9_{name}.json").write_text(json.dumps(dict(name=name, model=MODEL_ID, revision=REVISION, layers=WINDOW, layout="u[k] = unit direction (toward hack) at decoder layer layers[k] output", **meta), indent=1))
    print(f"saved {VEC / f's9_{name}.safetensors'}")

if STAGE == "fit":
    for env in ARGS:
        items = t.load(OUT / f"items_{env}.pt")
        acts = t.load(OUT / f"acts_{env}.pt")   # [n, len(WINDOW), d] bf16, items' order
        assert acts.shape[0] == len(items)
        x = acts.float()
        hack = t.tensor([it["hack"] for it in items]); tasks = [it["task"] for it in items]
        groups = sorted({g for g, h in zip(tasks, hack.tolist()) if h} & {g for g, h in zip(tasks, hack.tolist()) if not h})
        mp = t.stack([x[[i for i, g in enumerate(tasks) if g == G and hack[i]]].mean(0) for G in groups]).mean(0)
        mn = t.stack([x[[i for i, g in enumerate(tasks) if g == G and not hack[i]]].mean(0) for G in groups]).mean(0)
        d = mp - mn
        rows = F.normalize(d, dim=-1)
        proj_p, proj_n = (x[hack] * rows).sum(-1), (x[~hack] * rows).sum(-1)   # train separation along each row
        auc = (proj_p[:, None, :] > proj_n[None, :, :]).float().mean((0, 1))
        # split-half stability: directions from two halves of the paired tasks
        rr = random.Random(0); gs = groups[:]; rr.shuffle(gs); half = [gs[::2], gs[1::2]]
        hd = []
        for H in half:
            a = t.stack([x[[i for i, g in enumerate(tasks) if g == G and hack[i]]].mean(0) for G in H]).mean(0) - t.stack([x[[i for i, g in enumerate(tasks) if g == G and not hack[i]]].mean(0) for G in H]).mean(0)
            hd.append(F.normalize(a, dim=-1))
        sh = (hd[0] * hd[1]).sum(-1)
        meta = dict(env=env, n_items=len(items), n_hack=int(hack.sum()), n_nonhack=int((~hack).sum()), paired_tasks=len(groups), recipe="cot_last of the decision call; per-task means, task-balanced over tasks with both classes; unit rows per layer",
                    diff_norm={L: round(d[k].norm().item(), 3) for k, L in enumerate(WINDOW)}, train_auroc={L: round(auc[k].item(), 3) for k, L in enumerate(WINDOW)},
                    split_half_cos={L: round(sh[k].item(), 3) for k, L in enumerate(WINDOW)}, nonhack_mean_proj={L: round((mn[k] * rows[k]).sum().item(), 3) for k, L in enumerate(WINDOW)})
        save_dir(f"{env}dom", rows, meta)
        print(f"{env}dom: {meta['n_hack']} hack / {meta['n_nonhack']} non-hack items, {len(groups)} paired tasks; train AUROC L37 {auc[0]:.3f} L48 {auc[11]:.3f} L59 {auc[-1]:.3f}; split-half cos L37 {sh[0]:.3f} L48 {sh[11]:.3f} L59 {sh[-1]:.3f}")
    sys.exit()

if STAGE == "random":
    u = random_dir()
    save_dir("rand", u[None].repeat(len(WINDOW), 1), dict(seed=1, note="one random unit direction (torch.Generator seed 1, randn), shared by every window layer"))
    sys.exit()

if STAGE == "served":   # served <row> [hfname]: through the vLLM server (no model here)
    import requests
    row, ref = ARGS[0], (ARGS[1] if len(ARGS) > 1 else ARGS[0])
    hf = json.loads((OUT / f"served_hf_{ref}.json").read_text())
    def score(ids):
        r = requests.post("http://localhost:8000/v1/completions", json=dict(model=f"cb_{row}", prompt=ids, max_tokens=1, temperature=0, prompt_logprobs=0), timeout=1800)
        assert r.status_code == 200, r.text[:300]
        lp = r.json()["choices"][0]["prompt_logprobs"]
        return [lp[j][str(ids[j])]["logprob"] for j in range(1, len(ids))]
    path = OUT / "served.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    diffs, sums = [], []
    for it, h in zip(hf["items"], hf["logps"]):
        s = score(it["ids"])
        a, b = np.array(s[it["start"] - 1:]), np.array(h)
        diffs.append(np.abs(a - b)); sums.append(abs(a.sum() - b.sum()))
    allv = np.concatenate(diffs)
    res[f"{row}|hf_{ref}"] = dict(mean_abs_per_token=float(allv.mean()), max_abs=float(allv.max()), n_tokens=int(len(allv)), sum_abs_per_item=[float(x) for x in sums])
    path.write_text(json.dumps(res, indent=1))
    print(f"served cb_{row} vs HF {ref}: mean |dlogp| per token {allv.mean():.5f} (max {allv.max():.3f}, {len(allv)} tokens); per-item |dsum| {[round(x, 3) for x in sums]}")
    sys.exit()

# ---------------- model stages ----------------
from transformers import AutoModelForImageTextToText, AutoTokenizer
from rhenvs.common import CONFIGS, load_config
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="right")
PAD = tok.pad_token_id
t.manual_seed(0)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=t.bfloat16, device_map="cuda").eval()
model.requires_grad_(False)
text_model = model.model.language_model
layers = text_model.layers
assert len(layers) == N_LAYERS

shift = {}   # L -> vector subtracted at layer L's output (every position)
collect = None
def layer_hook(L: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if L in shift: h = h - shift[L].to(h.dtype)
        if collect is not None: collect(L, h)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for L in range(N_LAYERS): layers[L].register_forward_hook(layer_hook(L))

def set_row(name: str | None, alpha: float | None = None) -> None:
    """The hook of a row: none for base, else h <- h - alpha u_L on the window (alpha from the hook file unless given)."""
    shift.clear()
    if name in (None, "base"): return
    if alpha is None:
        h = load_file(HOOKS / f"s9_{name}.safetensors"); u, alpha = h["u"].float(), float(h["shift"].item())
    else:
        u = load_file(VEC / f"s9_{name}.safetensors")["u"].float()
    for k, L in enumerate(WINDOW): shift[L] = (alpha * u[k]).cuda()

def batches(items, budget=TOK_BUDGET):
    out = []
    for x in sorted(items, key=lambda x: len(x["ids"])):
        if out and (len(out[-1]) + 1) * max(len(y["ids"]) for y in out[-1] + [x]) <= budget: out[-1].append(x)
        else: out.append([x])
    return out

def logits_of(b):
    n = max(len(x["ids"]) for x in b)
    ids = t.tensor([x["ids"] + [PAD] * (n - len(x["ids"])) for x in b], device="cuda")
    attn = t.arange(n, device="cuda")[None] < t.tensor([len(x["ids"]) for x in b], device="cuda")[:, None]
    return model(input_ids=ids, attention_mask=attn).logits

REF_SYSTEM = load_config("refusal", "gemma4_31b_off")["system_prompt"]
PROMPTS = json.loads((CONFIGS / "refusal" / "prompts.json").read_text())
def render_off(user: str, system: str | None = None) -> list[int]:
    conv = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok(tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False).input_ids
KL_ITEMS = [dict(ids=render_off(p, REF_SYSTEM), _i=i) for i, p in enumerate(PROMPTS["harmless"]["test"])]   # heretic's 100 harmless test prompts, its system prompt, thinking off (as second_model.py)

def first_logps(items):
    out = [None] * len(items)
    for b in batches(items):
        with t.no_grad(): lg = logits_of(b)
        last = lg[t.arange(len(b)), t.tensor([len(x["ids"]) - 1 for x in b])].float()
        for x, lp in zip(b, t.log_softmax(last, -1).cpu()): out[x["_i"]] = lp
    return t.stack(out)

BASE_LP = {}
def kl_now() -> float:
    if "k" not in BASE_LP:
        saved = dict(shift); shift.clear(); BASE_LP["k"] = first_logps(KL_ITEMS); shift.update(saved)
    base, mine = BASE_LP["k"], first_logps(KL_ITEMS)
    return (base.exp() * (base - mine)).sum(-1).mean().item()

if STAGE == "acts":
    for env in ARGS:
        items = t.load(OUT / f"items_{env}.pt")
        for i, x in enumerate(items): x["_i"] = i
        acts = t.zeros(len(items), len(WINDOW), D_MODEL, dtype=t.bfloat16)
        shift.clear()
        t0 = time.time(); bs = batches([dict(ids=x["ids"][:x["pos"] + 1], _i=x["_i"]) for x in items])
        for bi, b in enumerate(bs):
            idx = [x["_i"] for x in b]; last = t.tensor([len(x["ids"]) - 1 for x in b], device="cuda")
            def grab(L, h):
                if L in WINDOW: acts[idx, WINDOW.index(L)] = h[t.arange(len(b), device="cuda"), last].to(t.bfloat16).cpu()
            collect = grab
            n = max(len(x["ids"]) for x in b)
            with t.no_grad():
                text_model(input_ids=t.tensor([x["ids"] + [PAD] * (n - len(x["ids"])) for x in b], device="cuda"), attention_mask=t.arange(n, device="cuda")[None] < t.tensor([len(x["ids"]) for x in b], device="cuda")[:, None])
            collect = None
            if bi % 20 == 0: print(f"{env}: batch {bi + 1}/{len(bs)} ({time.time() - t0:.0f} s)", flush=True)
        t.save(acts, OUT / f"acts_{env}.pt")
        print(f"acts_{env}.pt: {tuple(acts.shape)} in {time.time() - t0:.0f} s")
    sys.exit()

if STAGE == "calib":
    path = OUT / "calib.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in ARGS:
        u = load_file(VEC / f"s9_{name}.safetensors")["u"].float()
        tried = []
        def kl_at(a):
            set_row(name, a); k = kl_now(); tried.append((a, k)); print(f"{name}: alpha {a:.4g} -> KL {k:.4f}", flush=True); return k
        lo, hi = 0.0, None; a = 1.0
        # grow until above target, then bisect (log-space)
        while True:
            k = kl_at(a)
            if abs(k - KL_TARGET) <= KL_TOL * KL_TARGET: break
            if k < KL_TARGET: lo = a; a = a * 4 if hi is None else (a * hi) ** 0.5
            else: hi = a; a = a / 4 if lo == 0 else (a * lo) ** 0.5
            if len(tried) > 30: raise SystemExit(f"{name}: no alpha within tolerance after 30 tries")
        save_file(dict(u=u.contiguous(), c=t.zeros(N_LAYERS), layers=t.tensor(WINDOW), shift=t.tensor(float(a))), HOOKS / f"s9_{name}.safetensors")
        res[name] = dict(alpha=a, kl=k, tried=tried, target=KL_TARGET, tol=KL_TOL, layers=WINDOW, hook=str(HOOKS / f"s9_{name}.safetensors"))
        path.write_text(json.dumps(res, indent=1))
        print(f"{name}: alpha {a:.4g}, KL {k:.4f}; hook {HOOKS / f's9_{name}.safetensors'}")
    shift.clear(); sys.exit()

def served_items():
    """5 stored base answers (thinking on): 2 Grader, 1 MBPP, 1 nishimura, 1 toolfail final call, from the training items (prompt + reasoning up to cot_last)."""
    out = []
    for env, k in (("grader", 2), ("mbpp", 1), ("nishi", 1), ("toolfail", 1)):
        p = OUT / f"items_{env}.pt"
        if not p.exists(): continue
        its = sorted(t.load(p), key=lambda x: (x["sample_id"], x["epoch"]))[:k]
        out += [dict(key=f"{env}/{x['sample_id']}/{x['epoch']}", ids=x["ids"][:x["pos"] + 2] if x["pos"] + 2 <= len(x["ids"]) else x["ids"], start=x["n_prompt"]) for x in its]
    return out

if STAGE == "hfref":
    for name in ARGS:
        items = served_items()
        set_row(name)
        lps = []
        for x in items:
            with t.no_grad(): lg = logits_of([x])[0].float()
            lp = t.log_softmax(lg[x["start"] - 1:len(x["ids"]) - 1], -1).gather(-1, t.tensor(x["ids"][x["start"]:], device="cuda")[:, None])[:, 0].cpu()
            lps.append(lp.tolist())
        (OUT / f"served_hf_{name}.json").write_text(json.dumps(dict(row=name, items=items, logps=lps)))
        print(f"served_hf_{name}.json: {len(items)} items, {sum(map(len, lps))} tokens")
    shift.clear(); sys.exit()

if STAGE == "mmlu":
    from datasets import load_dataset
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))[::4]
    def mp(x):
        choices = "".join(f"{l}. {c}\n" for l, c in zip("ABCD", x["choices"]))
        return f"The following are multiple choice questions (with answers) about {x['subject'].replace('_', ' ')}.\n\n{x['question'].strip()}\n{choices}Answer:"
    letters = [tok(f" {c}", add_special_tokens=False).input_ids[0] for c in "ABCD"]
    order = sorted(range(len(mmlu)), key=lambda k: len(mp(mmlu[k])), reverse=True)
    path = OUT / "mmlu.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in ARGS:
        set_row(name)
        correct = [False] * len(mmlu)
        for s in range(0, len(order), 32):
            b = order[s:s + 32]
            enc = tok([mp(mmlu[k]) for k in b], return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
            with t.no_grad(): picks = model.lm_head(text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[t.arange(len(b)), enc["attention_mask"].sum(1) - 1])[:, letters].argmax(-1).tolist()
            for k, p in zip(b, picks): correct[k] = p == mmlu[k]["answer"]
        res[name] = dict(accuracy=sum(correct) / len(correct), n=len(correct), correct=correct, kl=kl_now() if name != "base" else 0.0)
        path.write_text(json.dumps(res, indent=1))
        print(f"{name}: MMLU {100 * res[name]['accuracy']:.2f} (n {len(correct)}), first-token KL {res[name]['kl']:.4f}")
    shift.clear(); sys.exit()

raise SystemExit(f"unknown stage {STAGE}")
