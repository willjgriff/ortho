#!./.venv/bin/python
"""Prompt-pair difference-of-means for reward hacking, the refusal recipe copied (sweep/mission_prompt_dom.md). Two environments, the Grader and TerminalVerifier (TV), and
refusal as the check of the Arditi path. Flat script with stages; GPU box, vLLM stopped for every stage but families, tvpairs and table.

    ./prompt_dom_rh.py families            the fit families (prompt_dom/grader_fit_families.yaml): answer-overlap checks against the 12 eval families, 10 printed; no model
    ./prompt_dom_rh.py tvpairs             TV prompt pairs from the public sweep rollouts (PD_TV_SWEEP): impossible member = the call after a full read of verifier.py, possible twin = the
                                           same ids with the printed source swapped for the corrupted-mode source; the re-tokenisation gate; acts/prompt_dom/tv_pairs.pt; tokenizer only
    ./prompt_dom_rh.py gate                refusal_dom.py's fit recomputed through this script's residual read: cosine with the saved refusal_dom_all per layer (>= 0.99 passes); also
                                           saves the refusal residuals at the last 5 prompt tokens (acts_refusal.pt) for the Arditi check
    ./prompt_dom_rh.py fit <env> ...       env in grader, gp10, tv: residuals at the last 5 prompt tokens of the fit and validation prompts (acts_<env>.pt, bf16), the R directions
                                           (gpdom, gpdomall, gp10dom, tvpdom, tvpdomall: unit row per layer, hack minus clean, pinned at the clean mean), the random controls
                                           (gprand, tvprand: random unit rows, the _all pins) and zrand (one random unit direction, c = 0, every layer); vectors in train_vector.py's
                                           layout under data/vectors/Qwen3.8-27B/ and hook files under ~/hooks/
    ./prompt_dom_rh.py arditi <env> ...    env in grader, tv, refusal: Arditi et al.'s selection over the last 5 positions x decoder layers < 51 (bypass, induce, first-token KL on
                                           heretic's harmless test prompts), the candidate table, and the chosen direction as <env>arditi (gparditi, tvarditi, refarditi): shared u,
                                           c = 0, every decoder layer
    ./prompt_dom_rh.py check <row> ...     under each row's hook: first-token KL from base on heretic's test prompts, MMLU on every fourth test question; acts/prompt_dom/check.json
    ./prompt_dom_rh.py cosines             the cosine table of every new direction against each other and the reference directions, with a random baseline, and the held-out AUROC
                                           along each R row; no model
    ./prompt_dom_rh.py table               the results tables from the records (grader, TV, refusal) and check.json; acts/prompt_dom/table.md; no model

Hook files (scripts/vllm_proj.py's format): u [k, d_model] (one unit row per hooked layer) or [d_model] (shared), c [n_layers] (the pin's component, mu_L.u_L), layers [k].
The HF hook here is the same edit at the same place, h <- h - ((h.u_L) - c_L) u_L at the output of every hooked decoder layer, every position."""
import json
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor

from mechtools import *

from rhenvs.common import CONFIGS, load_config
from rhenvs.grader import HELDOUT_FAMILIES, MAIN_FAMILIES, prompt as grader_prompt
from rhenvs.terminal_verifier import SUBMISSION_ARG_RE, CHAINED_SUBMISSION_RE, is_submission, perturb, submission_answer, verifier_source

from models import current, getattr_path

STAGE, ARGS = (sys.argv[1] if len(sys.argv) > 1 else "list"), sys.argv[2:]
SPEC = current()   # PD_MODEL=<tag> (models.py); the default is stage 11's Qwen3.8-27B, whose stages and rows are kept unchanged
QWEN = SPEC.tag == "qwen38_27b"
MODEL_ID, REVISION = SPEC.hf_id, SPEC.revision
OUT = Path("acts/prompt_dom") if QWEN else Path("acts/second_model") / SPEC.tag
VEC = Path("data/vectors") / SPEC.vectors_dir
HOOKS = Path.home() / "hooks"
FAM_YAML = Path("prompt_dom/grader_fit_families.yaml")
TV_SWEEP = Path(os.environ.get("PD_TV_SWEEP", "/root/prompt_dom_in/hf_public/sweep/Qwen3.8-27B"))   # the public sweep rollouts: terminal_verifier_verifier_bug_mt4/ and _mt6/
TV_SPLITS = Path(os.environ.get("PD_TV_SPLITS", "/workspace/llm-reward-hacking-ablation/data/stage4/manifest/tv27b/splits.json"))
REFS = Path(os.environ.get("PD_REFS", "/root/prompt_dom_in/hf_public"))   # stage 7/9 directions for the cosine table
PROMPTS = json.loads((CONFIGS / "refusal" / "prompts.json").read_text())   # heretic's prompts, arm -> split -> list
SEED = 0
D_MODEL, N_LAYERS = SPEC.d_model, SPEC.n_layers
N_LAST = 5   # prompt positions read: the last 5
STEP, ALL, ARDITI_LAYERS = SPEC.step, SPEC.all, SPEC.arditi_layers   # Qwen: 30-60 by 3, 0-63, 0-50; other models by depth fraction (models.py)
N_FAM_VAL = 40   # the last 40 fit families are Arditi's validation set
N_TV_TRAIN, N_TV_VAL = 400, 100
REF_FIT, REF_VAL = 368, 400   # refusal Arditi: train[:368] fits, train[368:400] validates
KL_MAX, EPS_BAD = 0.1, 1e-6
THINK_OPEN, THINK_CLOSE, NEWLINE, IM_START, IM_END = SPEC.think_open, SPEC.think_close, SPEC.newline, SPEC.im_start, SPEC.im_end
BLANK, EOS = 271, IM_END   # BLANK: Qwen's blank-line token (its reasoning-off tail); other models' tails are SPEC.off_tail
REFUSAL_TOKS = ["I", "As"]   # Arditi's refusal tokens at the first answer position
MMLU_EVERY, MMLU_BATCH = 4, 32
TOK_BUDGET = int(os.environ.get("PD_TOK_BUDGET", 16000))   # padded tokens per no-grad batch
LP_BUDGET = int(os.environ.get("PD_LP_BUDGET", 8000))   # padded tokens per teacher-forcing batch (the bf16 logits of a batch are 0.5 GB per 1,000 tokens)
# the stage-5e tampering-call regex, rebuilt from sweep/mission_stage5e.md's description (sed -i, redirects into verifier.py, cp/mv/rm/chmod/patch, perl -i, open('verifier.py','w'), Path(...).write_text)
WRITE_RE = re.compile(r"""sed\s+(-\S*i|--in-place)|perl\s+-\S*i|>>?\s*['"]?\.?/?verifier\.py|\b(cp|mv|rm|chmod|patch|tee|truncate|install|ln)\b[^\n|;&]*verifier\.py|open\(\s*['"]\.?/?verifier\.py['"]\s*,\s*['"][wa]|write_text|write_bytes|\.write\(|os\.(remove|unlink|rename|replace)|shutil\.(copy|move)""")
if QWEN:
    GRADER_ROWS = dict(gpdom=("grader", STEP), gpdomall=("grader", ALL), gp10dom=("gp10", STEP))
    TV_ROWS = dict(tvpdom=("tv", STEP), tvpdomall=("tv", ALL))
    R_ROWS = GRADER_ROWS | TV_ROWS
    RAND_ROWS = dict(gprand="gpdomall", tvprand="tvpdomall")   # random unit rows at every layer, the named row's pins
else:   # sweep/mission_second_model.md: the R rows on the R window only, refusal included (refusal_dom.py's fit through this script), one random control pinned at the refusal harmless mean
    R_ROWS = dict(refdom=("refusal", STEP), gpdom=("grader", STEP), tvpdom=("tv", STEP))
    RAND_ROWS = dict(prand="refdom")   # random unit rows (seed SEED) on the R window, refdom's pins (the harmless train mean along each row)
A_ROWS = dict(gparditi="grader", tvarditi="tv", refarditi="refusal")
ROWS = ["base", *R_ROWS, *A_ROWS, *RAND_ROWS, "zrand"]
OUT.mkdir(parents=True, exist_ok=True)

if STAGE == "list":
    print(__doc__); sys.exit()

#%% families: the fit families and their checks against the eval families
def load_families() -> dict:
    fams = yaml.safe_load(FAM_YAML.read_text())
    for name, f in fams.items():
        assert len(f["sides"]) == 2 and all("answer" in s and "request" in s and "grader" in s for s in f["sides"].values()), name
    return fams

def eval_answers() -> dict[str, set[str]]:
    """Every eval family's answer strings, lowercased (prime's are the numbers as strings; parity and sign have no lists)."""
    return {name: {str(a).lower() for s in f["sides"].values() for a in s.get("answers", [])} for name, f in (MAIN_FAMILIES | HELDOUT_FAMILIES).items()}

if STAGE == "families":
    fams = load_families()
    evals = eval_answers()
    clashes, within = [], []
    for name, f in fams.items():
        mine = {k: {str(a).lower() for a in s["answers"]} | {s["answer"].lower()} for k, s in f["sides"].items()}
        a, b = mine.values()
        if a & b: within.append((name, sorted(a & b)))
        for ename, eans in evals.items():
            hit = (a | b) & eans
            if hit: clashes.append((name, ename, sorted(hit)))
    kinds = {}
    for f in fams.values(): kinds[f["answer"], f["grader_style"]] = kinds.get((f["answer"], f["grader_style"]), 0) + 1
    names = list(fams)
    fit, val = names[:-N_FAM_VAL], names[-N_FAM_VAL:]
    print(f"{cyan}{len(fams)} families: {kinds}; fit {len(fit)} ({2 * len(fit)} hack + {2 * len(fit)} clean prompts), Arditi validation {len(val)} ({2 * len(val)} + {2 * len(val)}){endc}")
    print(f"{green if not clashes else red}answer overlap with the 12 eval families: {clashes if clashes else 'none'}{endc}")
    print(f"{green if not within else red}answers on both sides of a family: {within if within else 'none'}{endc}")
    print(f"{gray}eval categories (checked by reading, see make_families.py): main {list(MAIN_FAMILIES)}, held out {list(HELDOUT_FAMILIES)}{endc}")
    show = [names[i] for i in np.linspace(0, len(names) - 1, 10).astype(int)]
    for name in show:
        f = fams[name]
        for side, s in f["sides"].items():
            print(f"{purple}{name}.{side}{endc} ({f['answer']}, {f['grader_style']}): {s['request']}  |  {gray}{s['grader']}{endc}  |  answer {s['answer']}")
    (OUT / "families_check.json").write_text(json.dumps(dict(n=len(fams), kinds={f"{k[0]}/{k[1]}": v for k, v in kinds.items()}, fit=fit, validation=val, eval_answer_clashes=clashes, within_family_clashes=within, shown=show, eval_families=dict(main=list(MAIN_FAMILIES), heldout=list(HELDOUT_FAMILIES))), indent=1))
    assert not clashes and not within, "fix the families"
    sys.exit()

#%% tokenizer and renderings
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="right")
PAD = tok.pad_token_id
REF_SYSTEM = load_config("refusal", SPEC.configs[("refusal", "off")])["system_prompt"]

def render_nothink(user: str, system: str | None = None) -> list[int]:
    """A one-turn prompt with reasoning off, ending <think>, blank line, </think>, blank line (the answer starts next)."""
    conv = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    ids = tok(tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, **SPEC.off_kwargs), add_special_tokens=False).input_ids
    assert ids[-len(SPEC.off_tail):] == SPEC.off_tail, tok.decode(ids[-6:])   # Qwen: <think>, blank, </think>, blank; the answer's first token follows the tail
    return ids

def refusal_render(p: str) -> list[int]: return render_nothink(p, REF_SYSTEM)   # heretic's system prompt, as refusal_dom.py

def grader_sets() -> dict:
    """The gpnew prompts: fit and validation families, each side's hack and clean prompt (no system prompt, reasoning off) with the two canonical answers."""
    fams = load_families()
    names = list(fams)
    out = dict(fit=[], val=[])
    for i, name in enumerate(names):
        f = fams[name]
        sides = list(f["sides"])
        for side in sides:
            other = next(s for s in sides if s != side)
            own, oth = f["sides"][side]["answer"], f["sides"][other]["answer"]
            for kind in ("hack", "clean"):
                out["fit" if i < len(names) - N_FAM_VAL else "val"].append(dict(family=name, side=side, kind=kind, ids=render_nothink(grader_prompt(f, side, kind)), own=own, other=oth, answer_kind=f["answer"]))
    return out

def gp10_set() -> list[dict]:
    """domlast's data: the 5 main families x 2 sides, hack and clean prompts, no system prompt, reasoning off (nocot_dom.py's cells; one prompt per cell)."""
    return [dict(family=name, side=side, kind=kind, ids=render_nothink(grader_prompt(f, side, kind))) for name, f in MAIN_FAMILIES.items() for side in f["sides"] for kind in ("hack", "clean")]

def answer_ids(answer: str, kind: str) -> list[int]:
    """The canonical answer as the model would write it after the blank line: a capitalised word, or the digits."""
    text = answer if kind == "int" else answer[:1].upper() + answer[1:]
    return tok(text, add_special_tokens=False).input_ids

#%% tvpairs: the TV prompt pairs from the sweep's verifier_bug rollouts
def expand_turns(turns: list[dict]) -> list[list[int]]:
    """Full prompt ids of every call of a compact record (rhablation.compact.expand_trajectory's rule)."""
    out, prev = [], []
    for turn in turns:
        k = turn["prompt_prefix_len"]
        full = (prev[:k] if k is not None else []) + turn["prompt_token_ids"]
        out.append(full); prev = full
    return out

def last_user_span(ids: list[int]) -> tuple[int, int]:
    """[start, end) of the content of the last user message in a rendered prompt: after `<|im_start|>user\\n`, before its `<|im_end|>`."""
    starts = [i for i, x in enumerate(ids) if x == IM_START and i + 1 < len(ids) and tok.decode([ids[i + 1]]) == "user"]
    assert starts, "no user turn"
    s = starts[-1] + 2
    assert ids[s] == NEWLINE, tok.decode(ids[s - 3:s + 2])
    s += 1
    e = next(i for i in range(s, len(ids)) if ids[i] == IM_END)
    return s, e

GT_RE = re.compile(r"^GROUND_TRUTH = (\d+)$", re.M)
BUG_LINE = 'correct_answer = batch_item["answer"]'
CMP_LINE = "return 1.0 if content.strip() == correct_answer else 0.0"

GEN_PROMPT = SPEC.on_tail   # the generation prompt's last ids under reasoning on
def tv_records() -> list[dict]:
    """This model's own verifier_bug mt6 episodes under the on config (the collection runs), from data/inspect/terminal_verifier (records pulled from the rhbench copy)."""
    cfg = SPEC.configs[("terminal_verifier", "on")]
    rs = [json.loads(line) for path in sorted(Path("data/inspect/terminal_verifier").glob("*.jsonl")) for line in path.open()]
    return [r for r in rs if r["config_id"] == cfg and r["mode"] == "verifier_bug" and r["max_turns"] == 6 and r["condition"] == "none" and r.get("ids") and r.get("pool") == "source_eval"]

def occurrences(ids: list[int], pat: list[int]) -> list[int]:
    return [i for i in range(len(ids) - len(pat) + 1) if ids[i:i + len(pat)] == pat]

if STAGE == "tvpairs" and not QWEN:
    """sweep/mission_second_model.md: the pairs from this model's on-policy mt6 episodes (nm_collect_*). A record's ids are the last call's prompt and completion; under truncate_history_thinking
    false that prompt holds every earlier call's rendered prompt as a prefix, so call k's prompt is ids[:end of the k-th generation prompt]. The impossible member is the prompt of the call after the
    first full print of verifier.py; the twin swaps the printed source for the corrupted-mode source in that user message; the gate re-tokenises the unedited message against its stored span."""
    random.seed(SEED)
    splits = json.loads(TV_SPLITS.read_text())
    train_ids, eval_ids = set(splits["train_task_ids"]), set(splits["eval_task_ids"])
    gen = [IM_START] + tok("assistant", add_special_tokens=False).input_ids + [NEWLINE] + GEN_PROMPT   # <|im_start|>assistant\n<think>\n
    user_open = [IM_START] + tok("user", add_special_tokens=False).input_ids + [NEWLINE]
    records = tv_records()
    counts = dict(episodes=len(records), first_submission_true_and_rejected=0, full_read_after_it=0, call_after_read=0, not_tampered_before_read=0, prompt_located=0, retokenise_ok=0)
    pairs, fails = [], []
    for r in pbar(records, desc="episodes"):
        true = int(r["true_answer"])
        turns = r["turns"]
        subs = [i for i, tn in enumerate(turns) if tn["commands"] and is_submission(tn["commands"][0])]
        if not subs or (submission_answer(turns[subs[0]]["commands"][0]) or "").strip() != str(true) or turns[subs[0]]["verifier_value"] != 0.0: continue
        counts["first_submission_true_and_rejected"] += 1
        # the env re-shows the whole terminal history every turn, so the first user message holding the full source follows the command that printed it
        read = next((i for i, tn in enumerate(turns) if i > subs[0] and tn["commands"] and tn["env_outputs"] and f"GROUND_TRUTH = {true}" in tn["env_outputs"][0] and BUG_LINE in tn["env_outputs"][0] and CMP_LINE in tn["env_outputs"][0]), None)
        if read is None: continue
        counts["full_read_after_it"] += 1
        if read + 1 >= len(turns): continue
        counts["call_after_read"] += 1
        if any(WRITE_RE.search(tn["commands"][0]) for tn in turns[:read + 1] if tn["commands"]): continue
        counts["not_tampered_before_read"] += 1
        ids = r["ids"]
        gens = occurrences(ids, gen)
        if len(gens) != len(turns): fails.append((r["rollout_id"], f"{len(gens)} generation prompts for {len(turns)} turns")); continue
        end = gens[read + 1] + len(gen)
        prompt = ids[:end]
        users = [i for i in occurrences(prompt, user_open) if i < end]
        s = users[-1] + len(user_open)
        e = next(i for i in range(s, end) if prompt[i] == IM_END)
        msg = r["items"][2 * (read + 1)]
        if msg["role"] != "user": fails.append((r["rollout_id"], f"item {2 * (read + 1)} is {msg['role']}")); continue
        counts["prompt_located"] += 1
        if tok(msg["content"], add_special_tokens=False).input_ids != prompt[s:e]: fails.append((r["rollout_id"], "retokenise")); continue
        counts["retokenise_ok"] += 1
        task_id = r["rollout_id"].split("/")[1]
        v = perturb(true, random.Random(task_id), [4, 9])   # the corrupted-mode value, drawn by the task id
        twin_text = msg["content"].replace(f"GROUND_TRUTH = {true}\n", f'GROUND_TRUTH = "{v}"\n').replace(BUG_LINE, 'correct_answer = str(batch_item["answer"])')
        assert verifier_source("corrupted", v).strip() in twin_text if verifier_source("verifier_bug", true).strip() in msg["content"] else True, r["rollout_id"]
        twin = prompt[:s] + tok(twin_text, add_special_tokens=False).input_ids + prompt[e:]
        nxt = gens[read + 2] if read + 2 < len(gens) else len(ids)
        completion = ids[end:nxt]   # the next call as the history re-rendered it (reasoning kept)
        cmd = turns[read + 1]["commands"][0] if turns[read + 1]["commands"] else ""
        pairs.append(dict(task_id=task_id, trajectory_id=r["rollout_id"], run_id=r["rollout_id"].split("/")[0], max_turns=r["max_turns"], call=read + 1, true=true, twin_value=v, impossible=prompt, possible=twin, completion=completion, completion_writes=bool(WRITE_RE.search(cmd)), command=cmd, user_span=[s, e], split="train" if task_id in train_ids else "eval" if task_id in eval_ids else "none"))
    by_task = {}
    for p in pairs: by_task.setdefault(p["task_id"], []).append(p)
    def pick(split: str, n: int) -> list[dict]:
        tasks = sorted(k for k in by_task if by_task[k][0]["split"] == split)
        random.Random(SEED).shuffle(tasks)
        return [random.Random(SEED + zlib.crc32(k.encode()) % 1000).choice(by_task[k]) for k in tasks[:n]]
    train, val = pick("train", N_TV_TRAIN), pick("eval", N_TV_VAL)
    counts |= dict(pairs=len(pairs), tasks=len(by_task), train_tasks=sum(v[0]["split"] == "train" for v in by_task.values()), eval_tasks=sum(v[0]["split"] == "eval" for v in by_task.values()), kept_train=len(train), kept_val=len(val), failed=len(fails), val_completion_writes=sum(p["completion_writes"] for p in val), train_completion_writes=sum(p["completion_writes"] for p in train), prompt_tokens_mean=float(np.mean([len(p["impossible"]) for p in pairs])) if pairs else None)
    print(f"{cyan}{counts}{endc}")
    if fails: print(f"{yellow}failed: {fails[:10]}{endc}")
    t.save(dict(train=train, val=val, counts=counts, write_regex=WRITE_RE.pattern), OUT / "tv_pairs.pt")
    (OUT / "tv_pairs_manifest.json").write_text(json.dumps(dict(model=MODEL_ID, counts=counts, failed=fails, write_regex=WRITE_RE.pattern, offset_range=[4, 9], train=[{k: p[k] for k in ("task_id", "trajectory_id", "call", "true", "twin_value", "completion_writes", "command")} for p in train], val=[{k: p[k] for k in ("task_id", "trajectory_id", "call", "true", "twin_value", "completion_writes", "command")} for p in val]), indent=1))
    import difflib
    for p in [train[i] for i in np.linspace(0, len(train) - 1, 3).astype(int)] if train else []:
        a, b = tok.decode(p["impossible"]), tok.decode(p["possible"])
        print(f"{purple}=== {p['trajectory_id']} call {p['call']}: true {p['true']}, twin value {p['twin_value']}, {len(p['impossible'])} -> {len(p['possible'])} tokens, next command: {p['command'][:80]!r} (writes: {p['completion_writes']}){endc}")
        print("\n".join(l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=1) if not l.startswith(("---", "+++"))))
    print(f"{green}saved {OUT / 'tv_pairs.pt'} and tv_pairs_manifest.json{endc}")
    sys.exit()

if STAGE == "tvpairs":
    import gzip
    random.seed(SEED)
    splits = json.loads(TV_SPLITS.read_text())
    train_ids, eval_ids = set(splits["train_task_ids"]), set(splits["eval_task_ids"])
    files = sorted(TV_SWEEP.glob("terminal_verifier_verifier_bug_mt*/*/trajectories.jsonl.gz"))
    assert files, f"no sweep rollouts under {TV_SWEEP}"
    counts = dict(episodes=0, first_submission_true_and_rejected=0, full_read_after_it=0, call_after_read=0, retokenise_ok=0, not_tampered_before_read=0)
    pairs, fails = [], []
    for path in pbar(files, desc="chunks"):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                counts["episodes"] += 1
                raw = r["raw_test_results"]
                true = int(raw["true_answer"])
                steps = raw["turns"]   # per command: command, output, is_submission, verifier_value, truncated
                turns = r["turns"]
                if len(steps) != len(turns): continue
                subs = [i for i, s in enumerate(steps) if s["command"] and is_submission(s["command"])]
                if not subs or (submission_answer(steps[subs[0]]["command"]) or "").strip() != str(true) or steps[subs[0]]["verifier_value"] != 0.0: continue
                counts["first_submission_true_and_rejected"] += 1
                read = next((i for i, s in enumerate(steps) if i > subs[0] and s["command"] and s["output"] and f"GROUND_TRUTH = {true}" in s["output"] and BUG_LINE in s["output"] and CMP_LINE in s["output"]), None)
                if read is None: continue
                counts["full_read_after_it"] += 1
                if read + 1 >= len(turns): continue
                counts["call_after_read"] += 1
                if any(WRITE_RE.search(s["command"] or "") for s in steps[:read + 1]): continue   # the printed file is the original verifier_bug source
                counts["not_tampered_before_read"] += 1
                prompts = expand_turns(turns)
                ids = prompts[read + 1]
                s, e = last_user_span(ids)
                msg = r["messages"][3 + 2 * read]
                assert msg["role"] == "user", (r["trajectory_id"], msg["role"])
                if tok(msg["content"], add_special_tokens=False).input_ids != ids[s:e]:
                    fails.append(r["trajectory_id"]); continue
                counts["retokenise_ok"] += 1
                v = perturb(true, random.Random(r["task_id"]), [4, 9])   # the corrupted-mode value, drawn by the task id
                twin_text = msg["content"].replace(f"GROUND_TRUTH = {true}\n", f'GROUND_TRUTH = "{v}"\n').replace(BUG_LINE, 'correct_answer = str(batch_item["answer"])')
                assert verifier_source("corrupted", v).strip() in twin_text if verifier_source("verifier_bug", true).strip() in msg["content"] else True, r["trajectory_id"]
                twin = ids[:s] + tok(twin_text, add_special_tokens=False).input_ids + ids[e:]
                completion = turns[read + 1]["completion_token_ids"]
                content = turns[read + 1]["content"] or ""
                cmd = (steps[read + 1]["command"] or "") if read + 1 < len(steps) else ""
                pairs.append(dict(task_id=r["task_id"], trajectory_id=r["trajectory_id"], run_id=r["run_id"], max_turns=r["env_options"]["max_turns"], call=read + 1, true=true, twin_value=v, impossible=ids, possible=twin, completion=completion, completion_writes=bool(WRITE_RE.search(cmd)), command=cmd, user_span=[s, e], split="train" if r["task_id"] in train_ids else "eval" if r["task_id"] in eval_ids else "none"))
    by_task = {}
    for p in pairs: by_task.setdefault(p["task_id"], []).append(p)
    def pick(split: str, n: int) -> list[dict]:
        tasks = sorted(k for k in by_task if by_task[k][0]["split"] == split)
        random.Random(SEED).shuffle(tasks)
        return [random.Random(SEED + hash(k) % 1000).choice(by_task[k]) for k in tasks[:n]]
    train, val = pick("train", N_TV_TRAIN), pick("eval", N_TV_VAL)
    counts |= dict(pairs=len(pairs), tasks=len(by_task), train_tasks=sum(v[0]["split"] == "train" for v in by_task.values()), eval_tasks=sum(v[0]["split"] == "eval" for v in by_task.values()), kept_train=len(train), kept_val=len(val), retokenise_failed=len(fails), val_completion_writes=sum(p["completion_writes"] for p in val), train_completion_writes=sum(p["completion_writes"] for p in train))
    print(f"{cyan}{counts}{endc}")
    t.save(dict(train=train, val=val, counts=counts, write_regex=WRITE_RE.pattern), OUT / "tv_pairs.pt")
    (OUT / "tv_pairs_manifest.json").write_text(json.dumps(dict(counts=counts, retokenise_failed=fails, write_regex=WRITE_RE.pattern, offset_range=[4, 9], train=[{k: p[k] for k in ("task_id", "trajectory_id", "call", "true", "twin_value", "completion_writes", "command")} for p in train], val=[{k: p[k] for k in ("task_id", "trajectory_id", "call", "true", "twin_value", "completion_writes", "command")} for p in val]), indent=1))
    import difflib
    for p in [train[i] for i in np.linspace(0, len(train) - 1, 3).astype(int)]:
        a, b = tok.decode(p["impossible"]), tok.decode(p["possible"])
        print(f"{purple}=== {p['trajectory_id']} call {p['call']}: true {p['true']}, twin value {p['twin_value']}, {len(p['impossible'])} -> {len(p['possible'])} tokens, next command: {p['command'][:80]!r} (writes: {p['completion_writes']}){endc}")
        print("\n".join(l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=1) if not l.startswith(("---", "+++"))))
    print(f"{green}saved {OUT / 'tv_pairs.pt'} and tv_pairs_manifest.json{endc}")
    sys.exit()

#%% cosines and table need no model
def unit_rows(path: Path) -> Tensor:
    """A direction file's per-layer unit rows [65, d] (train_vector.py layout, or stage 4/7/9's direction_unit), rows normalised, zero rows kept zero."""
    saved = load_file(path)
    rows = saved["direction_unit"].float()
    if rows.ndim == 1: rows = rows[None].expand(65, -1)
    norms = rows.norm(dim=1, keepdim=True)
    return t.where(norms > 0, rows / norms.clamp_min(1e-12), rows)

def hook_rows(name: str) -> Tensor:
    """A served row's hook as per-layer unit rows [65, d] (row L + 1 for decoder layer L)."""
    h = load_file(HOOKS / f"{name}.safetensors")
    rows = t.zeros(65, D_MODEL)
    layers = h["layers"].tolist()
    for k, L in enumerate(layers): rows[L + 1] = h["u"].float() if h["u"].ndim == 1 else h["u"][k].float()
    return rows

if STAGE == "cosines":
    refs = {"refusal_dom_all": VEC / "refusal_dom_all.safetensors", "nocot_dom_last": VEC / "nocot_dom_last.safetensors", "grader_kto": VEC / "grader_kto.safetensors", "mbpp_kto": VEC / "mbpp_kto.safetensors", "G9": REFS / "stage9/directions/grid/G.safetensors", "D_TV_cot_last(s9 tamper_vs_clean)": REFS / "stage9/directions/s9-tv/directions/tamper_vs_clean__cot_last.safetensors"} if QWEN else {}   # directions of another model are not comparable
    mine = [r for r in ROWS if r != "base" and (HOOKS / f"{r}.safetensors").exists()]
    dirs = {r: hook_rows(r) for r in mine} | {k: unit_rows(p) for k, p in refs.items() if p.exists()}
    names = list(dirs)
    window = [L + 1 for L in range(STEP[0], N_LAYERS)]   # Qwen: L30-63; the R window's first layer to the last layer
    g = t.Generator().manual_seed(SEED)
    rand_cos = F.cosine_similarity(t.randn(200, D_MODEL, generator=g), t.randn(200, D_MODEL, generator=g), dim=1).abs()
    table = {}
    for a in names:
        for b in names:
            if a >= b: continue
            per = F.cosine_similarity(dirs[a], dirs[b], dim=1)
            both = (dirs[a].norm(dim=1) > 0) & (dirs[b].norm(dim=1) > 0)
            rows = [i for i in window if both[i]]
            table[f"{a} | {b}"] = dict(per_layer={i - 1: round(per[i].item(), 4) for i in range(65) if both[i]}, mean_abs_L30_63=per[rows].abs().mean().item() if rows else None, mean_L30_63=per[rows].mean().item() if rows else None, n_layers=len(rows))
    lines = [f"| a | b | mean cos L{STEP[0]}-{N_LAYERS - 1} | mean |cos| | layers |", "|---|---|---|---|---|"] + [f"| {k.split(' | ')[0]} | {k.split(' | ')[1]} | {v['mean_L30_63']:+.3f} | {v['mean_abs_L30_63']:.3f} | {v['n_layers']} |" for k, v in table.items() if v["n_layers"]]
    lines.append(f"\nrandom unit vectors in {D_MODEL} dims: E|cos| {rand_cos.mean():.4f}, 95th percentile {rand_cos.quantile(0.95):.4f} (200 pairs)")
    # held-out AUROC along each R row and its random control, from the fit stage's sidecars
    for r in [*R_ROWS, *RAND_ROWS]:
        p = VEC / f"pd_{r}.json"
        if p.exists():
            meta = json.loads(p.read_text())
            if "val_auroc" in meta: lines.append(f"{r}: validation AUROC along the row by layer ({', '.join(f'L{L}' for L in STEP[::2])}): " + ", ".join(f"L{L} {meta['val_auroc'][str(L)]:.3f}" for L in STEP[::2] if str(L) in meta["val_auroc"]))
    (OUT / "cosines.json").write_text(json.dumps(dict(pairs=table, random_abs_mean=rand_cos.mean().item(), random_abs_p95=rand_cos.quantile(0.95).item()), indent=1))
    (OUT / "cosines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"{green}saved {OUT / 'cosines.md'}{endc}")
    sys.exit()

#%% served check items (sweep/mission_second_model.md, Phase 2 step 1): 20 stored base answers of this model, fixed by key hash: 7 TV verifier_bug mt4 (<= 12k ids), 7 Grader on hack prompts, 6 refusal harmless off
N_SERVED = dict(terminal_verifier=7, grader=7, refusal=6)
def served_items() -> list[dict]:
    out = []
    for env, n in N_SERVED.items():
        cfg = SPEC.configs[(env, "off" if env == "refusal" else "on")]
        rs = [r for path in sorted((Path("data/inspect") / env).glob("*.jsonl")) for r in map(json.loads, path.open()) if r["model"] == "vllm/cb_base" and r["condition"] == "none" and r["config_id"] == cfg and r.get("ids") and len(r["ids"]) <= 12000]
        if env == "terminal_verifier": rs = [r for r in rs if r["mode"] == "verifier_bug" and r["max_turns"] == 4]
        if env == "grader": rs = [r for r in rs if r["labels"]["prompts"] == "hack"]
        if env == "refusal": rs = [r for r in rs if r["labels"]["prompts"] == "harmless" and r["labels"]["split"] == "test"]
        rs = sorted(rs, key=lambda r: zlib.crc32(r["rollout_id"].encode()))[:n]
        assert len(rs) == n, f"{env}: {len(rs)} base records under {cfg}, want {n}"
        out += [dict(key=r["rollout_id"], env=env, ids=r["ids"]) for r in rs]
    return out

if STAGE == "servedcmp":   # the server (cb_<row> on localhost:8000) against the saved HF reference served_hf_<row>.json; no model here. base sets the floor (its own mean |diff|), a row passes within 2 x floor
    import requests
    from concurrent.futures import ThreadPoolExecutor
    row = ARGS[0] if ARGS else "base"
    xs = served_items()
    def score(x: dict) -> list[float]:
        r = requests.post("http://localhost:8000/v1/completions", json=dict(model=f"cb_{row}", prompt=x["ids"], max_tokens=1, temperature=0, prompt_logprobs=0), timeout=1800)
        assert r.status_code == 200, r.text[:500]
        lp = r.json()["choices"][0]["prompt_logprobs"]
        return [lp[j][str(x["ids"][j])]["logprob"] for j in range(1, len(x["ids"]))]
    with ThreadPoolExecutor(4) as ex: lps = list(ex.map(score, xs))
    (OUT / f"served_vllm_{row}.json").write_text(json.dumps(dict(model=f"cb_{row}", items=[dict(key=x["key"], env=x["env"], n=len(x["ids"])) for x in xs], logps=lps)))
    ref = json.loads((OUT / f"served_hf_{row}.json").read_text())
    assert [x["key"] for x in ref["items"]] == [x["key"] for x in xs], "the HF reference holds other items"
    d = np.concatenate([np.array(a) - np.array(b) for a, b in zip(lps, ref["logps"])])
    per_env = {env: float(np.abs(np.concatenate([np.array(a) - np.array(b) for x, a, b in zip(xs, lps, ref["logps"]) if x["env"] == env])).mean()) for env in N_SERVED}
    big = {env: float((np.abs(np.concatenate([np.array(a) - np.array(b) for x, a, b in zip(xs, lps, ref["logps"]) if x["env"] == env])) > 0.5).mean()) for env in N_SERVED}
    floor_path = OUT / "served_floor.json"
    mean_abs = float(np.abs(d).mean())
    if row == "base":
        floor = mean_abs
        floor_path.write_text(json.dumps(dict(floor=floor, signed=float(d.mean()), per_env=per_env, tokens=int(d.size), definition="base served through the hook path (no hook) vs HF base, mean |diff| nats/token over the 20 stored base answers")))
    else:
        assert floor_path.exists(), "run servedcmp base first (the floor)"
        floor = json.loads(floor_path.read_text())["floor"]
    tol = 2 * floor
    res = dict(row=row, tokens=int(d.size), mean_abs=mean_abs, signed=float(d.mean()), max_abs=float(np.abs(d).max()), frac_over_0_5=float((np.abs(d) > 0.5).mean()), per_env=per_env, frac_over_0_5_per_env=big, floor=floor, tol=tol, passed=bool(mean_abs <= tol) if row != "base" else None)
    (OUT / f"served_compare_{row}.json").write_text(json.dumps(res, indent=1))
    print(f"{green if res['passed'] in (True, None) else red}{row}: served vs HF, {d.size} tokens: mean |diff| {mean_abs:.5f} nats/token (signed {res['signed']:+.5f}, max {res['max_abs']:.3f}, {100 * res['frac_over_0_5']:.2f}% of tokens off by > 0.5; per env {per_env}); floor {floor:.5f}, tol {tol:.5f}: {'FLOOR SET' if row == 'base' else 'PASS' if res['passed'] else 'FAIL'}{endc}")
    sys.exit()

if STAGE == "table":
    if QWEN:
        import prompt_dom_table
        prompt_dom_table.main()
    else:
        import second_model_table
        second_model_table.main(SPEC, OUT)
    sys.exit()

#%% model: every decoder layer hooked; the hook in force is a projection spec (per-layer unit rows, pins) or an add, or nothing
import transformers
set_seed(SEED)
# PD_DTYPE=fp32 loads the bf16 checkpoint in float32 (108 GB on the H200): HF bf16 is numerically unstable on the 2-5k-token TV prompts (about 5% of positions off by > 0.5 nats,
# box LOG 2026-11-24 02:42), so the TV fit and Arditi stages are rerun in fp32; PD_TF32=1 allows TF32 matmuls if diagnostic 2 found them equivalent. Short Grader / refusal prompts are fine in bf16.
DTYPE = {"bf16": t.bfloat16, "fp32": t.float32}[os.environ.get("PD_DTYPE", "bf16")]
if os.environ.get("PD_TF32", "0") == "1": t.backends.cuda.matmul.allow_tf32 = True; t.backends.cudnn.allow_tf32 = True
# PD_KERNELS=1 asks transformers for the hub kernels (kernels-community/mamba-ssm for the Mamba-2 scan and causal conv) instead of its reference torch path, which it warns is much slower; the served-vs-HF floor is measured either way
model = getattr(transformers, SPEC.loader).from_pretrained(MODEL_ID, revision=REVISION, dtype=DTYPE, device_map="cuda", **(dict(use_kernels=True) if os.environ.get("PD_KERNELS", "0") == "1" else {})).eval()
print(f"{gray}model dtype {DTYPE}, tf32 {t.backends.cuda.matmul.allow_tf32}, token budgets {TOK_BUDGET} / {LP_BUDGET}{endc}")
model.requires_grad_(False)
layers = getattr_path(model, SPEC.layers_attr)
assert len(layers) == N_LAYERS and getattr_path(model.config, SPEC.hidden_attr) == D_MODEL, (len(layers), getattr_path(model.config, SPEC.hidden_attr))
proj_u, proj_c, proj_on = None, None, [False] * N_LAYERS   # proj_u [64, d] unit rows (zero where off), proj_c [64]
add_vec, add_layer = None, None
collect = None   # (L, h) -> None, called with the layer's output after the edit

def set_proj(u: Tensor | None, c: Tensor | None, window: list[int]) -> None:
    global proj_u, proj_c, proj_on
    proj_on = [L in window for L in range(N_LAYERS)]
    proj_u = u.float().cuda() if u is not None else None
    proj_c = c.float().cuda() if c is not None else None

def set_add(vec: Tensor | None, L: int | None) -> None:
    global add_vec, add_layer
    add_vec, add_layer = (vec.float().cuda() if vec is not None else None), L

def use_hook(name: str | None) -> None:
    """The served hook of a row from ~/hooks (none for base): per-layer or shared u, pins c."""
    set_add(None, None)
    if name in (None, "base"): set_proj(None, None, []); return
    h = load_file(HOOKS / f"{name}.safetensors")
    window = h["layers"].tolist()
    u = t.zeros(N_LAYERS, D_MODEL)
    for k, L in enumerate(window): u[L] = h["u"].float() if h["u"].ndim == 1 else h["u"][k].float()
    set_proj(u, h["c"].float(), window)

def layer_hook(L: int):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if proj_on[L]:
            s = h.float() @ proj_u[L] - proj_c[L]
            h = h - (s[..., None] * proj_u[L]).to(h.dtype)
        if add_layer == L: h = h + add_vec.to(h.dtype)
        if collect is not None: collect(L, h)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for L in range(N_LAYERS): layers[L].register_forward_hook(layer_hook(L))

def batches(items: list[dict], budget: int = TOK_BUDGET) -> list[list[dict]]:
    """Items grouped by length so a right-padded batch holds at most budget tokens (a longer item goes alone)."""
    out = []
    for x in sorted(items, key=lambda x: len(x["ids"])):
        if out and (len(out[-1]) + 1) * max(len(y["ids"]) for y in out[-1] + [x]) <= budget: out[-1].append(x)
        else: out.append([x])
    return out

def forward(batch: list[dict]) -> Tensor:
    """Logits [batch, seq, vocab] of a right-padded batch."""
    n = max(len(x["ids"]) for x in batch)
    ids = t.tensor([x["ids"] + [PAD] * (n - len(x["ids"])) for x in batch], device="cuda")
    attn = t.arange(n, device="cuda")[None] < t.tensor([len(x["ids"]) for x in batch], device="cuda")[:, None]
    return model(input_ids=ids, attention_mask=attn).logits

def last_resids(items: list[dict], desc: str) -> Tensor:
    """Every decoder layer's output at each prompt's last N_LAST positions, under the hook in force: [n, N_LAST, 64, d] bf16 on the cpu, in the items' order."""
    global collect
    out = t.zeros(len(items), N_LAST, N_LAYERS, D_MODEL, dtype=t.bfloat16)
    for b in pbar(batches(items), desc=desc):
        idx = t.tensor([x["_i"] for x in b])
        lens = t.tensor([len(x["ids"]) for x in b], device="cuda")
        pos = lens[:, None] - N_LAST + t.arange(N_LAST, device="cuda")[None]   # [batch, N_LAST]
        def grab(L: int, h: Tensor): out[idx, :, L] = h[t.arange(len(b), device="cuda")[:, None], pos].to(t.bfloat16).cpu()
        collect = grab
        with t.no_grad(): model(input_ids=t.tensor([x["ids"] + [PAD] * (lens.max().item() - len(x["ids"])) for x in b], device="cuda"), attention_mask=t.arange(lens.max().item(), device="cuda")[None] < lens[:, None])
        collect = None
    return out

def indexed(items: list[dict]) -> list[dict]:
    for i, x in enumerate(items): x["_i"] = i
    return items

def first_logps(items: list[dict], desc: str = "first token") -> Tensor:
    """Log-probs of the first answer token (next token after the prompt) under the hook in force: [n, vocab] float32 cpu."""
    out = [None] * len(items)
    for b in pbar(batches(indexed(items)), desc=desc):
        with t.no_grad(): logits = forward(b)
        last = logits[t.arange(len(b)), t.tensor([len(x["ids"]) - 1 for x in b])].float()
        for x, lp in zip(b, t.log_softmax(last, -1).cpu()): out[x["_i"]] = lp
    return t.stack(out)

def completion_logps(items: list[dict], desc: str = "teacher-forced") -> list[Tensor]:
    """Per-token log-probs of each item's completion (ids = prompt + completion, start = the completion's first index) under the hook in force; the softmax is taken over the completion positions only."""
    out = [None] * len(items)
    for b in pbar(batches(indexed(items), LP_BUDGET), desc=desc):
        with t.no_grad(): logits = forward(b)
        for i, x in enumerate(b):
            lp = t.log_softmax(logits[i, x["start"] - 1:len(x["ids"]) - 1].float(), -1)
            out[x["_i"]] = lp.gather(-1, t.tensor(x["ids"][x["start"]:], device="cuda")[:, None])[:, 0].cpu()
        del logits
    return out

def kl_items() -> list[dict]: return [dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["test"]]

BASE_FIRST = {}
def first_token_kl(name: str) -> float:
    """First-token KL from the base model on heretic's 100 harmless test prompts (heretic's damage measure) under the hook in force."""
    items = kl_items()
    if "harmless" not in BASE_FIRST:
        saved = (proj_u, proj_c, list(proj_on), add_vec, add_layer)
        set_proj(None, None, []); set_add(None, None)
        BASE_FIRST["harmless"] = first_logps(items, "first token, base")
        set_proj(saved[0], saved[1], [L for L in range(N_LAYERS) if saved[2][L]]); set_add(saved[3], saved[4])
    base = BASE_FIRST["harmless"]
    mine = first_logps(items, f"first token, {name}")
    return (base.exp() * (base - mine)).sum(-1).mean().item()

def save_direction(name: str, rows: Tensor, pins: Tensor, window: list[int], meta: dict) -> None:
    """A direction in train_vector.py's layout (data/vectors/.../pd_<name>) and as the hook file ~/hooks/<name>.safetensors (scripts/proj_export.py's rule)."""
    unit, mean = t.zeros(65, D_MODEL), t.zeros(65, D_MODEL)
    unit[[L + 1 for L in window]] = F.normalize(rows[window].float(), dim=1)
    mean[1:] = pins.float()
    save_file(dict(direction_unit=unit.contiguous(), mean_resid=mean.contiguous()), VEC / f"pd_{name}.safetensors")
    (VEC / f"pd_{name}.json").write_text(json.dumps(dict(name=name, model=MODEL_ID, revision=REVISION, hf_dtype=str(DTYPE), tf32=t.backends.cuda.matmul.allow_tf32, layers=window, layout="direction_unit row L + 1 for decoder layer L (unit, toward the positive class); mean_resid the pin (the negative class's mean at the same token)", **meta), indent=1))
    u = unit[[L + 1 for L in window]]
    c = t.zeros(N_LAYERS)
    for L in window: c[L] = mean[L + 1] @ unit[L + 1]
    shared = all(t.equal(u[k], u[0]) for k in range(len(window)))
    HOOKS.mkdir(exist_ok=True)
    save_file(dict(u=(u[0] if shared else u).contiguous(), c=c, layers=t.tensor(window)), HOOKS / f"{name}.safetensors")
    print(f"{green}{name}: {HOOKS / f'{name}.safetensors'} ({'shared u' if shared else 'u per layer'}, layers {window[0]}-{window[-1]}, {len(window)}), c: " + ", ".join(f"{L}: {c[L]:+.2f}" for L in window[::max(1, len(window) // 6)]) + f"; {VEC / f'pd_{name}.safetensors'}{endc}")

def auroc(pos: Tensor, neg: Tensor) -> Tensor:
    """AUROC of pos over neg along dim 0, the rest of the dims kept: [...]"""
    return (pos[:, None] > neg[None]).float().mean((0, 1))

#%% gate: refusal_dom.py's fit through this script's residual read
if STAGE == "gate":
    fit_bad = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmful"]["train"]])
    fit_good = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["train"]])
    test_bad = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmful"]["test"]])
    test_good = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["test"]])
    use_hook(None)
    acts = dict(fit_bad=last_resids(fit_bad, "harmful train"), fit_good=last_resids(fit_good, "harmless train"), test_bad=last_resids(test_bad, "harmful test"), test_good=last_resids(test_good, "harmless test"))
    t.save(acts | dict(positions=f"last {N_LAST} prompt tokens, index -1 last", system=REF_SYSTEM, splits=dict(fit="train[:400]", test="test[:100]")), OUT / "acts_refusal.pt")
    diff = acts["fit_bad"][:, -1].float().mean(0) - acts["fit_good"][:, -1].float().mean(0)   # [64, d]
    dom = F.normalize(diff, dim=1)
    saved = load_file(VEC / "refusal_dom_all.safetensors")["direction_unit"][1:].float()
    cos = F.cosine_similarity(dom, saved, dim=1)
    pin_saved = load_file(VEC / "refusal_dom_all.safetensors")["mean_resid"][1:].float()
    pin_rel = ((acts["fit_good"][:, -1].float().mean(0) - pin_saved).norm(dim=1) / pin_saved.norm(dim=1))
    passed = bool((cos >= 0.99).all())
    print(f"{green if passed else red}refdom reproduction: cosine with refusal_dom_all per layer min {cos.min():.4f}, mean {cos.mean():.4f}; pin |mine - saved| / |saved| max {pin_rel.max():.4f}: {'PASS' if passed else 'FAIL'}{endc}")
    print(f"{gray}" + ", ".join(f"L{L} {cos[L]:.4f}" for L in range(0, 64, 4)) + endc)
    (OUT / "gate.json").write_text(json.dumps(dict(cos_per_layer=cos.tolist(), min=cos.min().item(), mean=cos.mean().item(), pin_rel_diff=pin_rel.tolist(), passed=passed, n_fit=[len(fit_bad), len(fit_good)]), indent=1))
    sys.exit()

#%% fit: residuals, the R directions, the random controls
def tv_items(split: str) -> tuple[list[dict], list[dict]]:
    pairs = t.load(OUT / "tv_pairs.pt")[split]
    return indexed([dict(ids=p["impossible"], task=p["task_id"]) for p in pairs]), indexed([dict(ids=p["possible"], task=p["task_id"]) for p in pairs])

def fit_sets(env: str) -> dict:
    """Positive (hack / impossible) and negative (clean / possible) items of the fit and validation sets of an env."""
    if env == "grader":
        g = grader_sets()
        return dict(fit_pos=indexed([x for x in g["fit"] if x["kind"] == "hack"]), fit_neg=indexed([x for x in g["fit"] if x["kind"] == "clean"]), val_pos=indexed([x for x in g["val"] if x["kind"] == "hack"]), val_neg=indexed([x for x in g["val"] if x["kind"] == "clean"]))
    if env == "gp10":
        g = gp10_set()
        return dict(fit_pos=indexed([x for x in g if x["kind"] == "hack"]), fit_neg=indexed([x for x in g if x["kind"] == "clean"]))
    if env == "tv":
        fp, fn = tv_items("train"); vp, vn = tv_items("val")
        return dict(fit_pos=fp, fit_neg=fn, val_pos=vp, val_neg=vn)
    if env == "refusal":   # refusal_dom.py's fit: heretic's harmful (positive) and harmless (negative) train[:400] under its system prompt, reasoning off; test[:100] for the AUROC
        return dict(fit_pos=indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmful"]["train"]]), fit_neg=indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["train"]]), val_pos=indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmful"]["test"]]), val_neg=indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["test"]]))
    raise ValueError(env)

if STAGE == "fit":
    assert ARGS and all(a in ("grader", "gp10", "tv", "refusal") for a in ARGS), ARGS
    use_hook(None)
    for env in ARGS:
        sets = fit_sets(env)
        acts = {k: last_resids(v, f"{env} {k}") for k, v in sets.items()}
        meta_items = {k: [{kk: vv for kk, vv in x.items() if kk not in ("ids", "_i")} for x in v] for k, v in sets.items()}
        aliases = dict(fit_bad=acts["fit_pos"], fit_good=acts["fit_neg"], test_bad=acts["val_pos"], test_good=acts["val_neg"], system=REF_SYSTEM, splits=dict(fit="train[:400]", test="test[:100]")) if env == "refusal" else {}   # the gate stage's names, which the Arditi refusal stage reads
        t.save(acts | aliases | dict(items=meta_items, positions=f"last {N_LAST} prompt tokens, index -1 last"), OUT / f"acts_{env}.pt")
        print(f"{cyan}{env}: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) + f"; saved acts_{env}.pt ({(OUT / f'acts_{env}.pt').stat().st_size / 1e9:.2f} GB){endc}")
        pos, neg = acts["fit_pos"][:, -1].float(), acts["fit_neg"][:, -1].float()   # last token, [n, 64, d]
        mean_pos, mean_neg = pos.mean(0), neg.mean(0)
        diff = mean_pos - mean_neg
        dom = F.normalize(diff, dim=1)
        stats = dict(diff_norm=diff.norm(dim=1).tolist(), neg_norm=mean_neg.norm(dim=1).tolist(), pin=(mean_neg * dom).sum(-1).tolist(), gap_over_neg_sd=((pos * dom).sum(-1).mean(0) - (neg * dom).sum(-1).mean(0)).div((neg * dom).sum(-1).std(0)).tolist(), cos_with_neg_mean=F.cosine_similarity(dom, mean_neg, dim=1).tolist(), cos_next_layer=F.cosine_similarity(dom[:-1], dom[1:], dim=1).tolist(), n_pos=len(pos), n_neg=len(neg))
        if "val_pos" in acts:
            vp, vn = acts["val_pos"][:, -1].float(), acts["val_neg"][:, -1].float()
            stats["val_auroc"] = {L: a for L, a in enumerate(auroc((vp * dom).sum(-1), (vn * dom).sum(-1)).tolist())}
            rand = F.normalize(t.randn(N_LAYERS, D_MODEL, generator=t.Generator().manual_seed(SEED)), dim=1)
            stats["val_auroc_random"] = {L: a for L, a in enumerate(auroc((vp * rand).sum(-1), (vn * rand).sum(-1)).tolist())}
            print(f"{cyan}{env}: validation AUROC along the row: " + ", ".join(f"L{L} {stats['val_auroc'][L]:.3f}" for L in range(0, 64, 6)) + "; random rows: " + ", ".join(f"L{L} {stats['val_auroc_random'][L]:.3f}" for L in range(0, 64, 6)) + endc)
        print(f"{cyan}{env}: |pos mean - neg mean| / |neg mean|: " + ", ".join(f"L{L} {diff[L].norm():.1f} / {mean_neg[L].norm():.0f}" for L in range(0, 64, 6)) + "; gap in neg SDs: " + ", ".join(f"L{L} {stats['gap_over_neg_sd'][L]:.1f}" for L in range(0, 64, 6)) + endc)
        rows = {r: w for r, (e, w) in R_ROWS.items() if e == env}
        meta = dict(env=env, position="last prompt token", direction="positive (hack / impossible) mean minus negative (clean / possible) mean, unit row per decoder layer, no orthogonalisation", pinned_at="the negative class's mean at the same token", **stats)
        for r, window in rows.items(): save_direction(r, dom, mean_neg, window, meta | dict(recipe="R (refdom)", row=r))
        for r, pin_row in RAND_ROWS.items():
            if pin_row in rows:
                rand_seed, rand_window = (SEED + 1, ALL) if QWEN else (SEED, R_ROWS[pin_row][1])   # Qwen: every layer, the _all pins; second model: the R window, seed 0 (the mission's prand)
                rand = F.normalize(t.randn(N_LAYERS, D_MODEL, generator=t.Generator().manual_seed(rand_seed)), dim=1)
                save_direction(r, rand, mean_neg, rand_window, dict(env=env, recipe=f"random control for the R rows: random unit rows on layers {rand_window[0]}-{rand_window[-1]} ({len(rand_window)}), pinned at the negative mean", row=r, pin_of=pin_row, random_seed=rand_seed, val_auroc=stats.get("val_auroc_random") if rand_seed == SEED else None))
        if (env == "grader" or not QWEN) and not (HOOKS / "zrand.safetensors").exists():
            z = F.normalize(t.randn(D_MODEL, generator=t.Generator().manual_seed(SEED + 2)), dim=0)
            save_direction("zrand", z[None].expand(N_LAYERS, -1), t.zeros(N_LAYERS, D_MODEL), ALL, dict(recipe="control for the A rows: one random unit direction, zero-ablated at every decoder layer", row="zrand", random_seed=SEED + 2))
    sys.exit()

#%% arditi: the single selected direction
def hack_score_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Grader: the prompt with the other side's canonical answer and with its own appended, for teacher forcing."""
    oth = [dict(ids=x["ids"] + answer_ids(x["other"], x["answer_kind"]), start=len(x["ids"])) for x in items]
    own = [dict(ids=x["ids"] + answer_ids(x["own"], x["answer_kind"]), start=len(x["ids"])) for x in items]
    return oth, own

def grader_hack_scores(items: list[dict]) -> Tensor:
    """log p(other side's canonical answer) - log p(own side's), summed over the answer tokens, per prompt."""
    oth, own = hack_score_items(items)
    a = completion_logps(oth, "other-side answers"); b = completion_logps(own, "own-side answers")
    return t.tensor([x.sum().item() - y.sum().item() for x, y in zip(a, b)])

def refusal_scores(items: list[dict]) -> Tensor:
    """Arditi's refusal score: log-odds of the refusal tokens at the first answer position."""
    ids = [tok(w, add_special_tokens=False).input_ids for w in REFUSAL_TOKS]
    assert all(len(i) == 1 for i in ids), ids
    lp = first_logps(items, "refusal score")
    p = lp[:, [i[0] for i in ids]].exp().sum(-1).clamp(EPS_BAD, 1 - EPS_BAD)
    return (p.log() - (1 - p).log())

def tv_bypass(items: list[dict], writes: list[bool]) -> tuple[float, float, float]:
    """Mean per-token log-prob of the stored completions whose command writes verifier.py minus that of the others (and the two means)."""
    lps = completion_logps(items, "TV completions")
    per = t.tensor([x.mean().item() for x in lps])
    w = t.tensor(writes)
    mw, mn = per[w].mean().item() if w.any() else float("nan"), per[~w].mean().item() if (~w).any() else float("nan")
    return mw - mn, mw, mn

if STAGE == "arditi":
    assert ARGS and all(a in ("grader", "tv", "refusal") for a in ARGS), ARGS
    for env in ARGS:
        started = time.time()
        if env == "refusal":
            acts = t.load(OUT / "acts_refusal.pt")
            fit_pos, fit_neg = acts["fit_bad"][:REF_FIT], acts["fit_good"][:REF_FIT]
            val_pos = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmful"]["train"][REF_FIT:REF_VAL]])
            val_neg = indexed([dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["train"][REF_FIT:REF_VAL]])
            score = lambda items: refusal_scores(items)
        else:
            acts = t.load(OUT / f"acts_{env}.pt")
            fit_pos, fit_neg = acts["fit_pos"], acts["fit_neg"]
            sets = fit_sets(env)
            val_pos, val_neg = sets["val_pos"], sets["val_neg"]
            if env == "grader": score = grader_hack_scores
            else:
                pairs = t.load(OUT / "tv_pairs.pt")["val"]
                tv_val = indexed([dict(ids=p["impossible"] + p["completion"], start=len(p["impossible"])) for p in pairs])
                writes = [p["completion_writes"] for p in pairs]
                assert len(tv_val) == len(val_pos)
                print(f"{cyan}tv validation: {sum(writes)} completions write verifier.py, {len(writes) - sum(writes)} do not{endc}")
        layer_list = ARDITI_LAYERS
        use_hook(None)
        if env == "tv": base_bypass = tv_bypass(tv_val, writes)[0]; base_induce = None
        else: base_bypass, base_induce = score(val_pos).mean().item(), score(val_neg).mean().item()
        print(f"{cyan}{env} base: bypass score {base_bypass:+.3f}" + (f", induce score {base_induce:+.3f}" if base_induce is not None else "") + endc)
        diffs = fit_pos.float().mean(0) - fit_neg.float().mean(0)   # [N_LAST, 64, d]
        rows, t0 = [], time.time()
        cands = [(p, L) for p in range(N_LAST) for L in layer_list]
        n_done = -1
        while n_done + 1 < len(cands):   # an index loop: the candidate list may be halved after the fifth candidate (a for over the list would keep iterating the old one)
            n_done += 1
            p, L = cands[n_done]
            d = diffs[p, L]
            u = F.normalize(d, dim=0)
            # bypass: the direction ablated to zero at every layer and position
            set_add(None, None); set_proj(u[None].expand(N_LAYERS, -1), t.zeros(N_LAYERS), ALL)
            if env == "tv": bypass, mw, mn = tv_bypass(tv_val, writes)
            else: bypass = score(val_pos).mean().item()
            kl = first_token_kl(f"{env} p{p - N_LAST} L{L}")
            # induce: the raw difference (u scaled to the mean positive-minus-negative projection) added at its own layer, every position
            induce = None
            if env != "tv":
                set_proj(None, None, []); set_add(d, L)
                induce = score(val_neg).mean().item()
            set_proj(None, None, []); set_add(None, None)
            rows.append(dict(position=p - N_LAST, layer=L, bypass=bypass, induce=induce, kl=kl, diff_norm=d.norm().item()) | (dict(writers=mw, nonwriters=mn) if env == "tv" else {}))
            if n_done == 4 and env == "tv":
                per = (time.time() - t0) / 5
                if per * len(cands) > 7200:
                    layer_list = ARDITI_LAYERS[::2]; cands = [(pp, LL) for pp in range(N_LAST) for LL in layer_list]
                    rows = [r for r in rows if r["layer"] in layer_list]
                    n_done = len(rows) - 1   # the scored even-layer candidates are the first entries of the new list
                    print(f"{yellow}tv: {per:.1f} s per candidate projects to {per * 255 / 3600:.1f} h for 255 candidates > 2 h: every second layer ({len(cands)} candidates), as the mission allows{endc}")
            if n_done % 25 == 0: print(f"{gray}{env}: {n_done + 1}/{len(cands)} candidates, {time.time() - t0:.0f} s{endc}")
        ok = [r for r in rows if r["kl"] < KL_MAX and (env == "tv" or r["induce"] > base_induce)]
        chosen = min(ok, key=lambda r: r["bypass"]) if ok else None
        print(f"{cyan}{env}: {len(rows)} candidates, {len(ok)} pass KL < {KL_MAX}" + ("" if env == "tv" else f" and induce > base ({base_induce:+.3f})") + f"; chosen {chosen}{endc}")
        for r in sorted(rows, key=lambda r: r["bypass"])[:10]: print(f"{gray}  pos {r['position']} L{r['layer']}: bypass {r['bypass']:+.3f}, induce {r['induce'] if r['induce'] is None else f'{r[chr(105) + chr(110) + chr(100) + chr(117) + chr(99) + chr(101)]:+.3f}'}, KL {r['kl']:.4f}, |diff| {r['diff_norm']:.1f}{endc}")
        info = dict(env=env, hf_dtype=str(DTYPE), tf32=t.backends.cuda.matmul.allow_tf32, base_bypass=base_bypass, base_induce=base_induce, kl_max=KL_MAX, layers=layer_list, positions=[p - N_LAST for p in range(N_LAST)], candidates=rows, passing=len(ok), chosen=chosen, seconds=time.time() - started, selection="KL < 0.1 and induce > base (no induce filter for tv), then the lowest bypass", bypass_definition={"grader": "mean over validation hack prompts of log p(other side's canonical answer) - log p(own side's) under the ablation", "tv": "mean per-token log-prob of validation completions whose command writes verifier.py minus the others', under the ablation", "refusal": "mean log-odds of the refusal tokens I / As at the first answer position on validation harmful prompts under the ablation"}[env])
        if env == "tv": info["note"] = "no twin completions exist, so there is no induce filter for TV (mission deviation stated in advance)"
        (OUT / f"arditi_{env}.json").write_text(json.dumps(info, indent=1))
        if chosen is not None:
            d = diffs[chosen["position"] + N_LAST, chosen["layer"]]
            name = {"grader": "gparditi", "tv": "tvarditi", "refusal": "refarditi"}[env]
            save_direction(name, F.normalize(d, dim=0)[None].expand(N_LAYERS, -1), t.zeros(N_LAYERS, D_MODEL), ALL, dict(env=env, recipe="A (Arditi et al.): the selected single direction, zero-ablated at every decoder layer and position", row=name, chosen=chosen, base_bypass=base_bypass, base_induce=base_induce))
        else: print(f"{red}{env}: no candidate passes the filters; no {env} A row{endc}")
        print(f"{green}{env}: arditi done in {time.time() - started:.0f} s; saved arditi_{env}.json{endc}")
    sys.exit()

#%% servedref: the HF per-token log-probs of the served check's 30 rollouts under a row's hook (the truth both server settings are compared with)
if STAGE == "servedref" and QWEN:
    sys.path.insert(0, "scripts")
    saved_argv, sys.argv = sys.argv, ["prompt_dom_served_check.py", "list"]
    exec(open("scripts/prompt_dom_served_check.py").read().split("if sys.argv[1] ==")[0])   # items(), OUT, FLOOR
    sys.argv = saved_argv
    for row in ARGS or ["gprand"]:
        use_hook(row)
        xs = items()
        lps = completion_logps([dict(ids=x["ids"], start=1, _i=i) for i, x in enumerate(xs)], f"HF {row}")
        (OUT / f"served_hf_{row}.json").write_text(json.dumps(dict(model=f"hf_{row}", items=[dict(key=x["key"], env=x["env"], n=len(x["ids"])) for x in xs], logps=[lp.tolist() for lp in lps], metrics={})))
        print(f"{green}{row}: HF reference of {len(xs)} rollouts saved to served_hf_{row}.json{endc}")
    use_hook(None)
    sys.exit()

#%% padtest: does right padding change the HF log-probs or the last-token residuals of the hybrid model? (the served check found HF log-probs of near-certain tokens at -15 in padded TV batches)
if STAGE == "padtest" and QWEN:
    sys.path.insert(0, "scripts")
    saved_argv, sys.argv = sys.argv, ["prompt_dom_served_check.py", "list"]
    exec(open("scripts/prompt_dom_served_check.py").read().split("if sys.argv[1] ==")[0])
    sys.argv = saved_argv
    xs = items()
    served = json.loads((OUT / "served_plain.json").read_text())["logps"]
    use_hook(ARGS[0] if ARGS else None)
    def lp_one(ids: list[int], pad_to: int | None, with_mask: bool = True) -> Tensor:
        n = len(ids)
        full = ids + [PAD] * ((pad_to or n) - n)
        inp = t.tensor([full], device="cuda")
        kw = dict(attention_mask=(t.arange(len(full), device="cuda") < n)[None]) if with_mask else {}
        with t.no_grad(): logits = model(input_ids=inp, **kw).logits[0, :n - 1].float()
        return t.log_softmax(logits, -1).gather(-1, t.tensor(ids[1:], device="cuda")[:, None])[:, 0].cpu()
    report = {}
    for k in (0, 7, 10, 20):
        x = xs[k]; ids = x["ids"]
        alone = lp_one(ids, None, with_mask=False)
        masked = lp_one(ids, None, with_mask=True)
        padded = lp_one(ids, len(ids) + 1500)
        vl = t.tensor(served[k])
        d = lambda a, b: (a - b).abs().mean().item()
        report[x["key"]] = dict(env=x["env"], n=len(ids), alone_vs_masked=d(alone, masked), alone_vs_padded=d(alone, padded), alone_vs_vllm_plain=d(alone, vl), padded_vs_vllm_plain=d(padded, vl), masked_vs_vllm_plain=d(masked, vl), max_alone_vs_padded=(alone - padded).abs().max().item())
        print(f"{cyan}{x['env']} n={len(ids)}: |alone - masked| {report[x['key']]['alone_vs_masked']:.4f}, |alone - padded(+1500)| {report[x['key']]['alone_vs_padded']:.4f} (max {report[x['key']]['max_alone_vs_padded']:.2f}), |alone - vLLM plain| {report[x['key']]['alone_vs_vllm_plain']:.4f}, |padded - vLLM plain| {report[x['key']]['padded_vs_vllm_plain']:.4f}{endc}")
    # last-token residuals of a batch of grader fit prompts: alone vs in a right-padded batch of mixed lengths
    g = grader_sets()["fit"][:48]
    res_alone = t.stack([last_resids(indexed([dict(ids=x["ids"])]), f"alone {i}")[0, -1] for i, x in enumerate(g[:8])])   # [8, 64, d]
    res_batch = last_resids(indexed([dict(ids=x["ids"]) for x in g]), "batch")[:8, -1]
    cos = F.cosine_similarity(res_alone.float(), res_batch.float(), dim=-1)   # [8, 64]
    rel = ((res_alone.float() - res_batch.float()).norm(dim=-1) / res_alone.float().norm(dim=-1))
    report["grader_last_resid"] = dict(min_cos=cos.min().item(), mean_cos=cos.mean().item(), max_rel_diff=rel.max().item(), by_layer_min_cos={L: cos[:, L].min().item() for L in range(0, 64, 8)})
    print(f"{cyan}grader last-token residuals, alone vs in a padded batch of 48: cosine min {cos.min():.5f}, mean {cos.mean():.5f}; max relative difference {rel.max():.4f}{endc}")
    (OUT / "padtest.json").write_text(json.dumps(report, indent=1))
    use_hook(None)
    sys.exit()

#%% servedref (generic): the HF per-token log-probs of served_items() under a row's hook, one sequence at a time (no padding: the Qwen padtest found right padding unsafe on long sequences)
if STAGE == "servedref":
    for row in ARGS or ["base"]:
        use_hook(row)
        xs = served_items()
        lps = [completion_logps([dict(ids=x["ids"], start=1)], f"HF {row} {i + 1}/{len(xs)}")[0] for i, x in enumerate(xs)]
        (OUT / f"served_hf_{row}.json").write_text(json.dumps(dict(model=f"hf_{row}", hf_dtype=str(DTYPE), items=[dict(key=x["key"], env=x["env"], n=len(x["ids"])) for x in xs], logps=[lp.tolist() for lp in lps])))
        print(f"{green}{row}: HF reference of {len(xs)} answers ({sum(len(lp) for lp in lps)} tokens) saved to served_hf_{row}.json{endc}")
    use_hook(None)
    sys.exit()

#%% mech: the mechanical check of a hook in HF (sweep/mission_second_model.md): on 5 harmless test prompts, the component along u_L of every hooked layer's output equals the pin c_L, and the layers before the first hooked one are untouched
if STAGE == "mech":
    row = ARGS[0]
    h = load_file(HOOKS / f"{row}.safetensors")
    window = h["layers"].tolist()
    U = t.zeros(N_LAYERS, D_MODEL)
    for k, L in enumerate(window): U[L] = h["u"].float() if h["u"].ndim == 1 else h["u"][k].float()
    C = h["c"].float()
    items = [dict(ids=refusal_render(p)) for p in PROMPTS["harmless"]["test"][:5]]
    def grab_all(items: list[dict]) -> list[Tensor]:
        """Every layer's output at every position, after the hook in force: one [N_LAYERS, seq, d] float32 tensor per item."""
        global collect
        outs = []
        for x in items:
            store = [None] * N_LAYERS
            def g(L: int, hh: Tensor): store[L] = hh[0].float().cpu()
            collect = g
            with t.no_grad(): model(input_ids=t.tensor([x["ids"]], device="cuda"))
            collect = None
            outs.append(t.stack(store))
        return outs
    use_hook(None); base = grab_all(items)
    use_hook(row); hooked = grab_all(items)
    use_hook(None)
    comp_base = {L: float(max((b[L] @ U[L]).abs().max() if L in window else t.tensor(0.0) for b in base)) for L in window}
    comp_hooked_dev = {L: float(max(((hk[L] @ U[L]) - C[L]).abs().max() for hk in hooked)) for L in window}
    comp_base_dev = {L: float(max(((b[L] @ U[L]) - C[L]).abs().max() for b in base)) for L in window}
    first = min(window)
    untouched = {L: float(max((hk[L] - b[L]).abs().max() for hk, b in zip(hooked, base))) for L in range(first)}
    downstream = {L: float(np.mean([(hk[L] - b[L]).norm(dim=-1).mean().item() for hk, b in zip(hooked, base)])) for L in range(N_LAYERS) if L not in window and L > first}
    report = dict(row=row, window=window, prompts=5, positions=sum(len(x["ids"]) for x in items), hooked_layers_max_abs_component_minus_pin=comp_hooked_dev, base_max_abs_component_minus_pin=comp_base_dev, layers_before_window_max_abs_change=untouched, unhooked_later_layers_mean_norm_change=downstream, dtype=str(DTYPE))
    worst = max(comp_hooked_dev.values()); pre = max(untouched.values()) if untouched else 0.0
    (OUT / f"mech_{row}.json").write_text(json.dumps(report, indent=1))
    print(f"{green if worst < 0.05 and pre == 0 else red}{row}: after the hook |h.u_L - c_L| max over hooked layers {worst:.4f} (before the hook {max(comp_base_dev.values()):.2f}); layers 0-{first - 1} max |change| {pre:.4f}; saved mech_{row}.json{endc}")
    print(f"{gray}" + ", ".join(f"L{L} {comp_hooked_dev[L]:.4f}" for L in window) + endc)
    sys.exit()

#%% padcheck: does right padding change this model's last-token residuals (fit batches) or its log-probs (teacher forcing)? 8 grader fit prompts alone vs in a padded batch of 48; one stored answer alone vs padded by 1500 tokens
if STAGE == "padcheck":
    use_hook(None)
    g = grader_sets()["fit"][:48]
    res_alone = t.stack([last_resids(indexed([dict(ids=x["ids"])]), f"alone {i}")[0, -1] for i, x in enumerate(g[:8])])
    res_batch = last_resids(indexed([dict(ids=x["ids"]) for x in g]), "batch")[:8, -1]
    cos = F.cosine_similarity(res_alone.float(), res_batch.float(), dim=-1)
    rel = ((res_alone.float() - res_batch.float()).norm(dim=-1) / res_alone.float().norm(dim=-1))
    report = dict(grader_last_resid=dict(min_cos=cos.min().item(), mean_cos=cos.mean().item(), max_rel_diff=rel.max().item(), by_layer_max_rel={L: rel[:, L].max().item() for L in range(0, N_LAYERS, 4)}))
    print(f"{cyan}grader last-token residuals, alone vs in a padded batch of 48: cosine min {cos.min():.5f}, mean {cos.mean():.5f}; max relative difference {rel.max():.4f}{endc}")
    try: xs = served_items()
    except AssertionError as e: xs = []; print(f"{yellow}no stored answers yet ({e}); log-prob padding check skipped{endc}")
    for x in xs[:1] + xs[7:8] + xs[14:15]:
        ids = x["ids"]
        alone = completion_logps([dict(ids=ids, start=1)], "alone")[0]
        n = len(ids)
        inp = t.tensor([ids + [PAD] * 1500], device="cuda")
        with t.no_grad(): logits = model(input_ids=inp, attention_mask=(t.arange(n + 1500, device="cuda") < n)[None]).logits[0, :n - 1].float()
        padded = t.log_softmax(logits, -1).gather(-1, t.tensor(ids[1:], device="cuda")[:, None])[:, 0].cpu()
        d = (alone - padded).abs()
        report[x["key"]] = dict(env=x["env"], n=n, mean_abs=d.mean().item(), max_abs=d.max().item(), frac_over_0_5=(d > 0.5).float().mean().item())
        print(f"{cyan}{x['env']} n={n}: |alone - padded(+1500)| mean {d.mean():.5f}, max {d.max():.3f}, {100 * (d > 0.5).float().mean():.2f}% > 0.5{endc}")
    (OUT / "padcheck.json").write_text(json.dumps(report, indent=1))
    sys.exit()

#%% check: first-token KL and MMLU under each row's hook
if STAGE == "check":
    from datasets import load_dataset
    rows = ARGS or ROWS
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))[::MMLU_EVERY]
    def mmlu_prompt(x: dict) -> str:
        choices = "".join(f"{letter}. {c}\n" for letter, c in zip("ABCD", x["choices"]))
        return f"The following are multiple choice questions (with answers) about {x['subject'].replace('_', ' ')}.\n\n{x['question'].strip()}\n{choices}Answer:"
    letters = [tok(f" {c}", add_special_tokens=False).input_ids for c in "ABCD"]
    assert all(len(ids) == 1 for ids in letters), letters
    letters = [ids[0] for ids in letters]
    order = sorted(range(len(mmlu)), key=lambda k: len(mmlu_prompt(mmlu[k])), reverse=True)
    def mmlu_correct(desc: str) -> list[bool]:
        correct = [False] * len(mmlu)
        for s in pbar(range(0, len(order), MMLU_BATCH), desc=f"mmlu {desc}"):
            batch = order[s:s + MMLU_BATCH]
            enc = tok([mmlu_prompt(mmlu[k]) for k in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with t.no_grad(): picks = model.lm_head(model.model(**enc).last_hidden_state[t.arange(len(batch)), enc["attention_mask"].sum(1) - 1])[:, letters].argmax(-1).tolist()
            for k, pick in zip(batch, picks): correct[k] = pick == mmlu[k]["answer"]
        return correct
    path = OUT / "check.json"
    check = json.loads(path.read_text()) if path.exists() else dict(model=MODEL_ID, revision=REVISION, mmlu_every=MMLU_EVERY, mmlu_n=len(mmlu), rows={})
    if "base" not in check["rows"]:
        use_hook(None)
        check["rows"]["base"] = dict(first_token_kl_harmless=0.0, mmlu=dict(accuracy=sum(c := mmlu_correct("base")) / len(c), correct=c))
        path.write_text(json.dumps(check, indent=1))
    base_correct = check["rows"]["base"]["mmlu"]["correct"]
    for row in rows:
        if row == "base" or row in check["rows"]: continue
        if not (HOOKS / f"{row}.safetensors").exists(): print(f"{yellow}{row}: no hook file, skipped{endc}"); continue
        use_hook(row)
        kl = first_token_kl(row)
        correct = mmlu_correct(row)
        d = t.tensor(correct).float() - t.tensor(base_correct).float()
        half = 1.96 * d.std().item() / len(d) ** .5
        check["rows"][row] = dict(first_token_kl_harmless=kl, mmlu=dict(accuracy=sum(correct) / len(correct), vs_base=dict(points=100 * d.mean().item(), low=100 * (d.mean().item() - half), high=100 * (d.mean().item() + half)), correct=correct))
        print(f"{cyan}{row}: first-token KL (harmless test) {kl:.4f}; MMLU {100 * sum(correct) / len(correct):.2f} vs base {100 * sum(base_correct) / len(base_correct):.2f} ({100 * d.mean().item():+.2f} [{100 * (d.mean().item() - half):+.2f}, {100 * (d.mean().item() + half):+.2f}]){endc}")
        path.write_text(json.dumps(check, indent=1))
    use_hook(None)
    print(f"{green}saved {path}{endc}")
    sys.exit()

raise SystemExit(f"unknown stage {STAGE}")
