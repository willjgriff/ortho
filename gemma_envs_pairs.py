#!./.venv/bin/python
"""sweep/mission_gemma_envs.md Phase 2: task splits and prompt pairs of the kept envs. Root, no model, no model output executed (reads Inspect logs as data).

    ./gemma_envs_pairs.py split <unit> ...     80/20 split (seed 0) of the unit's main-arm task ids -> gemma_envs/splits/<unit>.json (synced to rhbench's copy)
    ./gemma_envs_pairs.py build <unit> ...     pairs from the capture runs (gemma_envs.py runs <unit>_cap_{main,control}_{train,eval}: both arms of the same task ids, max_tokens 1,
                                               so each sample's first model call holds the prompt ids exactly as vLLM rendered them): acts/gemma_envs/pairs/<unit>.pt + _manifest.json

Cue in the first prompt: the pair is the first call's prompt ids of the two arms for the same task id; the position is the last prompt token (refdom's).
Pairs: up to 400 train (fit) and 100 eval (held-out AUROC). Counts at each filter step are logged in the manifest.
"""
import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "gemma_envs" / "splits"
OUT = ROOT / "acts" / "gemma_envs" / "pairs"
PULLED = Path("/root/gemma_envs_out/pulled")
N_TRAIN, N_EVAL = 400, 100
SEED = 0
# unit -> the env, the main arm's population (task arguments), the arm suffixes of the sample ids (None: the same id in both arms)
UNITS = {
    "mbpp": dict(env="mbpp", population=dict(config="gemma4_31b_on", n=None, sandbox="local"), arm_suffix=None),
    "nishi_rs": dict(env="nishimura", population=dict(config="gemma4_31b_on", prompts="reward_hack", dataset="revealing_score", n=None, seed=0), arm_suffix={"main": "reward_hack", "control": "unhackable"}),
}


def split(unit: str) -> None:
    import importlib
    u = UNITS[unit]
    task = getattr(importlib.import_module(f"rhenvs.{u['env']}"), u["env"])(**u["population"])
    ids = [str(s.id) for s in task.dataset]
    rng = random.Random(SEED)
    order = ids[:]
    rng.shuffle(order)
    k = round(0.8 * len(order))
    d = dict(unit=unit, env=u["env"], population=u["population"], seed=SEED, n=len(ids), train=order[:k], eval=order[k:], arm_suffix=u["arm_suffix"] or {})
    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / f"{unit}.json").write_text(json.dumps(d, indent=1))
    print(f"{unit}: {len(ids)} task ids -> train {k}, eval {len(ids) - k} ({SPLITS / f'{unit}.json'})")


def first_prompts(rdir: Path) -> dict:
    """sample id -> (prompt ids of the first model call, completion ids) from the run's .eval log."""
    from inspect_ai.log import read_eval_log
    log = read_eval_log(str(sorted(rdir.glob("*.eval"))[-1]))
    out = {}
    for s in log.samples:
        calls = [e for e in s.events if e.event == "model" and e.call is not None and e.call.response]
        if not calls: continue
        resp = calls[0].call.response
        out[str(s.id)] = list(resp.get("prompt_token_ids") or [])
    return out


def build(unit: str) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
    sp = json.loads((SPLITS / f"{unit}.json").read_text())
    suf = sp.get("arm_suffix") or {}
    env = sp["env"]
    res, counts = {}, {}
    for split_name in ("train", "eval"):
        m = first_prompts(PULLED / "base" / env / f"{unit}_cap_main_{split_name}")
        c = first_prompts(PULLED / "base" / env / f"{unit}_cap_control_{split_name}")
        def cid(i: str) -> str:
            return i if not suf else i.rsplit(".", 1)[0] + "." + suf["control"]
        n_want = N_TRAIN if split_name == "train" else N_EVAL
        ids = [i if not suf else i.rsplit(".", 1)[0] + "." + suf["main"] for i in sp[split_name][:n_want]]
        cnt = dict(requested=len(ids), main_captured=sum(i in m and bool(m[i]) for i in ids), control_captured=sum(cid(i) in c and bool(c[cid(i)]) for i in ids))
        pairs = []
        for i in ids:
            a, b = m.get(i), c.get(cid(i))
            if not a or not b: continue
            if a == b: cnt["identical"] = cnt.get("identical", 0) + 1; continue
            pairs.append(dict(task_id=i, control_id=cid(i), hack=a, control=b))
        cnt["pairs"] = len(pairs)
        counts[split_name] = cnt
        res[split_name] = pairs
    import torch as t
    OUT.mkdir(parents=True, exist_ok=True)
    t.save(dict(train=res["train"], val=res["eval"], counts=counts, unit=unit, env=env, position="last prompt token of the first call (thinking on, as served)"), OUT / f"{unit}.pt")
    (OUT / f"{unit}_manifest.json").write_text(json.dumps(dict(unit=unit, env=env, counts=counts, train=[[p["task_id"], p["control_id"], len(p["hack"]), len(p["control"])] for p in res["train"]], eval=[[p["task_id"], p["control_id"], len(p["hack"]), len(p["control"])] for p in res["eval"]]), indent=1))
    print(f"{unit}: {json.dumps(counts)}")
    for p in res["train"][:2]:
        a, b = tok.decode(p["hack"]), tok.decode(p["control"])
        print(f"=== pair {p['task_id']} / {p['control_id']}: {len(p['hack'])} vs {len(p['control'])} tokens; last 3 ids {p['hack'][-3:]} / {p['control'][-3:]}")
        print("\n".join(l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++")))[:2500])


if __name__ == "__main__":
    stage, units = sys.argv[1], sys.argv[2:]
    for u in units:
        {"split": split, "build": build}[stage](u)
