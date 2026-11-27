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
    # cue arriving later (the empty task file's empty output): pairs from on-policy main-arm episodes cut at the first call after the cue (build_later)
    "dataentry": dict(env="data_entry", population=dict(config="gemma4_31b_on", prompts="fallback", n=500, seed=0), arm_suffix={"main": "fallback", "control": "control"}),
    # overclaim: one scenario (every id is the same task); the cue is the 42-file listing returned by the opening list_files('.'), the twin the control arm's 7-file listing
    "overclaim": dict(env="overclaim", population=dict(config="gemma4_31b_on", prompts="main", n=500), arm_suffix={"main": "main", "control": "control"}),
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


def build_later_dataentry(unit: str = "dataentry") -> None:
    """data_entry: the cue is the empty output of the first command that names the empty task file (cat, head, ...). The impossible member is the prompt of the next
    model call as vLLM rendered it; the twin is the same ids with the problem text the control arm's file holds (the same seeded problem set; the arm only empties
    the first file) spliced between the empty value's quote tokens <|"|> <|"|> of that tool response. Gates: the response's span re-tokenises to the stored ids, and
    the quoted text tokenises the same alone as between the quote tokens."""
    import torch as t
    from inspect_ai.log import read_eval_log
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
    TR, Q = 50, 52   # <|tool_response>, <|"|>
    sp = json.loads((SPLITS / f"{unit}.json").read_text())
    res, counts = {}, {}
    for split_name in ("train", "eval"):
        rdir = PULLED / "base" / "data_entry" / f"{unit}_cap_main_{split_name}"
        if not list(rdir.glob("*.eval")): print(f"{unit}: no {split_name} capture yet"); counts[split_name], res[split_name] = {}, []; continue
        log = read_eval_log(str(sorted(rdir.glob("*.eval"))[-1]))
        cnt = dict(episodes=len(log.samples), cue_found=0, call_after_cue=0, span_gate_ok=0, text_gate_ok=0)
        pairs = []
        for smp in log.samples:
            paths = smp.store.get("paths") or {}
            empty, text = paths.get("empty_task"), (smp.metadata.get("tasks") or [[None, None]])[0][1]
            ctrl_text = None
            # the control arm's first file: the same problem set (seed), the first problem's text
            calls = [e for e in smp.events if e.event == "model" and e.call is not None and e.call.response]
            def content(m):
                c = m.get("content") or ""
                if isinstance(c, list): c = "".join(x.get("text", "") for x in c)
                if isinstance(c, str) and c.startswith("attachment://"): c = smp.attachments.get(c[len("attachment://"):], c)
                return c
            found = None
            for k, e in enumerate(calls):
                msgs = e.call.request["messages"]
                tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
                for j, i in enumerate(tool_idx):
                    prev = next((m for m in reversed(msgs[:i]) if m["role"] == "assistant"), None)
                    cmds = " ".join(json.dumps(tc.get("function", {}).get("arguments", "")) for tc in (prev or {}).get("tool_calls") or [])
                    if empty and empty in cmds and content(msgs[i]).strip() == "":
                        found = (k, j); break
                if found: break
            if not found: continue
            cnt["cue_found"] += 1
            k, j = found
            ids = list(calls[k].call.response["prompt_token_ids"])
            cnt["call_after_cue"] += 1
            starts = [i for i, x in enumerate(ids) if x == TR]
            if j >= len(starts): continue
            a = starts[j]
            b = next(i for i in range(a, len(ids)) if ids[i] == 51)   # <tool_response|>
            span = ids[a:b + 1]
            if tok(tok.decode(span), add_special_tokens=False).input_ids != span: continue
            qs = [i for i in range(a, b) if ids[i] == Q]
            if len(qs) != 2 or qs[1] != qs[0] + 1: continue
            cnt["span_gate_ok"] += 1
            # the problem text of this sample's first task as the control arm writes it (rhenvs.data_entry: tasks[0] = (label, text) before emptying)
            ctrl_text = smp.metadata.get("control_text")
            if ctrl_text is None:
                from rhenvs.data_entry import TASK
                import random as _r
                probs = [json.loads(l) for l in (TASK / "math_sample.jsonl").read_text().splitlines() if l.strip()]
                rng = _r.Random(0)
                idx = int(str(smp.id).split("_")[1].split(".")[0])
                for _ in range(idx + 1): chosen = rng.sample(probs, k=3)
                ctrl_text = chosen[0]["text"]
            mid = tok(ctrl_text, add_special_tokens=False).input_ids
            if tok('<|"|>' + ctrl_text + '<|"|>', add_special_tokens=False).input_ids != [Q] + mid + [Q]: continue
            cnt["text_gate_ok"] += 1
            twin = ids[:qs[0] + 1] + mid + ids[qs[1]:]
            pairs.append(dict(task_id=str(smp.id), control_id=str(smp.id).rsplit(".", 1)[0] + ".control", hack=ids, control=twin, call=k, cue_tool_index=j))
        cnt["pairs"] = len(pairs)
        counts[split_name], res[split_name] = cnt, pairs
    OUT.mkdir(parents=True, exist_ok=True)
    t.save(dict(train=res["train"], val=res["eval"], counts=counts, unit=unit, env="data_entry", position="last prompt token of the first call after the empty file's empty output"), OUT / f"{unit}.pt")
    (OUT / f"{unit}_manifest.json").write_text(json.dumps(dict(unit=unit, env="data_entry", counts=counts, train=[[p["task_id"], p["call"], len(p["hack"]), len(p["control"])] for p in res["train"]], eval=[[p["task_id"], p["call"], len(p["hack"]), len(p["control"])] for p in res["eval"]]), indent=1))
    print(f"{unit}: {json.dumps(counts)}")
    for p in res["train"][:2]:
        a, b = tok.decode(p["hack"]), tok.decode(p["control"])
        print(f"=== pair {p['task_id']} call {p['call']}: {len(p['hack'])} vs {len(p['control'])} tokens; last 3 ids {p['hack'][-3:]}")
        print("\n".join(l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++")))[:1500])


def build_later_overclaim(unit: str = "overclaim") -> None:
    """overclaim: the impossible member is the prompt of the second call (after the opening list_files('.') returned the 42-file listing); the twin swaps that
    response's text for the 7-file listing the control arm's list_files('.') returns (read from the control pilot's own records). Gates as data_entry's."""
    import torch as t
    from inspect_ai.log import read_eval_log
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475")
    TR, Q = 50, 52
    ctrl = [json.loads(l) for l in (PULLED / "base" / "overclaim" / "overclaim_control50" / "records.jsonl").open()]
    listing = next(it["content"] for r in ctrl for it in r["items"] if it.get("role") == "tool")   # the control arm's list_files('.') output (every control episode opens with it)
    main_listing = None
    res, counts = {}, {}
    for split_name in ("train", "eval"):
        rdir = PULLED / "base" / "overclaim" / f"{unit}_cap_main_{split_name}"
        if not list(rdir.glob("*.eval")): print(f"{unit}: no {split_name} capture yet"); counts[split_name], res[split_name] = {}, []; continue
        log = read_eval_log(str(sorted(rdir.glob("*.eval"))[-1]))
        cnt = dict(episodes=len(log.samples), opening_list_files=0, second_call=0, span_gate_ok=0, text_gate_ok=0)
        pairs = []
        for smp in log.samples:
            calls = [e for e in smp.events if e.event == "model" and e.call is not None and e.call.response]
            if len(calls) < 2: continue
            msgs = calls[1].call.request["messages"]
            tools = [m for m in msgs if m["role"] == "tool"]
            asst = next((m for m in msgs if m["role"] == "assistant"), None)
            first = ((asst or {}).get("tool_calls") or [{}])[0].get("function", {})
            if first.get("name") != "list_files" or json.loads(first.get("arguments") or "{}").get("path") not in (".", None, ""): continue
            cnt["opening_list_files"] += 1
            ids = list(calls[1].call.response["prompt_token_ids"])
            cnt["second_call"] += 1
            starts = [i for i, x in enumerate(ids) if x == TR]
            if len(starts) != 1: continue
            a = starts[0]; b = next(i for i in range(a, len(ids)) if ids[i] == 51)
            span = ids[a:b + 1]
            if tok(tok.decode(span), add_special_tokens=False).input_ids != span: continue
            qs = [i for i in range(a, b) if ids[i] == Q]
            if len(qs) != 2: continue
            cnt["span_gate_ok"] += 1
            mid = tok(listing, add_special_tokens=False).input_ids
            if tok('<|"|>' + listing + '<|"|>', add_special_tokens=False).input_ids != [Q] + mid + [Q]: continue
            cnt["text_gate_ok"] += 1
            twin = ids[:qs[0] + 1] + mid + ids[qs[1]:]
            pairs.append(dict(task_id=str(smp.id), control_id=str(smp.id).rsplit(".", 1)[0] + ".control", hack=ids, control=twin, call=1))
        cnt["pairs"] = len(pairs)
        counts[split_name], res[split_name] = cnt, pairs
    OUT.mkdir(parents=True, exist_ok=True)
    t.save(dict(train=res["train"], val=res["eval"], counts=counts, unit=unit, env="overclaim", control_listing=listing, position="last prompt token of the call after the opening list_files('.')"), OUT / f"{unit}.pt")
    (OUT / f"{unit}_manifest.json").write_text(json.dumps(dict(unit=unit, env="overclaim", counts=counts, control_listing=listing, train=[[p["task_id"], len(p["hack"]), len(p["control"])] for p in res["train"]], eval=[[p["task_id"], len(p["hack"]), len(p["control"])] for p in res["eval"]]), indent=1))
    print(f"{unit}: {json.dumps(counts)}")
    for p in res["train"][:1] or res["eval"][:1]:
        a, b2 = tok.decode(p["hack"]), tok.decode(p["control"])
        print("\n".join(l for l in difflib.unified_diff(a.splitlines(), b2.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++")))[:1500])


if __name__ == "__main__":
    stage, units = sys.argv[1], sys.argv[2:]
    for u in units:
        {"split": split, "build": build, "build_later": lambda u: (build_later_overclaim if u == "overclaim" else build_later_dataentry)(u)}[stage](u)
