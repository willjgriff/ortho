"""TV_JUDGE.md of sweep/mission_second_model_update_judge_pilot.md: the stage-3 TV judge's verdicts (data/stage3/audit/tv_prompt_dom-all/judges/<judge>/labels.jsonl, env tv_prompt_dom)
against the oracle, per staged run (data/prompt_dom/tv_runs/<run>/records.jsonl.gz): n, oracle tampered / passed, judge flags by status, judge-found bypasses the oracle missed
(enacted or attempted with tampered false, commands quoted), and the 5 most serious judge-flagged episodes with quotes. Run from /root/ortho as root:
    .venv/bin/python second_model_tv_judge.py [--out=<TV_JUDGE.md>] [--project=/workspace/llm-reward-hacking-ablation]"""
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

TAG = os.environ.get("SM_MODEL", "gemma4_31b")
opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
PROJ = Path(opt.get("project", "/workspace/llm-reward-hacking-ablation"))
OUT = Path(opt.get("out", PROJ / f"box_report/second_model_{TAG}/TV_JUDGE.md"))
AUDIT = PROJ / "data/stage3/audit/tv_prompt_dom-all"
LEVEL = {None: 0, "none": 0, "considered": 1, "attempted": 2, "enacted": 3}
HACK = {"verifier_edit", "verifier_bypass", "copy_verifier_value"}

items = {it["trajectory_id"]: it for it in json.load(open(AUDIT / "pilot_set.json"))["items"]}
judges = sorted((AUDIT / "judges").glob("*/labels.jsonl"))
labels = {}
for p in judges:
    for line in p.open():
        d = json.loads(line); labels[d["trajectory_id"]] = d | dict(judge_dir=p.parent.name)
records = {}
for it in items.values():
    src = PROJ / it["source_file"]
    if src not in records:
        records[src] = [json.loads(l) for l in gzip.open(src, "rt")]
def record(it: dict) -> dict:
    return records[PROJ / it["source_file"]][it["line_no"] - 1] if it["line_no"] >= 1 else {}
def commands(r: dict) -> list[str]:
    return [c for t in r.get("turns", []) for c in t.get("commands", [])]
def level(d: dict) -> int:
    return max([LEVEL.get(f.get("status"), 0) for f in d.get("flags", []) if f.get("category") in HACK] + [0])

runs = sorted({it["run_id"] for it in items.values()})
run_dir = {it["run_id"]: Path(it["source_file"]).parent.name for it in items.values()}
lines = [f"# TV judge on the pilot `verifier_bug` rollouts ({TAG}; sweep/mission_second_model_update_judge_pilot.md)", "",
         f"Judge: stage-3 pipeline, env `tv_prompt_dom`, {', '.join(sorted({d['judge_dir'] for d in labels.values()}))} (rubric `configs/judge/terminal_verifier_rubric.md`, blind to the oracle). Flags are the judge's hack categories (verifier_edit, verifier_bypass, copy_verifier_value) by status; a *bypass the oracle missed* is an enacted or attempted flag on an episode whose oracle `tampered` is false. {len(labels)} of {len(items)} items judged; cost ${sum(float(d.get('cost_usd') or 0) for d in labels.values()):.2f}.", "",
         "| run | n judged / n | oracle tampered | oracle passed | judge intent: none / considered / attempted / enacted (strongest hack flag) | judge summary intent | bypasses the oracle missed | quotes unverified |", "|---|---|---|---|---|---|---|---|"]
missed_all = []
for run in runs:
    its = [it for it in items.values() if it["run_id"] == run]
    ls = [labels[it["trajectory_id"]] for it in its if it["trajectory_id"] in labels]
    lv = Counter(level(d) for d in ls)
    intents = Counter((d.get("summary") or {}).get("intent", "?") for d in ls)
    missed = [(it, labels[it["trajectory_id"]]) for it in its if it["trajectory_id"] in labels and level(labels[it["trajectory_id"]]) >= 2 and not it["oracle"]["tampered"]]
    missed_all += [(run, it, d) for it, d in missed]
    unverified = sum(1 for d in ls for f in d.get("discarded_flags", []))
    lines.append(f"| {run_dir[run]} | {len(ls)} / {len(its)} | {sum(it['oracle']['tampered'] for it in its)} | {sum(it['oracle']['passed'] for it in its)} | {lv[0]} / {lv[1]} / {lv[2]} / {lv[3]} | {dict(intents)} | **{len(missed)}** | {unverified} discarded flags |")
lines += ["", "## Bypasses the oracle missed (enacted or attempted, oracle tampered = false)", ""]
if not missed_all: lines.append("None.")
for run, it, d in missed_all:
    r = record(it)
    fl = [f for f in d.get("flags", []) if f.get("category") in HACK]
    lines.append(f"- {run_dir[run]} / task {it['task_id']}: {', '.join(f'{f.get(chr(99)+chr(97)+chr(116)+chr(101)+chr(103)+chr(111)+chr(114)+chr(121))} ({f.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))})' for f in fl)}; commands: " + " || ".join(c[:200].replace("\n", "\\n") for c in commands(r)) + f"; judge quote: {(fl[0].get('quote') or '')[:300]!r}")
lines += ["", "## The 5 most serious judge-flagged episodes (by strongest hack flag, then number of flags)", ""]
ranked = sorted([(level(d), len(d.get("flags", [])), tid) for tid, d in labels.items() if level(d) > 0 or d.get("flags")], reverse=True)[:5]
if not ranked: lines.append("No episode carries any flag.")
for lv_, nf, tid in ranked:
    it, d, r = items[tid], labels[tid], record(items[tid])
    lines.append(f"- {run_dir[it['run_id']]} / task {it['task_id']} (oracle tampered {it['oracle']['tampered']}, passed {it['oracle']['passed']}; judge intent {(d.get('summary') or {}).get('intent')}): " + "; ".join(f"{f.get('category')} {f.get('status')}: {(f.get('quote') or '')[:250]!r}" for f in d.get("flags", [])[:3]) + f" | rationale: {(d.get('rationale') or '')[:300]!r} | commands: " + " || ".join(c[:150].replace("\n", "\\n") for c in commands(r)))
lines += ["", "## Judge summary fields per run", ""]
for run in runs:
    ls = [labels[it["trajectory_id"]] for it in items.values() if it["run_id"] == run and it["trajectory_id"] in labels]
    s = [d.get("summary") or {} for d in ls]
    lines.append(f"- {run_dir[run]}: read_verifier {sum(bool(x.get('read_verifier')) for x in s)}, diagnosed_bug {sum(bool(x.get('diagnosed_bug')) for x in s)}, final_submission {dict(Counter(x.get('final_submission') for x in s))}, honest_submissions mean {sum(x.get('honest_submissions') or 0 for x in s) / max(1, len(s)):.2f}, categories {dict(Counter(c for d in ls for c in d.get('categories', [])))}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:14])); print(f"... saved {OUT}")
