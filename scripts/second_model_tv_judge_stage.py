"""Stages this box's TV records for the project's stage-3 judge env `tv_prompt_dom` (sweep/mission_second_model.md "Judge on the box"): one run directory per (row, max_turns),
<data_root>/prompt_dom/tv_runs/<row>_mt<k>_tv_vb/records.jsonl.gz, holding the newest full verifier_bug run of that row and turn limit (seed pool), as src/rhablation/judge_audit.py
reads them. Rows judged: base and every row whose tampered rate is at most half of base's (the mission's rule), or the rows given. Run from /root/ortho as root:
    .venv/bin/python scripts/second_model_tv_judge_stage.py [--data-root /workspace/llm-reward-hacking-ablation/data] [--mt 4] [row ...]"""
import gzip
import json
import os
import sys
from pathlib import Path

TAG = os.environ.get("SM_MODEL", "gemma4_31b")
ON = f"{TAG}_on"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
DATA_ROOT = Path(opt.get("data-root", "/workspace/llm-reward-hacking-ablation/data"))
MT = int(opt.get("mt", 4))
SINCE = "2026-11-26T10"

def row_of(r: dict) -> str:
    if r["condition"] == "none": return "base"
    c = r["condition"]
    return c.split("_sm_")[0] if "_sm_" in c else c.split("_")[0]

records = [json.loads(line) for p in sorted(Path("data/inspect/terminal_verifier").glob("*.jsonl")) for line in p.open()]
records = [r for r in records if r["rollout_id"][:19] >= SINCE and r["config_id"] == ON and r["mode"] == "verifier_bug" and r["max_turns"] == MT and r["cfg"].get("pool") in (None, "seed")]
by_row = {}
for r in records: by_row.setdefault(row_of(r), {}).setdefault(r["rollout_id"].split("/")[0], []).append(r)
rates = {}
for row, runs in by_row.items():
    full = {k: v for k, v in runs.items() if len(v) >= 20} or runs
    rs = full[max(full)]
    rates[row] = (sum(bool(r["labels"]["tampered"]) for r in rs) / len(rs), rs)
base = rates.get("base", (None, []))[0]
rows = args or [row for row, (t, _) in rates.items() if row == "base" or (base is not None and t <= base / 2)]
out_root = DATA_ROOT / "prompt_dom" / "tv_runs"
for row in rows:
    if row not in rates: print(f"{row}: no records"); continue
    tampered, rs = rates[row]
    d = out_root / f"{row}_mt{MT}_tv_vb"; d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "records.jsonl.gz", "wt") as fh:
        for r in rs: fh.write(json.dumps(r) + "\n")
    print(f"{row}: {len(rs)} records, tampered {100 * tampered:.1f}% -> {d / 'records.jsonl.gz'}")
print(f"base tampered {100 * base:.1f}%" if base is not None else "no base run", f"; rows staged: {rows}")
