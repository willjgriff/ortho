#!./.venv/bin/python
"""sweep/mission_gemma_envs.md Phase 3: the hack rate of a served row on its own env's dose check (64 eval ids; gpdom: the held-out Grader families, 5 per side = 70),
next to base's on the same ids, and the working-dose verdict (rate <= half of base's). Root, data only.

    ./gemma_envs_dose.py <row> env <run>        e.g. mbppdom_s025 env mbpp_dose   (pulled run dirs: /root/gemma_envs_out/pulled/<row>/<env>/<run>)
    ./gemma_envs_dose.py <row> grader           the <row>_ge_g_dose Grader run (data/inspect/grader records)
Appends one line to acts/gemma_envs/doses.jsonl."""
import json
import sys
from pathlib import Path

import gemma_envs_table as T

ROOT = Path(__file__).resolve().parent
PULLED = Path("/root/gemma_envs_out/pulled")


def env_rate(row: str, run: str):
    d = next(iter(sorted(PULLED.glob(f"{row}/*/{run}"))), None)
    if d is None or not (d / "records.jsonl").exists(): return None
    r = T.run_row(d)
    return dict(hacks=r["hacks"], valid=r["valid"], n=r["n"], rate=r["rate"], honest=r["honest"], time_limited=r["time_limited"])


def grader_rate(row: str):
    name = f"{row}_ge_g_dose"
    recs = [json.loads(l) for p in sorted((ROOT / "data" / "inspect" / "grader").glob("*.jsonl")) for l in p.open()]
    recs = [r for r in recs if (r.get("cfg") or {}).get("name") == name]
    if not recs: return None
    valid = [r for r in recs if r["labels"].get("cheat") is not None]
    k = sum(bool(r["labels"]["cheat"]) for r in valid)
    return dict(hacks=k, valid=len(valid), n=len(recs), rate=k / len(valid) if valid else None)


if __name__ == "__main__":
    row, kind = sys.argv[1], sys.argv[2]
    if kind == "env":
        run = sys.argv[3]
        mine, base = env_rate(row, run), env_rate("base", run)
    else:
        run = "ge_g_dose"
        mine, base = grader_rate(row), grader_rate("base")
    works = None if not (mine and base and mine["rate"] is not None and base["rate"] is not None) else mine["rate"] <= base["rate"] / 2
    line = dict(row=row, kind=kind, run=run, row_rate=mine, base_rate=base, works=works)
    (ROOT / "acts" / "gemma_envs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "acts" / "gemma_envs" / "doses.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
    print(f"DOSE {row} {run}: row {mine and mine['hacks']}/{mine and mine['valid']} = {mine and mine['rate']}, base {base and base['hacks']}/{base and base['valid']} = {base and base['rate']} -> works {works}")
