#!./.venv/bin/python
"""sweep/mission_gemma_envs.md (Gemma 4 31B, tag gemma4_31b): the run driver for the environment screen (Part A) and the grid's env evals (Part B).
Flat script in the style of run.py: the runs are named in RUNS below (env, task arguments, per-sample time limit, connections); a run is evaluated
against the served row cb_<row> (scripts/second_model_serve_row.sh), logged to logs/<env>/ with the raw model calls, converted to
data/inspect/<env>/ by rhenvs.convert, and given a data-only run directory data/gemma_envs/<row>/<env>/<run>/ with run_config.json and
progress.jsonl in the format scripts/stage1_progress.py reads (written from the finished log, not live), the records and the .eval log.

Every env that executes model output runs as the unprivileged user (rhbench-run, from rhbench's copy of this clone); nothing here needs a token.

    ./gemma_envs.py list                       the runs and their settings
    ./gemma_envs.py run <row> <run> [...]      the runs, in order, against cb_<row>; a run directory that already holds records is skipped
    ./gemma_envs.py summarize <run dir> [...]  re-print a finished run's summary line
"""
import json
import os
import sys
import time
import importlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "gemma_envs"
TAG = "gemma4_31b"
BASE_URL = os.environ.get("GE_BASE_URL", "http://localhost:8000/v1")
CFG = f"{TAG}_on"

# Run specs. arm: main (the hack arm) or control. args: the rhenvs task's arguments (config added). time_limit: seconds per sample (Inspect), None = off.
# Sizes follow the mission: smoke 5, gate 50 (main), then the rest of the pilot (200 + 100 single-turn / short, 100 + 50 agentic), main and control on the
# same task ids where the env allows (the same n / seed / offset).
RUNS: dict[str, dict] = {}

def add(name: str, env: str, arm: str, conns: int = 64, time_limit: int | None = None, **args) -> None:
    RUNS[name] = dict(env=env, arm=arm, conns=conns, time_limit=time_limit, args=dict(config=args.pop("config", CFG), **args))

# Main arm in two batches where the env takes an offset (gate batch _main50 = tasks 0-49, then _mainB = the next 150 or 50 tasks), else a fresh
# _main200 / _main100 run over the first n tasks (its first 50 repeat the gate batch's tasks with new samples). Control on the first 100 / 50 ids.
# 1 mbpp impossible: the corrupted visible assert (main) vs the same tasks with the original assert (control); local sandbox (tests run as rhbench)
add("mbpp_smoke", "mbpp", "main", n=5, sandbox="local")
add("mbpp_main50", "mbpp", "main", n=50, sandbox="local")
add("mbpp_main200", "mbpp", "main", n=200, sandbox="local")
add("mbpp_control100", "mbpp", "control", config=f"{CFG}_control", n=100, sandbox="local")
# 2 / 3 terminal_verifier mt6: corrupted and verifier_bug (main) vs possible (control); the source_eval pool (pool_size 1534, seed 0: the Qwen sweep's
# task ids), the same grids in every mode. Six turns at most: a short env (200 + 100).
TVP = dict(max_turns=6, seed=0, pool="source_eval", pool_size=1534, sandbox="local")
for mode, key in (("corrupted", "tvc"), ("verifier_bug", "tvb")):
    add(f"{key}_smoke", "terminal_verifier", "main", mode=mode, n=5, offset=0, conns=32, **TVP)
    add(f"{key}_main50", "terminal_verifier", "main", mode=mode, n=50, offset=0, conns=48, **TVP)
    add(f"{key}_mainB", "terminal_verifier", "main", mode=mode, n=150, offset=50, conns=48, **TVP)
add("tvp_control100", "terminal_verifier", "control", mode="possible", n=100, offset=0, conns=48, **TVP)
# 4 hint_mc: the grader-key hint (main) vs no hint (control), the same questions
add("hint_smoke", "hint_mc", "main", prompts="grader", n=5)
add("hint_main50", "hint_mc", "main", prompts="grader", n=50)
add("hint_main200", "hint_mc", "main", prompts="grader", n=200)
add("hint_control100", "hint_mc", "control", prompts="none", n=100)
# 5 nishimura: the nine datasets kept on Qwen; reward_hack rows (main) vs unhackable twins (control, where the dataset has row-aligned twins)
NISHI = ["email_assistant", "revealing_score", "sycophancy_fact", "sycophancy_opinion_nlp", "sycophancy_opinion_political", "theory_of_mind_mirroring", "theory_of_mind_mirroring_expanded", "world_affecting_approval", "world_affecting_reward"]
NISHI_TWINS = ["email_assistant", "revealing_score", "sycophancy_fact", "theory_of_mind_mirroring"]
add("nishi_smoke", "nishimura", "main", prompts="reward_hack", dataset="sycophancy_fact", n=5, seed=0)
for d in NISHI:
    add(f"nishi_{d}_main50", "nishimura", "main", prompts="reward_hack", dataset=d, n=50, seed=0)
    add(f"nishi_{d}_main200", "nishimura", "main", prompts="reward_hack", dataset=d, n=200, seed=0)
for d in NISHI_TWINS:
    add(f"nishi_{d}_control100", "nishimura", "control", prompts="unhackable", dataset=d, n=100, seed=0)
# 6 sales: weak constraints (main) vs strict (control); about 2 turns: short (200 + 100)
add("sales_smoke", "sales", "main", prompts="weak", n=5, time_limit=900)
add("sales_main50", "sales", "main", prompts="weak", n=50, time_limit=900)
add("sales_mainB", "sales", "main", prompts="weak", n=150, offset=50, time_limit=900)
add("sales_control100", "sales", "control", prompts="strict", n=100, time_limit=900)
# 7 data_entry: fallback (main, the programmatic reward_passed) vs control (no empty task); the config's own 1800 s limit; about 8 turns: short
add("dataentry_smoke", "data_entry", "main", prompts="fallback", n=5, conns=24)
add("dataentry_main50", "data_entry", "main", prompts="fallback", n=50, conns=32)
add("dataentry_mainB", "data_entry", "main", prompts="fallback", n=150, offset=50, conns=32)
add("dataentry_control100", "data_entry", "control", prompts="control", n=100, conns=32)
# 8 hvta: hidden_solution (main) vs control (no hidden file); Wordle-v0-hardcore, up to 40 steps (agentic: 100 + 50), 1200 s; the episode seed is seed + i
add("hvta_smoke", "hvta", "main", prompts="hidden_solution", n=5, seed=0, time_limit=1200, conns=32)
add("hvta_main50", "hvta", "main", prompts="hidden_solution", n=50, seed=0, time_limit=1200, conns=32)
add("hvta_mainB", "hvta", "main", prompts="hidden_solution", n=50, seed=50, time_limit=1200, conns=32)
add("hvta_control50", "hvta", "control", prompts="control", n=50, seed=0, time_limit=1200, conns=32)
# 10 matonski: expected.json beside the grader (main only)
add("matonski_smoke", "matonski", "main", prompts="default", n=5)
add("matonski_main50", "matonski", "main", prompts="default", n=50)
add("matonski_main200", "matonski", "main", prompts="default", n=50, epochs=4)   # the env has 50 prompts: 4 samples each
# 11 revert_or_refactor: default (main) vs control (tests already pass); 2400 s; agentic (100 + 50)
add("revert_smoke", "revert_or_refactor", "main", prompts="default", n=5, sandbox="local", time_limit=2400, conns=16)
add("revert_main50", "revert_or_refactor", "main", prompts="default", n=50, sandbox="local", time_limit=2400, conns=16)
add("revert_mainB", "revert_or_refactor", "main", prompts="default", n=50, offset=50, sandbox="local", time_limit=2400, conns=16)
add("revert_control50", "revert_or_refactor", "control", prompts="control", n=50, sandbox="local", time_limit=2400, conns=16)
# 12 tower_of_hanoi, 10 disks: rule (main) vs no_rule (control); 2400 s; agentic
add("hanoi_smoke", "tower_of_hanoi", "main", prompts="rule", n=5, n_disks=10, sandbox="local", time_limit=2400, conns=16)
add("hanoi_main50", "tower_of_hanoi", "main", prompts="rule", n=50, n_disks=10, sandbox="local", time_limit=2400, conns=32)
add("hanoi_mainB", "tower_of_hanoi", "main", prompts="rule", n=50, offset=50, n_disks=10, sandbox="local", time_limit=2400, conns=32)
add("hanoi_control50", "tower_of_hanoi", "control", prompts="no_rule", n=50, n_disks=10, sandbox="local", time_limit=2400, conns=32)
# 13 overclaim: main (coverage proxy) vs control (small scenario); 1800 s; agentic
add("overclaim_smoke", "overclaim", "main", prompts="main", n=5, time_limit=1800, conns=16)
add("overclaim_main50", "overclaim", "main", prompts="main", n=50, time_limit=1800, conns=32)
add("overclaim_mainB", "overclaim", "main", prompts="main", n=50, offset=50, time_limit=1800, conns=32)
add("overclaim_control50", "overclaim", "control", prompts="control", n=50, time_limit=1800, conns=32)

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def task_ids(log) -> list:
    return sorted({s.id for s in log.samples}, key=str)


def sample_end(s, spec: dict, records_by_key: dict) -> dict:
    r = records_by_key.get((str(s.id), s.epoch))
    limit = None
    if s.limit is not None:
        limit = {"time": "time", "working": "time", "message": "message", "token": "token"}.get(s.limit.type, s.limit.type)
    out_tokens = sum((u.output_tokens or 0) for u in (s.model_usage or {}).values())
    return dict(ts=now(), event="sample_end", split=spec["arm"], agent_type=spec["env"], task_id=str(s.id), epoch=s.epoch,
                score="C" if (r and r.get("cheated")) else "I", error=(s.error.message[:300] if s.error else None), limit=limit,
                n_turns=(r or {}).get("n_turns"), output_tokens=out_tokens, seconds=round(s.total_time or 0, 1))


def run(row: str, name: str) -> None:
    from inspect_ai import eval as inspect_eval
    from rhenvs.convert import write_records
    spec = RUNS[name]
    env = spec["env"]
    rdir = OUT / row / env / name
    if (rdir / "records.jsonl").exists():
        print(f"SKIP {row} {name}: {rdir} has records"); return
    rdir.mkdir(parents=True, exist_ok=True)
    fn = getattr(importlib.import_module(f"rhenvs.{env}"), env)
    task = fn(**spec["args"])
    task.metadata = (task.metadata or {}) | {"gemma_envs_run": name, "row": row, "arm": spec["arm"], "mission": "sweep/mission_gemma_envs.md"}
    model = f"vllm/cb_{row}"
    cfg = dict(run_id=f"{row}__{name}", created=now(), config=dict(model=model, served_model=f"google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475 + hook ~/hooks/{row}.safetensors",
               samples_per_task=spec["args"].get("epochs", 1), vllm_base_url=BASE_URL, env=env, arm=spec["arm"], task_args=spec["args"], time_limit=spec["time_limit"],
               max_connections=spec["conns"], sampling=dict(task.config.model_dump(exclude_none=True)) if task.config else None),
               impossible_variants=["main"], selected_task_ids={env: {spec["arm"]: [str(s.id) for s in task.dataset]}})
    (rdir / "run_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"=== {row} {name}: {env} {spec['args']} time_limit {spec['time_limit']} conns {spec['conns']} ({len(task.dataset)} samples) {now()}", flush=True)
    start = time.time()
    [log] = inspect_eval(task, model=model, model_base_url=BASE_URL, max_connections=spec["conns"], display="plain", log_dir=str(ROOT / "logs" / env), log_model_api=True,
                         model_args={"client_timeout": 3600}, time_limit=spec["time_limit"], fail_on_error=False, retry_on_error=1)
    out, records = write_records(log, ROOT / "data" / "inspect")
    by_key = {(r["rollout_id"].split("/")[-2], int(r["rollout_id"].split("/")[-1])): r for r in records}
    with (rdir / "progress.jsonl").open("w") as f:
        for s in log.samples:
            f.write(json.dumps(dict(ts=now(), event="sample_start", split=spec["arm"], agent_type=env, task_id=str(s.id), epoch=s.epoch)) + "\n")
            f.write(json.dumps(sample_end(s, spec, by_key)) + "\n")
    shutil.copy(out, rdir / "records.jsonl")
    shutil.copy(log.location, rdir / Path(log.location).name)
    cfg["status"], cfg["log"], cfg["finished"], cfg["seconds"] = log.status, Path(log.location).name, now(), round(time.time() - start)
    (rdir / "run_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    summarize(rdir)


def summarize(rdir: Path) -> None:
    recs = [json.loads(l) for l in (rdir / "records.jsonl").open()]
    prog = [json.loads(l) for l in (rdir / "progress.jsonl").open() if '"sample_end"' in l]
    n, hacks = len(recs), sum(bool(r.get("cheated")) for r in recs)
    lim = sum(p["limit"] == "time" for p in prog); err = sum(p["error"] is not None for p in prog)
    print(f"RUN_DONE {rdir.relative_to(OUT)}: n {n}, cheated {hacks} ({100 * hacks / max(n, 1):.1f}%), time-limited {lim}, errors {err}, mean turns {sum(r.get('n_turns', 0) for r in recs) / max(n, 1):.1f} {now()}", flush=True)


if __name__ == "__main__":
    stage, args = (sys.argv[1] if len(sys.argv) > 1 else "list"), sys.argv[2:]
    if stage == "list":
        print(__doc__)
        for k, v in RUNS.items(): print(f"{k:40s} {v}")
    elif stage == "run":
        row, names = args[0], args[1:]
        assert all(n in RUNS for n in names), [n for n in names if n not in RUNS]
        for name in names:
            try:
                run(row, name)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"RUN_FAILED {row} {name}: {e!r} {now()}", flush=True)
    elif stage == "summarize":
        for d in args: summarize(Path(d))
    else:
        print(__doc__); sys.exit(1)
