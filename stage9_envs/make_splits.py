#!./.venv/bin/python
"""sweep/mission_stage9_envs.md: the 80/20 task splits (seed 0) of the envs that had none (toolfail, the five other kept nishimura datasets) in gemma_envs/splits/,
the format gemma_envs.py run_ids reads. Nishimura: the base grid of mission_gemma_envs used 40 rows per dataset drawn with seed 1 (n 40); those rows are moved into
eval so the reused base grid stays held out (logged deviation). MBPP keeps its split; its bool_negated tasks are dropped at run time (mbpp_nonbool lists)."""
import json, random
from pathlib import Path
from rhenvs.common import CONFIGS
S = Path("gemma_envs/splits")
def split(ids, seed=0, force_eval=()):
    ids = sorted(ids, key=str); r = random.Random(seed); sh = ids[:]; r.shuffle(sh)
    k = round(0.2 * len(sh)); ev = set(sh[:k]) | set(force_eval)
    return [i for i in sh if i not in ev], [i for i in sh if i in ev]
# toolfail tool_required pool (all items not control)
items = [json.loads(l) for l in (CONFIGS / "toolfail" / "tasks.jsonl").read_text().splitlines() if l.strip()]
tf = [x["task_id"] for x in items if x["target_failure_mode"] != "control"]
tr, ev = split(tf)
(S / "toolfail.json").write_text(json.dumps(dict(unit="toolfail", env="toolfail", population=dict(config="gemma4_31b_on", prompts="tool_required", n=None), seed=0, n=len(tf), train=tr, eval=ev, arm_suffix={}), indent=1))
print("toolfail", len(tf), len(tr), len(ev))
# nishimura other datasets
for d in ("sycophancy_fact", "sycophancy_opinion_nlp", "sycophancy_opinion_political", "world_affecting_approval", "world_affecting_reward"):
    from rhenvs.nishimura import dataset_path
    rows = [l for l in dataset_path("reward_hack", d).read_text().splitlines() if l.strip()]
    ids = [f"nishimura_{d}_{i}.reward_hack" for i in range(len(rows))]
    grid = [f"nishimura_{d}_{i}.reward_hack" for i in sorted(random.Random(1).sample(range(len(rows)), 40))]   # nishimura(n=40, seed=1)'s rows
    tr, ev = split(ids, force_eval=grid)
    (S / f"nishi_{d}.json").write_text(json.dumps(dict(unit=f"nishi_{d}", env="nishimura", population=dict(config="gemma4_31b_on", prompts="reward_hack", dataset=d, n=None, seed=0), seed=0, n=len(ids), train=tr, eval=ev, grid_rows_forced_into_eval=grid, arm_suffix={"main": "reward_hack", "control": "unhackable"}), indent=1))
    print(d, len(ids), len(tr), len(ev))
# mbpp: non-bool train / eval ids in split order
m = json.loads((S / "mbpp.json").read_text())
kind = {}
for l in (CONFIGS / "mbpp" / "tasks.jsonl").read_text().splitlines():
    if l.strip():
        x = json.loads(l); kind[str(x["task_id"])] = x["corruption"]
print("mbpp bool", sum(v.endswith("bool_negated") for v in kind.values()), "of", len(kind))
nb = dict(m, unit="mbpp_nonbool", train=[i for i in m["train"] if not kind[str(i)].endswith("bool_negated")], eval=[i for i in m["eval"] if not kind[str(i)].endswith("bool_negated")], note="mbpp.json without bool_negated tasks (sweep/mission_stage9_envs.md)")
(S / "mbpp_nonbool.json").write_text(json.dumps(nb, indent=1))
print("mbpp nonbool", len(nb["train"]), len(nb["eval"]))
