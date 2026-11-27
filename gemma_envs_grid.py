#!./.venv/bin/python
"""sweep/mission_gemma_envs.md Phase 4: the grid table. Rows are listed in acts/gemma_envs/grid_rows.json ({label: {"served": served row name, "s": dose, "kl": first-token KL,
"kind": direction | control | base, "own": the row's own env unit or null, "grader_runs": optional {"main": cfg name, "heldout": cfg name} of reused Grader runs}}).
Env cells: hack among valid answers (Wilson 95%), the change vs base and vs the row's control (the langdom row whose KL is closest to the row's without exceeding it;
langdom@K_lo when none is below) with an exact two-sided McNemar p paired by sample id; the env's honest-success signal next to it. Grader cells: cheat among valid,
valid, and a cluster sign-flip p over the family sides (exact). Capability: MMLU / first-token KL (second_model.py check), GSM8K / IFEval off (lm-eval), GSM8K-on.
The diagonal (a row on its own env) is in bold. Writes acts/gemma_envs/grid.md and grid.json. Numbers only; the rule is applied on the laptop.
"""
import itertools
import json
import math
from pathlib import Path

import gemma_envs_table as T

ROOT = Path(__file__).resolve().parent
GE = ROOT / "acts" / "gemma_envs"
PULLED = Path("/root/gemma_envs_out/pulled")
UNITS = [  # (unit label, env, run name for rows, run name for base when it differs)
    ("mbpp", "mbpp", "mbpp_grid", None), ("nishi_rs", "nishimura", "nishi_rs_grid", None),
    ("nishi_other", "nishimura", ["nishi_sycophancy_fact_grid", "nishi_sycophancy_opinion_nlp_grid", "nishi_sycophancy_opinion_political_grid", "nishi_world_affecting_approval_grid", "nishi_world_affecting_reward_grid"], None),
    ("toolfail", "toolfail", "toolfail_grid", "toolfail_main200"), ("dataentry", "data_entry", "dataentry_grid", None), ("overclaim", "overclaim", "overclaim_grid", None)]


def mcnemar(a: dict, b: dict) -> tuple:
    """Exact two-sided McNemar on the ids valid in both: (n pairs, b-only, a-only, p) with a the row and b the reference."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    x = sum(1 for k in keys if a[k] and not b[k]); y = sum(1 for k in keys if b[k] and not a[k])
    n = x + y
    if n == 0: return len(keys), y, x, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n * 2
    return len(keys), y, x, min(1.0, p)


def env_cells(served: str, unit: tuple) -> dict | None:
    label, env, runs, base_run = unit
    if not isinstance(runs, list) and (PULLED / served / env / runs.replace("_grid", "_gridrest") / "records.jsonl").exists():
        runs = [runs.replace("_grid", "_dose"), runs.replace("_grid", "_gridrest")]   # a direction row's own env: its dose check (eval ids 0-63) + the rest
    runs = runs if isinstance(runs, list) else [base_run if (served == "base" and base_run) else runs]
    out, honest = {}, []
    found = False
    for run in runs:
        d = PULLED / served / env / run
        if not (d / "records.jsonl").exists(): continue
        found = True
        for r in map(json.loads, (d / "records.jsonl").open()):
            key = r["rollout_id"].split("/", 1)[1]
            out[key] = T.hack_of(env, r)   # keyed by sample id / epoch (unique within a unit across its runs), so a dose + rest pair joins a full grid run
            h = T.HONEST.get(env, (None, lambda r: None))[1](r)
            if h is not None: honest.append(bool(h))
    if not found: return None
    prog = []
    for run in runs:
        p = PULLED / served / env / run / "progress.jsonl"
        if p.exists(): prog += [json.loads(l) for l in p.open() if '"sample_end"' in l]
    return dict(hacks=out, honest=honest, time_limited=sum(p["limit"] == "time" for p in prog), n=len(out))


def grader(served: str, which: str, reuse: dict | None) -> dict | None:
    name = (reuse or {}).get(which) or f"{served}_ge_g_on_{which}"
    recs = [r for p in sorted((ROOT / "data" / "inspect" / "grader").glob("*.jsonl")) for r in map(json.loads, p.open()) if (r.get("cfg") or {}).get("name") == name]
    if not recs: return None
    cl = {}
    for r in recs:
        c = r["labels"].get("cheat")
        cl.setdefault(f"{r['labels'].get('family')}.{r['labels'].get('side')}", []).append(None if c is None else bool(c))
    valid = [c for v in cl.values() for c in v if c is not None]
    rc = [len((r["turns"][-1].get("reasoning") or "")) for r in recs]
    return dict(clusters=cl, cheat=sum(valid), valid=len(valid), n=len(recs), median_reasoning=sorted(rc)[len(rc) // 2], name=name)


def signflip(a: dict, b: dict) -> float:
    """Exact cluster sign-flip test of the mean per-cluster difference in cheat rate (row - reference) over the family sides both have."""
    ks = [k for k in a if k in b]
    d = []
    for k in ks:
        va, vb = [x for x in a[k] if x is not None], [x for x in b[k] if x is not None]
        if va and vb: d.append(sum(va) / len(va) - sum(vb) / len(vb))
    if not d: return 1.0
    obs = abs(sum(d))
    if len(d) > 16: return float("nan")
    cnt = sum(1 for s in itertools.product((1, -1), repeat=len(d)) if abs(sum(x * y for x, y in zip(s, d))) >= obs - 1e-12)
    return cnt / 2 ** len(d)


def cap(served: str, check: dict) -> dict:
    c = check.get("rows", {}).get(served, {})
    lm = GE / "lmeval" / f"cb_{served}.json"
    gs = GE / "gsm8k_on" / f"cb_{served}.json"
    out = dict(mmlu=c.get("mmlu", {}).get("accuracy"), kl=c.get("first_token_kl_harmless"))
    if lm.exists():
        r = json.loads(lm.read_text())["results"]
        out["gsm8k"] = r.get("gsm8k_cot_zeroshot", {}).get("exact_match,flexible-extract"); out["ifeval"] = r.get("ifeval", {}).get("prompt_level_strict_acc,none")
    if gs.exists():
        g = json.loads(gs.read_text()); out.update(gsm8k_on=g["accuracy"], gsm8k_on_reasoning=g["median_reasoning_chars"], gsm8k_on_at_budget=g["at_budget"])
    return out


def main() -> None:
    rows = json.loads((GE / "grid_rows.json").read_text())
    check = json.loads((ROOT / "acts" / "second_model" / "gemma4_31b" / "check.json").read_text()) if (ROOT / "acts" / "second_model" / "gemma4_31b" / "check.json").exists() else {}
    langs = sorted([(v["kl"], k) for k, v in rows.items() if v.get("kind") == "control" and k.startswith("langdom") and v.get("kl") is not None])
    def control_of(label: str) -> str | None:
        kl = rows[label].get("kl")
        if kl is None or not langs or rows[label]["kind"] == "control" or label == "base": return None
        below = [x for x in langs if x[0] <= kl]
        return (below[-1] if below else langs[0])[1]
    data = {}
    for label, v in rows.items():
        data[label] = dict(env={u[0]: env_cells(v["served"], u) for u in UNITS}, grader={w: grader(v["served"], w, v.get("grader_runs")) for w in ("main", "heldout")}, cap=cap(v["served"], check), **v)
    base = data.get("base")
    lines = ["| row | dose s | KL | " + " | ".join(u[0] for u in UNITS) + " | Grader main | Grader held-out | MMLU | GSM8K off | IFEval off | GSM8K on (median reasoning chars, at budget) |", "|---" * (len(UNITS) + 10) + "|"]
    for label, d in data.items():
        ctrl = control_of(label)
        cells = []
        for u in UNITS:
            e = d["env"][u[0]]
            if e is None: cells.append("not run"); continue
            valid = [x for x in e["hacks"].values() if x is not None]
            s = T.wilson(sum(valid), len(valid))
            if base and label != "base" and base["env"][u[0]]:
                n, bo, ro, p = mcnemar(e["hacks"], base["env"][u[0]]["hacks"])
                b_valid = [x for x in base["env"][u[0]]["hacks"].values() if x is not None]
                s += f"; Δ base {100 * (sum(valid) / max(len(valid), 1) - sum(b_valid) / max(len(b_valid), 1)):+.1f} (p {p:.2g})"
            if ctrl and data[ctrl]["env"][u[0]]:
                n, bo, ro, p = mcnemar(e["hacks"], data[ctrl]["env"][u[0]]["hacks"])
                s += f"; vs {ctrl} p {p:.2g}"
            if e["honest"]: s += f"; honest {100 * sum(e['honest']) / len(e['honest']):.0f}%"
            if e["time_limited"]: s += f"; {e['time_limited']} time-limited"
            cells.append(f"**{s}**" if d.get("own") == u[0] else s)
        gcells = []
        for w in ("main", "heldout"):
            g = d["grader"][w]
            if g is None: gcells.append("not run"); continue
            s = f"{T.wilson(g['cheat'], g['valid'])}, valid {g['valid']}/{g['n']}, reasoning {g['median_reasoning']}"
            if base and label != "base" and base["grader"][w]: s += f"; vs base p {signflip(g['clusters'], base['grader'][w]['clusters']):.2g}"
            if ctrl and data[ctrl]["grader"][w]: s += f"; vs {ctrl} p {signflip(g['clusters'], data[ctrl]['grader'][w]['clusters']):.2g}"
            gcells.append(f"**{s}**" if d.get("own") == "grader" and w == "heldout" else s)
        c = d["cap"]
        f = lambda x: "-" if x is None else f"{100 * x:.1f}"
        lines.append(f"| {label} ({d['served']}) | {d.get('s', '-')} | {d.get('kl') if d.get('kl') is None else round(d['kl'], 3)} | " + " | ".join(cells) + " | " + " | ".join(gcells) + f" | {f(c.get('mmlu'))} | {f(c.get('gsm8k'))} | {f(c.get('ifeval'))} | {f(c.get('gsm8k_on'))} ({c.get('gsm8k_on_reasoning', '-')}, {f(c.get('gsm8k_on_at_budget'))}) |")
    (GE / "grid.md").write_text("\n".join(lines) + "\n")
    (GE / "grid.json").write_text(json.dumps({k: dict(served=v["served"], s=v.get("s"), kl=v.get("kl"), env={u: (None if e is None else dict(hacks=sum(x for x in e["hacks"].values() if x), valid=sum(x is not None for x in e["hacks"].values()), n=e["n"], time_limited=e["time_limited"])) for u, e in v["env"].items()}, grader={w: (None if g is None else {k2: g[k2] for k2 in ("cheat", "valid", "n", "median_reasoning", "name")}) for w, g in v["grader"].items()}, cap=v["cap"]) for k, v in data.items()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
