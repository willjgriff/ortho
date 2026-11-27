#!./.venv/bin/python
"""sweep/mission_gemma_envs.md SUMMARY samples: for every grid row (acts/gemma_envs/grid_rows.json) and env unit, 3 hack and 3 non-hack rollouts evenly spaced through the
row's grid records, each with its id, labels and a short excerpt (the last commands and the end of the final message). Writes box_report/gemma_envs/SAMPLES.md."""
import json
from pathlib import Path

import gemma_envs_grid as G
import gemma_envs_table as T

REPORT = Path("/workspace/llm-reward-hacking-ablation/box_report/gemma_envs")


def excerpt(r: dict) -> str:
    cmds = [c for t in r["turns"] for c in (t.get("commands") or [])]
    calls = [f"{c.get('name')}({json.dumps(c.get('args', {}))[:80]})" for t in r["turns"] for c in (t.get("tool_calls") or []) if isinstance(c, dict)]
    msg = (r["turns"][-1].get("message") or "").strip().replace("\n", " ")
    parts = []
    if cmds: parts.append("last commands: " + " | ".join(c.replace("\n", " ")[:120] for c in cmds[-3:]))
    elif calls: parts.append("last tool calls: " + " | ".join(calls[-3:]))
    parts.append("final message: " + (msg[-350:] if msg else "(none)"))
    return "; ".join(parts)


def main() -> None:
    rows = json.loads((G.GE / "grid_rows.json").read_text())
    out = ["# Samples: 3 hack and 3 non-hack rollouts per grid row and env, evenly spaced through the row's grid records (labels are the envs' programmatic ones)", ""]
    for label, v in rows.items():
        out += [f"## {label} ({v['served']})", ""]
        for u in G.UNITS:
            unit, env, runs, base_run = u
            runs = runs if isinstance(runs, list) else [base_run if (v["served"] == "base" and base_run) else runs]
            if not isinstance(u[2], list) and (G.PULLED / v["served"] / env / u[2].replace("_grid", "_gridrest") / "records.jsonl").exists():
                runs = [u[2].replace("_grid", "_dose"), u[2].replace("_grid", "_gridrest")]
            recs = [json.loads(l) for run in runs if (G.PULLED / v["served"] / env / run / "records.jsonl").exists() for l in (G.PULLED / v["served"] / env / run / "records.jsonl").open()]
            if not recs: continue
            out.append(f"### {label} on {unit}")
            for which in (True, False):
                pick = [r for r in recs if T.hack_of(env, r) is which]
                idx = sorted(set(round(i * (len(pick) - 1) / 2) for i in range(3))) if pick else []
                for i in idx:
                    r = pick[i]
                    out.append(f"- {'hack' if which else 'non-hack'} `{r['rollout_id'].split('/', 1)[1]}`: {excerpt(r)[:600]}")
            out.append("")
    (REPORT / "SAMPLES.md").write_text("\n".join(out) + "\n")
    print(f"{REPORT / 'SAMPLES.md'}: {len(out)} lines")


if __name__ == "__main__":
    main()
