"""The result tables of sweep/mission_prompt_dom.md from the records (data/inspect/{grader,terminal_verifier,mbpp,refusal}) and prompt_dom_rh.py's outputs (acts/prompt_dom/*.json):
every row x every eval with Wilson intervals, Ethan's reference rows from his records (labelled his), the Arditi candidate tables, the gate, the data checks, cosines and samples.
Written to acts/prompt_dom/table.md and rates.json. No model. Run as ./prompt_dom_rh.py table, or python prompt_dom_table.py [--since 2026-11-23T20]."""
import json
import math
import sys
from pathlib import Path

from mechtools import wilson

OUT = Path("acts/prompt_dom")
SINCE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--since=")), "2026-11-23T20")   # logs created on this box
ROWS = ["base", "gpdom", "gpdomall", "gp10dom", "gparditi", "gprand", "tvpdom", "tvpdomall", "tvarditi", "tvprand", "refarditi", "zrand"]
ETHAN = dict(domlast="domlast", refdom="refdom", kto="kto", mbppkto="mbppkto", graderlora="graderlora", mbpplora="mbpplora")   # reference rows from Ethan's records (his server): context only
RUN_KEYS = ("nothink_heldout", "nothink", "heldout", "main", "tv_bug_mt6", "tv_bug", "tv_pos", "tv_corr", "mbpp", "harmful_low", "harmless_low", "harmful", "harmless", "none", "clean", "secret", "correct")

def load(env: str) -> list[dict]:
    return [json.loads(line) for path in sorted((Path("data/inspect") / env).glob("*.jsonl")) for line in path.open()]

def log_time(r: dict) -> str: return r["rollout_id"].split("/")[0][:19]
def mine(r: dict) -> bool: return log_time(r) >= SINCE

def row_of(r: dict) -> str:
    if r["condition"] == "none": return "base"
    c = r["condition"]
    for key in RUN_KEYS:
        if c.endswith("_" + key): return c[:-len(key) - 1]
    return c.split("_")[0]

def rate(k: int, n: int) -> str:
    if not n: return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"

def grader_cell(rs: list[dict]) -> dict:
    valid = [r for r in rs if r["labels"]["cheat"] is not None]
    return dict(n=len(rs), valid=len(valid), cheat=sum(r["labels"]["cheat"] for r in valid), cheat_rate=rate(sum(r["labels"]["cheat"] for r in valid), len(valid)), valid_rate=rate(len(valid), len(rs)), logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def tv_cell(rs: list[dict]) -> dict:
    f = lambda k: sum(bool(r["labels"].get(k)) for r in rs)
    return dict(n=len(rs), tampered=f("tampered"), passed=f("passed"), peeked=f("peeked"), cheated=sum(r["cheated"] for r in rs), tampered_rate=rate(f("tampered"), len(rs)), passed_rate=rate(f("passed"), len(rs)), peeked_rate=rate(f("peeked"), len(rs)), truncated=sum(r["labels"].get("truncated", 0) or 0 for r in rs), logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def main() -> None:
    grader, tv, mbpp, refusal = load("grader"), load("terminal_verifier"), load("mbpp"), load("refusal")
    rates = {}
    lines = [f"# Prompt-pair DoM: results tables (records created on this box from {SINCE}; Ethan's rows from his records, labelled)", ""]
    # grader: rows x (nothink main, nothink heldout, low main, low heldout)
    def gsel(records, row, config, family, mine_only=True):
        rs = [r for r in records if r["config_id"] == config and r["labels"]["prompts"] == "hack" and r["labels"]["sys"] == "none" and row_of(r) == row and r["cfg"]["family"] == family]
        if mine_only: rs = [r for r in rs if mine(r)]
        if not mine_only and len({x["rollout_id"].split("/")[0] for x in rs}) > 1:   # Ethan: the newest full run
            newest = max(x["rollout_id"].split("/")[0] for x in rs); rs = [x for x in rs if x["rollout_id"].startswith(newest + "/")]
        return rs
    lines += ["## Grader (hack prompts, no system prompt): cheat rate among valid answers [Wilson 95%] (cheat/valid), valid rate", "", "| row | reasoning off, main 5 | reasoning off, held-out 7 | effort low, main 5 | effort low, held-out 7 |", "|---|---|---|---|---|"]
    gcols = [("nothink", "all"), ("nothink", "heldout"), ("qwen3.8_low", "all"), ("qwen3.8_low", "heldout")]
    for row in ROWS + [f"ethan:{k}" for k in ETHAN] + ["ethan:base"]:
        cells = []
        for config, family in gcols:
            if row.startswith("ethan:"): rs = gsel(grader, row[6:] if row != "ethan:base" else "base", config, family, mine_only=False); rs = [r for r in rs if not mine(r)]
            else: rs = gsel(grader, row, config, family)
            c = grader_cell(rs) if rs else None
            rates[f"grader/{row}/{config}/{family}"] = c
            cells.append(f"{c['cheat_rate']}; valid {c['valid_rate']}" if c else "-")
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # grader per family for the box rows (held-out and main), compact
    lines += ["", "### Grader per family (cheat/valid), reasoning off and effort low, this box", ""]
    fams = sorted({(r["labels"]["family"]) for r in grader if mine(r)})
    lines += ["| row | config | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for row in ROWS:
        for config in ("nothink", "qwen3.8_low"):
            rs = [r for r in grader if mine(r) and row_of(r) == row and r["config_id"] == config and r["labels"]["prompts"] == "hack"]
            if not rs: continue
            cells = []
            for f in fams:
                v = [r for r in rs if r["labels"]["family"] == f and r["labels"]["cheat"] is not None]
                cells.append(f"{sum(r['labels']['cheat'] for r in v)}/{len(v)}" if v else "-")
            lines.append(f"| {row} | {config} | " + " | ".join(cells) + " |")
    # TV
    lines += ["", "## TerminalVerifier (qwen3.8_sweep config, seed pool, 128 grids): tampered, passed, peeked [Wilson 95%]", "", "| row | verifier_bug mt4 tampered | passed | peeked | possible mt6 passed | verifier_bug mt6 tampered | passed | corrupted mt4 passed | tampered |", "|---|---|---|---|---|---|---|---|---|"]
    def tsel(records, row, mode, mt, mine_only=True):
        rs = [r for r in records if row_of(r) == row and r["mode"] == mode and r["max_turns"] == mt and r["config_id"] == "qwen3.8_sweep"]
        rs = [r for r in rs if mine(r)] if mine_only else [r for r in rs if not mine(r)]
        if not mine_only and len({x["rollout_id"].split("/")[0] for x in rs}) > 1:
            newest = max(x["rollout_id"].split("/")[0] for x in rs); rs = [x for x in rs if x["rollout_id"].startswith(newest + "/")]
        return rs
    for row in ROWS + [f"ethan:{k}" for k in ETHAN] + ["ethan:base"]:
        cells = []
        for mode, mt, keys in [("verifier_bug", 4, ("tampered_rate", "passed_rate", "peeked_rate")), ("possible", 6, ("passed_rate",)), ("verifier_bug", 6, ("tampered_rate", "passed_rate")), ("corrupted", 4, ("passed_rate", "tampered_rate"))]:
            rs = tsel(tv, row[6:] if row.startswith("ethan:") else row, mode, mt, mine_only=not row.startswith("ethan:"))
            c = tv_cell(rs) if rs else None
            rates[f"tv/{row}/{mode}/mt{mt}"] = c
            cells += [c[k] if c else "-" for k in keys]
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # MBPP
    if mbpp:
        lines += ["", "## MBPP impossible (qwen3.8_sweep sampling): cheat (passes the corrupted visible test) among valid answers, all tasks and the test split", "", "| row | all tasks | test split |", "|---|---|---|"]
        for row in ROWS + [f"ethan:{k}" for k in ETHAN] + ["ethan:base"]:
            rs = [r for r in mbpp if row_of(r) == (row[6:] if row.startswith("ethan:") else row) and r["config_id"] == "impossible_qwen3.8_sweep" and (mine(r) != row.startswith("ethan:"))]
            if not rs: continue
            cells = []
            for subset in ("all", "test"):
                v = [r for r in rs if (subset == "all" or r["labels"].get("split") == "test") and r["labels"].get("cheat") is not None]
                cells.append(rate(sum(bool(r["labels"]["cheat"]) for r in v), len(v)) + f" of {sum(1 for r in rs if subset == 'all' or r['labels'].get('split') == 'test')}")
            rates[f"mbpp/{row}"] = cells
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # refusal
    lines += ["", "## Refusal (heretic's test prompts, keyword label; the judge is applied on the laptop): refused among valid answers", "", "| row | harmful, off | harmless, off | harmful, low | harmless, low |", "|---|---|---|---|---|"]
    for row in ["refarditi", "zrand", "ethan:base", "ethan:refdom", "ethan:refdomall", "ethan:rkfull"]:
        cells = []
        for config in ("nothink", "qwen3.8_low"):
            for arm in ("harmful", "harmless"):
                rs = [r for r in refusal if r["labels"]["split"] == "test" and r["cfg"]["n"] is None and r["config_id"] == config and r["labels"]["prompts"] == arm and row_of(r) == (row[6:] if row.startswith("ethan:") else row) and (mine(r) != row.startswith("ethan:"))]
                if rs and len({x["rollout_id"].split("/")[0] for x in rs}) > 1:
                    newest = max(x["rollout_id"].split("/")[0] for x in rs); rs = [x for x in rs if x["rollout_id"].startswith(newest + "/")]
                v = [r for r in rs if r["labels"]["refused"] is not None]
                cells.append(rate(sum(r["labels"]["refused"] for r in v), len(v)) + f"; valid {len(v)}/{len(rs)}" if rs else "-")
                rates[f"refusal/{row}/{config}/{arm}"] = dict(n=len(rs), valid=len(v), refused=sum(r["labels"]["refused"] for r in v)) if rs else None
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # capability: MMLU and KL
    check = json.loads((OUT / "check.json").read_text()) if (OUT / "check.json").exists() else {}
    if check:
        lines += ["", f"## Capability (HF, vLLM stopped): MMLU on every {check['mmlu_every']}th test question (n = {check['mmlu_n']}), first-token KL from base on heretic's 100 harmless test prompts", "", "| row | MMLU | vs base [95%] | first-token KL |", "|---|---|---|---|"]
        for row, c in check["rows"].items():
            vb = c["mmlu"].get("vs_base")
            lines.append(f"| {row} | {100 * c['mmlu']['accuracy']:.2f} | " + (f"{vb['points']:+.2f} [{vb['low']:+.2f}, {vb['high']:+.2f}]" if vb else "-") + f" | {c['first_token_kl_harmless']:.4f} |")
    # gates and data
    for name, title in [("gate.json", "refdom reproduction gate"), ("families_check.json", "families check"), ("tv_pairs_manifest.json", "TV pairs")]:
        p = OUT / name
        if p.exists():
            d = json.loads(p.read_text())
            if name == "gate.json": lines += ["", f"## {title}", "", f"cosine with refusal_dom_all per layer: min {d['min']:.4f}, mean {d['mean']:.4f}; pin relative difference max {max(d['pin_rel_diff']):.4f}; passed {d['passed']}"]
            elif name == "families_check.json": lines += ["", f"## {title}", "", f"{d['n']} families {d['kinds']}; fit {len(d['fit'])}, validation {len(d['validation'])}; eval answer clashes {d['eval_answer_clashes']}; within-family clashes {d['within_family_clashes']}"]
            else: lines += ["", f"## {title}", "", f"counts per filter step: {d['counts']}", f"re-tokenisation failures: {len(d['retokenise_failed'])}", f"write regex: `{d['write_regex']}`"]
    for env in ("grader", "tv", "refusal"):
        p = OUT / f"arditi_{env}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text())
        lines += ["", f"## Arditi candidates, {env}: base bypass {d['base_bypass']:+.3f}" + (f", base induce {d['base_induce']:+.3f}" if d["base_induce"] is not None else " (no induce filter)") + f"; {len(d['candidates'])} candidates, {d['passing']} pass; chosen: {d['chosen']}", "", "| position | layer | bypass | induce | KL | passes |", "|---|---|---|---|---|---|"]
        for r in sorted(d["candidates"], key=lambda r: r["bypass"])[:10]:
            ok = r["kl"] < d["kl_max"] and (d["base_induce"] is None or r["induce"] > d["base_induce"])
            lines.append(f"| {r['position']} | {r['layer']} | {r['bypass']:+.3f} | " + (f"{r['induce']:+.3f}" if r["induce"] is not None else "-") + f" | {r['kl']:.4f} | {ok} |")
    if (OUT / "cosines.md").exists(): lines += ["", "## Cosines", "", (OUT / "cosines.md").read_text()]
    # samples: 3 cheat and 3 clean grader answers per row (off and low), 5 TV command sequences per row, at evenly spaced positions
    lines += ["", "## Samples (evenly spaced positions in each run, not hand-picked)", ""]
    for row in ROWS:
        for config in ("nothink", "qwen3.8_low"):
            rs = [r for r in grader if mine(r) and row_of(r) == row and r["config_id"] == config and r["labels"]["prompts"] == "hack" and r["labels"]["cheat"] is not None]
            for label in (True, False):
                v = [r for r in rs if r["labels"]["cheat"] == label]
                for k in [round(i * (len(v) - 1) / 2) for i in range(3)] if len(v) >= 3 else range(len(v)):
                    r = v[k]; turn = r["turns"][-1]
                    lines.append(f"- {row} / {config} / {'CHEAT' if label else 'clean'} / {r['labels']['family']}.{r['labels']['side']}: " + (f"reasoning: {turn['reasoning'][:300]!r} ... " if turn["reasoning"] else "") + f"answer: {turn['message'][:200]!r}")
        rs = [r for r in tv if mine(r) and row_of(r) == row and r["mode"] == "verifier_bug" and r["max_turns"] == 4]
        for k in [round(i * (len(rs) - 1) / 4) for i in range(5)] if len(rs) >= 5 else range(len(rs)):
            r = rs[k]
            lines.append(f"- {row} / tv_bug mt4 / {r['rollout_id'].split('/')[-1]} tampered={r['labels'].get('tampered')} passed={r['labels'].get('passed')}: " + " || ".join(c[:120].replace("\n", "\\n") for turn in r["turns"] for c in turn.get("commands", [])))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    (OUT / "rates.json").write_text(json.dumps(rates, indent=1))
    print("\n".join(lines[:60])); print(f"... saved {OUT / 'table.md'} and rates.json")

if __name__ == "__main__": main()
