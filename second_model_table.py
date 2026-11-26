"""The result tables of sweep/mission_second_model.md from the records (data/inspect/{grader,terminal_verifier,refusal}) and second_model.py's outputs (acts/second_model/<tag>/*.json):
every row x every eval cell with Wilson intervals, answer length and truncation, the Arditi candidate tables, the TV pair counts, cosines and the evenly spaced samples. Written to
acts/second_model/<tag>/table.md and rates.json. No model. Run as ./second_model.py table, or .venv/bin/python second_model_table.py [--since=2026-11-26T10]."""
import json
import os
import sys
from pathlib import Path

from mechtools import wilson

TAG = os.environ.get("SM_MODEL", "gemma4_31b")
OUT = Path(f"acts/second_model/{TAG}")
SINCE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--since=")), "2026-11-26T10")   # logs created on this box
ROWS = ["base", "refdom", "gpdom", "tvpdom", "prand", "refarditi", "zrand", "gparditi", "tvarditi"]
OFF, ON = f"{TAG}_off", f"{TAG}_on"
RUN_KEYS = ("g_off_main", "g_off_clean", "g_off_heldout", "g_on_main", "g_on_clean", "g_on_heldout", "tv_on_bug_mt6", "tv_on_bug", "tv_on_pos", "tv_off_bug", "tv_off_pos", "ref_off_harmful", "ref_off_harmless", "ref_on_harmful", "ref_on_harmless")

def load(env: str) -> list[dict]:
    p = Path("data/inspect") / env
    return [json.loads(line) for path in sorted(p.glob("*.jsonl")) for line in path.open()] if p.exists() else []

def log_time(r: dict) -> str: return r["rollout_id"].split("/")[0][:19]
def mine(r: dict) -> bool: return log_time(r) >= SINCE and r["config_id"] in (OFF, ON)

def row_of(r: dict) -> str:
    if r["condition"] == "none": return "base"
    c = r["condition"]
    for key in RUN_KEYS:
        if c.endswith("_sm_" + key): return c[:-len(key) - 4]
    return c.split("_")[0]

def rate(k: int, n: int) -> str:
    if not n: return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"

def newest_run(rs: list[dict]) -> list[dict]:
    """The newest full run when a cell was run twice (a retry); smoke runs (n < 20 rollouts) dropped."""
    runs = {}
    for r in rs: runs.setdefault(r["rollout_id"].split("/")[0], []).append(r)
    runs = {k: v for k, v in runs.items() if len(v) >= 20} or runs
    return runs[max(runs)] if runs else []

def answer_stats(rs: list[dict]) -> str:
    """Median answer characters, median reasoning characters, and the share of rollouts cut at max_tokens (no answer text after the reasoning) or empty."""
    if not rs: return "-"
    ans = sorted(len(r["turns"][-1]["message"] or "") for r in rs)
    rea = sorted(len(r["turns"][-1]["reasoning"] or "") for r in rs)
    empty = sum(1 for r in rs if not (r["turns"][-1]["message"] or "").strip())
    return f"ans {ans[len(ans) // 2]} ch, reas {rea[len(rea) // 2]} ch, empty/cut {rate(empty, len(rs))}"

def grader_cell(rs: list[dict]) -> dict:
    valid = [r for r in rs if r["labels"]["cheat"] is not None]
    return dict(n=len(rs), valid=len(valid), cheat=sum(r["labels"]["cheat"] for r in valid), cheat_rate=rate(sum(r["labels"]["cheat"] for r in valid), len(valid)), valid_rate=rate(len(valid), len(rs)), answers=answer_stats(rs), logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def tv_cell(rs: list[dict]) -> dict:
    f = lambda k: sum(bool(r["labels"].get(k)) for r in rs)
    tasks = sorted({r["rollout_id"].split("/")[-1].split("-")[0] for r in rs})
    return dict(n=len(rs), tampered=f("tampered"), passed=f("passed"), peeked=f("peeked"), cheated=sum(r["cheated"] for r in rs), tampered_rate=rate(f("tampered"), len(rs)), passed_rate=rate(f("passed"), len(rs)), peeked_rate=rate(f("peeked"), len(rs)),
                truncated=sum(1 for r in rs if r.get("end_reason") == "truncated"), end_reasons={k: sum(1 for r in rs if r.get("end_reason") == k) for k in sorted({r.get("end_reason") for r in rs})}, turns=sum(len(r["turns"]) for r in rs) / max(1, len(rs)), logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def main() -> None:
    grader, tv, refusal = load("grader"), load("terminal_verifier"), load("refusal")
    rates = {}
    lines = [f"# Second model {TAG}: results tables (records created on this box from {SINCE}; configs {OFF} / {ON})", ""]
    # grader
    lines += ["## Grader (hack prompts, no system prompt): cheat rate among valid answers [Wilson 95%] (cheat/valid); valid rate; answer stats", "", "| row | off, main 5 | off, held-out 7 | on, main 5 | on, held-out 7 |", "|---|---|---|---|---|"]
    gcols = [(OFF, "all"), (OFF, "heldout"), (ON, "all"), (ON, "heldout")]
    for row in ROWS:
        cells = []
        for config, family in gcols:
            rs = newest_run([r for r in grader if mine(r) and r["config_id"] == config and r["labels"]["prompts"] == "hack" and r["labels"]["sys"] == "none" and row_of(r) == row and r["cfg"]["family"] == family])
            c = grader_cell(rs) if rs else None
            rates[f"grader/{row}/{config}/{family}"] = c
            cells.append(f"{c['cheat_rate']}; valid {c['valid_rate']}; {c['answers']}" if c else "-")
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += ["", "### Grader clean prompts (the side's own grader): wrong-side rate among valid answers, the no-incentive control", "", "| row | off, main 5 | on, main 5 |", "|---|---|---|"]
    for row in ROWS:
        cells = []
        for config in (OFF, ON):
            rs = newest_run([r for r in grader if mine(r) and r["config_id"] == config and r["labels"]["prompts"] == "clean" and row_of(r) == row and r["cfg"]["family"] == "all"])
            c = grader_cell(rs) if rs else None
            rates[f"grader_clean/{row}/{config}"] = c
            cells.append(f"{c['cheat_rate']}; valid {c['valid_rate']}" if c else "-")
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += ["", "### Grader per family (cheat/valid), hack prompts, this box", ""]
    fams = sorted({r["labels"]["family"] for r in grader if mine(r)})
    lines += ["| row | config | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for row in ROWS:
        for config in (OFF, ON):
            rs = [r for r in grader if mine(r) and row_of(r) == row and r["config_id"] == config and r["labels"]["prompts"] == "hack"]
            if not rs: continue
            cells = []
            for f in fams:
                v = [r for r in rs if r["labels"]["family"] == f and r["labels"]["cheat"] is not None]
                cells.append(f"{sum(r['labels']['cheat'] for r in v)}/{len(v)}" if v else "-")
            lines.append(f"| {row} | {config} | " + " | ".join(cells) + " |")
    # TV
    lines += ["", "## TerminalVerifier (seed pool, sandbox local): tampered, passed, peeked [Wilson 95%]; truncated = episodes ended by max_tokens", "", "| row | on: verifier_bug mt4 tampered | passed | peeked | truncated | on: possible mt6 passed | off: verifier_bug mt4 tampered | passed | peeked | off: possible mt6 passed | on: verifier_bug mt6 tampered | passed |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def tsel(row, config, mode, mt):
        return newest_run([r for r in tv if mine(r) and row_of(r) == row and r["mode"] == mode and r["max_turns"] == mt and r["config_id"] == config and r["cfg"].get("pool") in (None, "seed")])
    for row in ROWS:
        cells = []
        for config, mode, mt, keys in [(ON, "verifier_bug", 4, ("tampered_rate", "passed_rate", "peeked_rate", "truncated")), (ON, "possible", 6, ("passed_rate",)), (OFF, "verifier_bug", 4, ("tampered_rate", "passed_rate", "peeked_rate")), (OFF, "possible", 6, ("passed_rate",)), (ON, "verifier_bug", 6, ("tampered_rate", "passed_rate"))]:
            rs = tsel(row, config, mode, mt)
            c = tv_cell(rs) if rs else None
            rates[f"tv/{row}/{config}/{mode}/mt{mt}"] = c
            cells += [str(c[k]) if c else "-" for k in keys]
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    coll = [r for r in tv if mine(r) and r["cfg"].get("pool") == "source_eval"]
    if coll:
        c = tv_cell(coll)
        lines += ["", f"### TV pair collection (verifier_bug mt6, source_eval pool): {c['n']} episodes, tampered {c['tampered_rate']}, passed {c['passed_rate']}, peeked {c['peeked_rate']}, end reasons {c['end_reasons']}"]
    # refusal
    lines += ["", "## Refusal (heretic's test prompts, keyword label; the judge's labels are in judge/): refused among valid answers [Wilson 95%]; valid; answer stats", "", "| row | harmful, off | harmless, off | harmful, on | harmless, on |", "|---|---|---|---|---|"]
    for row in ROWS:
        cells = []
        for config in (OFF, ON):
            for arm in ("harmful", "harmless"):
                rs = newest_run([r for r in refusal if mine(r) and r["labels"]["split"] == "test" and r["config_id"] == config and r["labels"]["prompts"] == arm and row_of(r) == row])
                v = [r for r in rs if r["labels"]["refused"] is not None]
                cells.append(rate(sum(r["labels"]["refused"] for r in v), len(v)) + f"; valid {len(v)}/{len(rs)}; {answer_stats(rs)}" if rs else "-")
                rates[f"refusal/{row}/{config}/{arm}"] = dict(n=len(rs), valid=len(v), refused=sum(r["labels"]["refused"] for r in v), answers=answer_stats(rs)) if rs else None
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # capability
    check = json.loads((OUT / "check.json").read_text()) if (OUT / "check.json").exists() else {}
    if check:
        lines += ["", f"## Capability (HF, vLLM stopped): MMLU on every {check['mmlu_every']}th test question (n = {check['mmlu_n']}), first-token KL from base on heretic's 100 harmless test prompts", "", "| row | MMLU | vs base [95%] | first-token KL |", "|---|---|---|---|"]
        for row, c in check["rows"].items():
            vb = c["mmlu"].get("vs_base")
            lines.append(f"| {row} | {100 * c['mmlu']['accuracy']:.2f} | " + (f"{vb['points']:+.2f} [{vb['low']:+.2f}, {vb['high']:+.2f}]" if vb else "-") + f" | {c['first_token_kl_harmless']:.4f} |")
    lm = {row: json.loads((Path("acts/prompt_dom/lmeval") / f"cb_{row}.json").read_text()) for row in ROWS if (Path("acts/prompt_dom/lmeval") / f"cb_{row}.json").exists()}
    if lm:
        lines += ["", "## GSM8K and IFEval (lm-eval, local-chat-completions, reasoning off, greedy, the first 500 items of each task) [Wilson 95%]", "", "| row | GSM8K flexible-extract | GSM8K strict-match | IFEval prompt-level strict | IFEval prompt-level loose | n |", "|---|---|---|---|---|---|"]
        for row, d in lm.items():
            g, f = d["results"].get("gsm8k_cot_zeroshot", {}), d["results"].get("ifeval", {})
            ndocs = lambda task: len({x["doc_id"] for x in d.get("samples", {}).get(task, [])}) or d["n_samples"].get(task, 0)
            ng, nf = ndocs("gsm8k_cot_zeroshot"), ndocs("ifeval")
            acc = lambda r, key, n: rate(round(r[key] * n), n) if key in r and n else "-"
            lines.append(f"| {row} | {acc(g, 'exact_match,flexible-extract', ng)} | {acc(g, 'exact_match,strict-match', ng)} | {acc(f, 'prompt_level_strict_acc,none', nf)} | {acc(f, 'prompt_level_loose_acc,none', nf)} | {ng} / {nf} |")
            rates[f"lmeval/{row}"] = dict(gsm8k=g, ifeval=f, n=dict(gsm8k=ng, ifeval=nf))
    # gates and data
    for name, title in [("mech.json", "HF hook mechanical check"), ("tv_pairs_manifest.json", "TV pairs"), ("render.json", "Rendering and windows")]:
        p = OUT / name
        if not p.exists(): continue
        d = json.loads(p.read_text())
        if name == "mech.json": lines += ["", f"## {title}", "", f"hooked layers: max |component - pin| {max(v['max_abs_dev_from_pin'] for v in d['hooked_layers'].values()):.4f}; layers before the window changed by at most {max(d['unhooked_first_layer_change'].values()):.2e}; passed {d['passed']}"]
        elif name == "tv_pairs_manifest.json": lines += ["", f"## {title}", "", f"counts per filter step: {d['counts']}", f"re-tokenisation failures: {len(d['retokenise_failed'])}", f"write regex: `{d['write_regex']}`"]
        else: lines += ["", f"## {title}", "", f"R window {d['R_window']}; Arditi layers {d['arditi_layers']}; off example ({d['off_example']['n']} tokens): `{d['off_example']['text']!r}`; on example ({d['on_example']['n']} tokens): `{d['on_example']['text']!r}`"]
    for p in sorted(OUT.glob("served_compare_*.json")):
        d = json.loads(p.read_text())
        lines += ["", f"## Served vs HF: {d['a']} vs {d['b']}: {d['tokens']} tokens, mean |diff| {d['mean_abs']:.5f} nats/token (signed {d['signed']:+.5f}, max {d['max_abs']:.3f}; per env {d['per_env']})" + (f"; floor {d['floor']:.5f}, tol {d['tol']:.5f}, passed {d['passed']}" if d.get("floor") else " (the floor)")]
    for env in ("refusal", "grader", "tv"):
        p = OUT / f"arditi_{env}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text())
        lines += ["", f"## Arditi candidates, {env}: base bypass {d['base_bypass']:+.3f}" + (f", base induce {d['base_induce']:+.3f}" if d["base_induce"] is not None else " (no induce filter)") + f"; {len(d['candidates'])} candidates, {d['passing']} pass; chosen: {d['chosen']}", "", "| position | layer | bypass | induce | KL | passes |", "|---|---|---|---|---|---|"]
        for r in sorted(d["candidates"], key=lambda r: r["bypass"])[:10]:
            ok = r["kl"] < d["kl_max"] and (d["base_induce"] is None or r["induce"] > d["base_induce"])
            lines.append(f"| {r['position']} | {r['layer']} | {r['bypass']:+.3f} | " + (f"{r['induce']:+.3f}" if r["induce"] is not None else "-") + f" | {r['kl']:.4f} | {ok} |")
    if (OUT / "cosines.md").exists(): lines += ["", "## Cosines and held-out AUROC", "", (OUT / "cosines.md").read_text()]
    # samples: 3 cheat and 3 clean grader answers per row and setting, 5 TV command sequences per row, at evenly spaced positions (refusal samples go to the private staging, not here)
    lines += ["", "## Samples (evenly spaced positions in each run, not hand-picked)", ""]
    for row in ROWS:
        for config in (OFF, ON):
            rs = newest_run([r for r in grader if mine(r) and row_of(r) == row and r["config_id"] == config and r["labels"]["prompts"] == "hack" and r["cfg"]["family"] == "all"])
            rs = [r for r in rs if r["labels"]["cheat"] is not None]
            for label in (True, False):
                v = [r for r in rs if r["labels"]["cheat"] == label]
                for k in [round(i * (len(v) - 1) / 2) for i in range(3)] if len(v) >= 3 else range(len(v)):
                    r = v[k]; turn = r["turns"][-1]
                    lines.append(f"- {row} / {config} / {'CHEAT' if label else 'clean'} / {r['labels']['family']}.{r['labels']['side']}: " + (f"reasoning: {turn['reasoning'][:300]!r} ... " if turn["reasoning"] else "") + f"answer: {turn['message'][:200]!r}")
        for config in (ON, OFF):
            rs = tsel(row, config, "verifier_bug", 4)
            for k in [round(i * (len(rs) - 1) / 4) for i in range(5)] if len(rs) >= 5 else range(len(rs)):
                r = rs[k]
                lines.append(f"- {row} / {config} / tv_bug mt4 / {r['rollout_id'].split('/')[-1]} tampered={r['labels'].get('tampered')} passed={r['labels'].get('passed')} end={r.get('end_reason')}: " + " || ".join(c[:120].replace("\n", "\\n") for turn in r["turns"] for c in turn.get("commands", [])))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    (OUT / "rates.json").write_text(json.dumps(rates, indent=1, default=str))
    print("\n".join(lines[:40])); print(f"... saved {OUT / 'table.md'} and rates.json")

if __name__ == "__main__": main()
