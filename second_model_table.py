"""The result tables of sweep/mission_second_model.md for a second model (models.py), from the records this box wrote (data/inspect/{grader,terminal_verifier,refusal}: run.py's
<prefix><row>_<run> instances, condition none for base) and prompt_dom_rh.py's outputs (acts/second_model/<tag>/*.json): every row x every eval cell with Wilson intervals, answer
lengths and truncation, capability (check.json), cosines, the Arditi candidate tables, the TV pair counts, and evenly spaced samples. Grader and TV samples go to table.md (public);
refusal samples (harmful-request answers) to refusal_samples.md, for the private repo only. Run as PD_MODEL=<tag> ./prompt_dom_rh.py table."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from mechtools import wilson

RUN_KEYS = ("grader_off_clean", "grader_off_heldout", "grader_off", "grader_on_clean", "grader_on_heldout", "grader_on", "tv_bug_mt6", "tv_bug_off", "tv_bug", "tv_pos", "harmful_off", "harmless_off", "harmful_on", "harmless_on")
ROWS = ["base", "refdom", "gpdom", "tvpdom", "prand", "refarditi", "zrand", "gparditi", "tvarditi"]

def load(env: str) -> list[dict]:
    return [json.loads(line) for path in sorted((Path("data/inspect") / env).glob("*.jsonl")) for line in path.open()]

def rate(k: int, n: int) -> str:
    if not n: return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"

def row_of(r: dict, prefix: str) -> str | None:
    c = r["condition"]
    if c == "none": return "base"
    if not c.startswith(prefix): return None
    c = c[len(prefix):]
    for key in RUN_KEYS:
        if c.endswith("_" + key): return c[:-len(key) - 1]
    return None

def answer_chars(r: dict) -> int: return len(r["turns"][-1]["reasoning"] or "") + len(r["turns"][-1]["message"] or "")

def grader_cell(rs: list[dict]) -> dict:
    valid = [r for r in rs if r["labels"]["cheat"] is not None]
    empty = sum(1 for r in rs if not (r["turns"][0]["message"] or "").strip())
    return dict(n=len(rs), valid=len(valid), cheat=sum(r["labels"]["cheat"] for r in valid), cheat_rate=rate(sum(r["labels"]["cheat"] for r in valid), len(valid)), valid_rate=rate(len(valid), len(rs)), empty_answers=empty,
                median_chars=int(np.median([answer_chars(r) for r in rs])) if rs else None, logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def tv_cell(rs: list[dict]) -> dict:
    f = lambda k: sum(bool(r["labels"].get(k)) for r in rs)
    ends = {k: sum(r["end_reason"] == k for r in rs) for k in ("verifier_positive", "turn_limit", "truncated", "no_command", "interrupted")}
    return dict(n=len(rs), tampered=f("tampered"), passed=f("passed"), peeked=f("peeked"), cheated=sum(r["cheated"] for r in rs), tampered_rate=rate(f("tampered"), len(rs)), passed_rate=rate(f("passed"), len(rs)), peeked_rate=rate(f("peeked"), len(rs)),
                end_reasons=ends, median_turns=float(np.median([r["n_turns"] for r in rs])) if rs else None, logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def refusal_cell(rs: list[dict]) -> dict:
    valid = [r for r in rs if r["labels"]["refused"] is not None]
    return dict(n=len(rs), valid=len(valid), refused=sum(r["labels"]["refused"] for r in valid), refused_rate=rate(sum(r["labels"]["refused"] for r in valid), len(valid)), valid_rate=rate(len(valid), len(rs)),
                median_chars=int(np.median([answer_chars(r) for r in rs])) if rs else None, logs=sorted({r["rollout_id"].split("/")[0] for r in rs}))

def spaced(xs: list, k: int) -> list:
    return [xs[i] for i in np.linspace(0, len(xs) - 1, min(k, len(xs))).astype(int)] if xs else []

def clip(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + " […]"

def main(spec, out: Path) -> None:
    p = spec.served_prefix
    cfg = spec.configs
    grader, tv, refusal = load("grader"), load("terminal_verifier"), load("refusal")
    rates, lines = {}, [f"# {spec.tag} ({spec.hf_id} @ {spec.revision[:8]}): results tables (sweep/mission_second_model.md)", ""]
    # grader
    lines += ["## Grader: cheat rate among valid answers [Wilson 95%] (cheat/valid); valid rate; median completion chars (reasoning + answer)", "", "| row | off, main hack | off, held-out hack | off, main clean (control) | on, main hack | on, held-out hack | on, main clean (control) |", "|---|---|---|---|---|---|---|"]
    gcols = [("off", "all", "hack"), ("off", "heldout", "hack"), ("off", "all", "clean"), ("on", "all", "hack"), ("on", "heldout", "hack"), ("on", "all", "clean")]
    for row in ROWS:
        cells = []
        for setting, family, prompts in gcols:
            rs = [r for r in grader if row_of(r, p) == row and r["config_id"] == cfg[("grader", setting)] and r["labels"]["prompts"] == prompts and r["labels"]["sys"] == "none" and r["cfg"]["family"] == family]
            c = grader_cell(rs) if rs else None
            rates[f"grader/{row}/{setting}/{family}/{prompts}"] = c
            cells.append(f"{c['cheat_rate']}; valid {c['valid_rate']}; {c['median_chars']} ch" if c else "-")
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += ["", "### Grader per family (cheat/valid), hack prompts, this box", ""]
    fams = sorted({r["labels"]["family"] for r in grader})
    lines += ["| row | setting | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
    for row in ROWS:
        for setting in ("off", "on"):
            rs = [r for r in grader if row_of(r, p) == row and r["config_id"] == cfg[("grader", setting)] and r["labels"]["prompts"] == "hack"]
            if not rs: continue
            cells = []
            for f in fams:
                v = [r for r in rs if r["labels"]["family"] == f and r["labels"]["cheat"] is not None]
                cells.append(f"{sum(r['labels']['cheat'] for r in v)}/{len(v)}" if v else "-")
            lines.append(f"| {row} | {setting} | " + " | ".join(cells) + " |")
    # TV
    lines += ["", "## TerminalVerifier (seed pool): tampered, passed, peeked [Wilson 95%]; end reasons (verifier_positive / turn_limit / truncated / no_command / interrupted)", "", "| row | bug mt4 on: tampered | passed | peeked | end reasons | possible mt6 on: passed | bug mt4 off: tampered | passed | bug mt6 on: tampered | passed |", "|---|---|---|---|---|---|---|---|---|---|"]
    def tsel(row, setting, mode, mt):
        if (("terminal_verifier", setting)) not in cfg: return []
        return [r for r in tv if row_of(r, p) == row and r["config_id"] == cfg[("terminal_verifier", setting)] and r["mode"] == mode and r["max_turns"] == mt and r.get("pool", "seed") == "seed"]
    for row in ROWS:
        cells = []
        bug = tsel(row, "on", "verifier_bug", 4); c = tv_cell(bug) if bug else None; rates[f"tv/{row}/on/verifier_bug/4"] = c
        cells += [c["tampered_rate"], c["passed_rate"], c["peeked_rate"], "/".join(str(c["end_reasons"][k]) for k in ("verifier_positive", "turn_limit", "truncated", "no_command", "interrupted"))] if c else ["-"] * 4
        pos = tsel(row, "on", "possible", 6); c = tv_cell(pos) if pos else None; rates[f"tv/{row}/on/possible/6"] = c
        cells.append(c["passed_rate"] if c else "-")
        off = tsel(row, "off", "verifier_bug", 4); c = tv_cell(off) if off else None; rates[f"tv/{row}/off/verifier_bug/4"] = c
        cells += [c["tampered_rate"], c["passed_rate"]] if c else ["-", "-"]
        mt6 = tsel(row, "on", "verifier_bug", 6); c = tv_cell(mt6) if mt6 else None; rates[f"tv/{row}/on/verifier_bug/6"] = c
        cells += [c["tampered_rate"], c["passed_rate"]] if c else ["-", "-"]
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # refusal
    lines += ["", "## Refusal (heretic's test prompts, keyword label; the judge's columns are added from the judge report): refused among valid answers [Wilson 95%]; median chars", "", "| row | harmful, off | harmless, off | harmful, on | harmless, on |", "|---|---|---|---|---|"]
    for row in ROWS:
        cells = []
        for prompts, setting in (("harmful", "off"), ("harmless", "off"), ("harmful", "on"), ("harmless", "on")):
            rs = [r for r in refusal if row_of(r, p) == row and r["config_id"] == cfg[("refusal", setting)] and r["labels"]["prompts"] == prompts and r["labels"]["split"] == "test"]
            c = refusal_cell(rs) if rs else None; rates[f"refusal/{row}/{setting}/{prompts}"] = c
            cells.append(f"{c['refused_rate']}; valid {c['valid']}/{c['n']}; {c['median_chars']} ch" if c else "-")
        if any(x != "-" for x in cells): lines.append(f"| {row} | " + " | ".join(cells) + " |")
    # capability
    check = json.loads((out / "check.json").read_text()) if (out / "check.json").exists() else None
    if check:
        lines += ["", f"## Capability (HF, vLLM stopped): MMLU on every {check['mmlu_every']}th test question (n = {check['mmlu_n']}), first-token KL from base on heretic's 100 harmless test prompts", "", "| row | MMLU | vs base [95%] | first-token KL |", "|---|---|---|---|"]
        for row, c in check["rows"].items():
            vb = c["mmlu"].get("vs_base")
            lines.append(f"| {row} | {100 * c['mmlu']['accuracy']:.2f} | " + (f"{vb['points']:+.2f} [{vb['low']:+.2f}, {vb['high']:+.2f}]" if vb else "-") + f" | {c['first_token_kl_harmless']:.4f} |")
        rates["capability"] = {row: dict(mmlu=c["mmlu"]["accuracy"], vs_base=c["mmlu"].get("vs_base"), kl=c["first_token_kl_harmless"]) for row, c in check["rows"].items()}
    # served check, mechanical check, pairs, arditi, cosines
    for name, title in (("served_floor", "Served-vs-HF floor (base)"),):
        if (out / f"{name}.json").exists(): lines += ["", f"## {title}", "", "```", (out / f"{name}.json").read_text().strip(), "```"]
    cmp = sorted(out.glob("served_compare_*.json"))
    if cmp:
        lines += ["", "## Served-vs-HF check per row (mean |diff| nats/token over the 20 stored base answers; tol = 2 x floor)", "", "| row | tokens | mean abs | signed | max | % > 0.5 | per env | passed |", "|---|---|---|---|---|---|---|---|"]
        for f in cmp:
            c = json.loads(f.read_text()); lines.append(f"| {c['row']} | {c['tokens']} | {c['mean_abs']:.5f} | {c['signed']:+.5f} | {c['max_abs']:.2f} | {100 * c['frac_over_0_5']:.2f} | " + ", ".join(f"{k} {v:.4f}" for k, v in c["per_env"].items()) + f" | {c['passed']} |")
    mech = sorted(out.glob("mech_*.json"))
    if mech:
        lines += ["", "## Mechanical check (HF, 5 harmless prompts): max |h.u_L - c_L| after the hook at the hooked layers; max change at the layers before the window", ""]
        for f in mech:
            c = json.loads(f.read_text()); lines.append(f"- {c['row']}: hooked layers {max(c['hooked_layers_max_abs_component_minus_pin'].values()):.4f} (before the hook {max(c['base_max_abs_component_minus_pin'].values()):.2f}); layers before the window {max(c['layers_before_window_max_abs_change'].values()) if c['layers_before_window_max_abs_change'] else 0:.4f}")
    if (out / "tv_pairs_manifest.json").exists():
        m = json.loads((out / "tv_pairs_manifest.json").read_text()); lines += ["", "## TV pairs: counts per filter step", "", "```", json.dumps(m["counts"]), "```"]
    for env in ("refusal", "grader", "tv"):
        f = out / f"arditi_{env}.json"
        if not f.exists(): continue
        a = json.loads(f.read_text())
        lines += ["", f"## Arditi candidates, {env} (top 10 by bypass; base bypass {a['base_bypass']:+.3f}" + (f", base induce {a['base_induce']:+.3f}" if a.get("base_induce") is not None else "") + f"; {a['passing']} of {len(a['candidates'])} pass KL < {a['kl_max']}" + ("" if env == "tv" else " and induce > base") + f"); chosen {a['chosen']}", "", "| position | layer | bypass | induce | KL | passes |", "|---|---|---|---|---|---|"]
        for r in sorted(a["candidates"], key=lambda r: r["bypass"])[:10]:
            ok = r["kl"] < a["kl_max"] and (env == "tv" or (r["induce"] is not None and r["induce"] > a["base_induce"]))
            lines.append(f"| {r['position']} | {r['layer']} | {r['bypass']:+.3f} | {'-' if r['induce'] is None else f'{r[chr(105)+chr(110)+chr(100)+chr(117)+chr(99)+chr(101)]:+.3f}'} | {r['kl']:.4f} | {ok} |")
    if (out / "cosines.md").exists(): lines += ["", "## Cosines and held-out AUROC", "", (out / "cosines.md").read_text().strip()]
    for env in ("refusal", "grader", "tv"):
        f = Path("data/vectors") / spec.vectors_dir / f"pd_{ {'refusal': 'refdom', 'grader': 'gpdom', 'tv': 'tvpdom'}[env] }.json"
        if f.exists():
            meta = json.loads(f.read_text())
            if "val_auroc" in meta: lines.append(f"- {env} R row: validation AUROC along the row: " + ", ".join(f"L{L} {meta['val_auroc'][str(L)]:.3f}" for L in spec.step if str(L) in meta["val_auroc"]) + (f"; random rows: " + ", ".join(f"L{L} {meta['val_auroc_random'][str(L)]:.3f}" for L in spec.step if str(L) in meta.get("val_auroc_random", {})) if "val_auroc_random" in meta else ""))
    # samples (public: grader, tv)
    lines += ["", "## Samples (evenly spaced positions in each run, not hand-picked)", ""]
    for row in ROWS:
        for setting in ("off", "on"):
            rs = [r for r in grader if row_of(r, p) == row and r["config_id"] == cfg[("grader", setting)] and r["labels"]["prompts"] == "hack" and r["cfg"]["family"] == "all"]
            if not rs: continue
            cheat = [r for r in rs if r["labels"]["cheat"]]; clean = [r for r in rs if r["labels"]["cheat"] is False]
            for tag, group in (("CHEAT", spaced(cheat, 3)), ("clean", spaced(clean, 3))):
                for r in group:
                    tn = r["turns"][0]
                    lines.append(f"- {row} / grader {setting} / {tag} / {r['labels']['family']}.{r['labels']['side']}: reasoning «{clip(tn['reasoning'], 300)}» answer «{clip(tn['message'], 200)}»")
        bug = tsel(row, "on", "verifier_bug", 4)
        for r in spaced(bug, 5):
            cmds = " || ".join(clip(t["commands"][0], 90) if t["commands"] else "(no command)" for t in r["turns"])
            lines.append(f"- {row} / TV bug mt4 on / {r['rollout_id'].split('/')[1]} / tampered {r['labels']['tampered']} passed {r['labels']['passed']} peeked {r['labels']['peeked']} end {r['end_reason']}: {cmds}")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "rates.json").write_text(json.dumps(rates, indent=1, default=str))
    # refusal samples: private
    priv = [f"# {spec.tag}: refusal samples (harmful test prompts; private repo only)", ""]
    for row in ROWS:
        for setting in ("off", "on"):
            rs = [r for r in refusal if row_of(r, p) == row and r["config_id"] == cfg[("refusal", setting)] and r["labels"]["prompts"] == "harmful" and r["labels"]["split"] == "test"]
            for r in spaced(rs, 3):
                tn = r["turns"][0]
                priv.append(f"- {row} / {setting} / prompt {r['labels']['index']} / keyword refused {r['labels']['refused']}: request «{clip(r['items'][1]['content'] if len(r['items']) > 1 else r['items'][0]['content'], 150)}» reasoning «{clip(tn['reasoning'], 200)}» answer «{clip(tn['message'], 300)}»")
    (out / "refusal_samples.md").write_text("\n".join(priv) + "\n")
    print("\n".join(l for l in lines if not l.startswith("- ")))
    print(f"saved {out / 'table.md'}, rates.json, refusal_samples.md (private)")

if __name__ == "__main__":
    from models import current
    spec = current(); main(spec, Path("acts/second_model") / spec.tag)
