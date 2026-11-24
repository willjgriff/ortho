"""Refreshes the "Results table" section of box_report/prompt_dom/SUMMARY.md from acts/prompt_dom/table.md (prompt_dom_table.py's output): the Grader, TV, MBPP, refusal and
capability tables. Run from /root/ortho after prompt_dom_table.py: python scripts/prompt_dom_summary.py"""
import re
from pathlib import Path

tab = Path("acts/prompt_dom/table.md").read_text()
def section(title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}.*?(?=^## |\Z)", tab, re.S | re.M)
    return m.group(0).strip() + "\n" if m else ""
grader = section("Grader (hack prompts, no system prompt)").split("### Grader per family")[0].strip() + "\n"
parts = [grader, section("TerminalVerifier (qwen3.8_sweep config, seed pool, 128 grids)"), section("MBPP impossible (qwen3.8_sweep sampling)"), section("Refusal (heretic's test prompts, keyword label"), section("Capability (HF, vLLM stopped)"), section("GSM8K and IFEval")]
p = Path("/workspace/llm-reward-hacking-ablation/box_report/prompt_dom/SUMMARY.md"); s = p.read_text()
a, b = s.index("## Results table"), s.index("## Cosine and AUROC tables")
body = "## Results table\nRows served on this box (records from 2026-11-23T20 on); `ethan:*` rows come from Ethan's records on his server and are context only. Full per-family tables, Arditi candidate tables and samples: `acts/prompt_dom/table.md` (uploaded under `prompt_dom/acts/`).\n\n" + "\n".join(x.replace("## ", "### ", 1) for x in parts if x.strip()) + "\n"
s = s[:a] + body + s[b:]
# samples: table.md's evenly spaced samples for base and the R / A rows (3 cheat + 3 clean Grader answers per reasoning setting, 5 TV command sequences per row)
samples = section("Samples (evenly spaced positions in each run, not hand-picked)").split("\n")[1:]
keep = ("base", "gpdom", "gpdomall", "gp10dom", "gparditi", "tvpdom", "tvpdomall", "tvarditi")
samples = [l for l in samples if l.startswith("- ") and l[2:].split(" / ")[0] in keep]
a, b = s.index("## Samples"), s.index("## Deviations from the mission")
body = "## Samples\nPicked at evenly spaced positions in each run (prompt_dom_table.py), not by hand: per row and reasoning setting 3 CHEAT and 3 clean Grader answers (`family.side`, reasoning and answer truncated), then 5 `verifier_bug` mt4 episodes as their command sequences (`||` separates commands). Every row's samples, the random controls included: `acts/prompt_dom/table.md`.\n\n" + "\n".join(samples) + "\n\n"
s = s[:a] + body + s[b:]
p.write_text(s); print(f"SUMMARY results table and samples refreshed ({len(samples)} sample lines)")
