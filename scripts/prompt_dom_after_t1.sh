#!/bin/bash
# After the first tier-1 batch (sweep/mission_prompt_dom.md, box 2026-11-24): the TV Arditi search again in fp32 with TF32 matmuls (HF bf16 is unstable on the 4k-token TV
# completions, LOG 02:42-02:49), its capability check and the cosine table, then tier 1 for tvarditi, the refusal runs of refarditi and zrand, and tier 2 for each environment's
# own R and A rows plus base. Run from /root/ortho as root: bash scripts/prompt_dom_after_t1.sh > /root/prompt_dom_out/logs/after_t1.log 2>&1
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
bash scripts/prompt_dom_box.sh stop
# the bf16 tvarditi entry of check.json is kept under its own name so `check tvarditi` recomputes the fp32 selection's KL and MMLU
.venv/bin/python - <<'PY'
import json; from pathlib import Path
p = Path("acts/prompt_dom/check.json"); d = json.loads(p.read_text())
if "tvarditi" in d["rows"] and "tvarditi_bf16" not in d["rows"]: d["rows"]["tvarditi_bf16"] = d["rows"].pop("tvarditi"); p.write_text(json.dumps(d, indent=1)); print("check.json: tvarditi -> tvarditi_bf16")
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PD_DTYPE=fp32 PD_TF32=1 PD_TOK_BUDGET=6000 PD_LP_BUDGET=4000 .venv/bin/python prompt_dom_rh.py arditi tv || echo "AFTER_T1_FAILED arditi tv"
[ -f ~/hooks/tvarditi.safetensors ] && { .venv/bin/python prompt_dom_rh.py check tvarditi || echo "AFTER_T1_FAILED check tvarditi"; }
.venv/bin/python prompt_dom_rh.py cosines > /dev/null || echo "AFTER_T1_FAILED cosines"
echo "HF_DONE $(date -u +%T)"
[ -f ~/hooks/tvarditi.safetensors ] && bash scripts/prompt_dom_box.sh tier1 tvarditi
bash scripts/prompt_dom_box.sh refusal refarditi zrand
bash scripts/prompt_dom_box.sh tier2 base gpdom gpdomall gparditi tvpdom tvpdomall tvarditi
bash scripts/prompt_dom_box.sh stop
echo "AFTER_T1_DONE $(date -u +%T)"
