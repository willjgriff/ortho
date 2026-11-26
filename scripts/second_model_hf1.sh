#!/bin/bash
# Phase 2 of sweep/mission_second_model.md on the GPU box after the pilot, as root from /root/ortho: vLLM stopped, then the HF stages of second_model.py (SM_MODEL picks the tag):
# the hook's mechanical check, the refusal and Grader fits (R rows refdom / gpdom, controls prand / zrand), Arditi on refusal (refarditi), the capability checks of every row with a
# hook file, the HF references of the served-vs-HF gate (base and refdom), the cosine table; then base and refdom served through the hook path and scored against those references.
# TV stages (tvpairs, fit tv, arditi tv) run only when SM_TV=1 (the TV cell passed its gate and the collection ran). Prints *_DONE / *_FAILED lines; tee it to a log.
#   bash scripts/second_model_hf1.sh > /root/second_model_out/logs/hf1.log 2>&1
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
export SM_MODEL=${SM_MODEL:-gemma4_31b}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
bash scripts/second_model_box.sh stop
for _ in $(seq 60); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${used:-0}" -lt 5000 ] && break; sleep 2; done
echo "GPU memory used ${used} MiB $(date -u +%T)"
$PY second_model.py mech || echo "HF1_FAILED mech"
$PY second_model.py fit refusal grader || echo "HF1_FAILED fit"
if [ "${SM_TV:-0}" = 1 ]; then
    $PY second_model.py tvpairs || echo "HF1_FAILED tvpairs"
    $PY second_model.py fit tv || echo "HF1_FAILED fit tv"
fi
$PY second_model.py arditi refusal || echo "HF1_FAILED arditi refusal"
rows="base refdom gpdom prand refarditi zrand"; [ "${SM_TV:-0}" = 1 ] && rows="$rows tvpdom"
$PY second_model.py check $rows || echo "HF1_FAILED check"
$PY second_model.py served base refdom || echo "HF1_FAILED served"
$PY second_model.py cosines || echo "HF1_FAILED cosines"
echo "HF1_HF_DONE $(date -u +%T)"
# the served-vs-HF gate: base (the floor) and refdom through the hook path
bash scripts/second_model_box.sh serve base && { $PY scripts/second_model_served_check.py score cb_base base base && $PY scripts/second_model_served_check.py compare base hf_base; } || echo "HF1_FAILED served base"
bash scripts/second_model_box.sh serve refdom && { $PY scripts/second_model_served_check.py score cb_refdom refdom refdom && $PY scripts/second_model_served_check.py compare refdom hf_refdom && $PY scripts/second_model_served_check.py compare refdom hf_base; } || echo "HF1_FAILED served refdom"
echo "HF1_DONE $(date -u +%T)"
