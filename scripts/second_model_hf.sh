#!/bin/bash
# On the GPU box, as root from /root/ortho, vLLM stopped: PD_MODEL=<tag> bash scripts/second_model_hf.sh <stage>, the HF half of sweep/mission_second_model.md Phase 2 (prompt_dom_rh.py stages).
#   fits        padcheck; servedref base; fit refusal grader [tv when acts/.../tv_pairs.pt exists]; mech prand; servedref prand; cosines
#   arditi      arditi refusal; arditi grader when GRADER_A=1 (refarditi passed its judge check); check on every row with a hook file; cosines
#   check       check (first-token KL + MMLU) on the rows given in ROWS (default: every hook)
# Prints HF_STAGE_DONE / HF_FAILED lines; tee to a log. PD_DTYPE=fp32 PD_TF32=1 for the fp32 reruns if HF bf16 proves unstable on long TV prompts.
set -u
cd ~/ortho
export PD_MODEL=${PD_MODEL:?set PD_MODEL}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=acts/second_model/$PD_MODEL
curl -sf -m 2 localhost:8000/v1/models > /dev/null && { echo "HF_FAILED vLLM is still serving; stop it first"; exit 1; }
run() { echo "--- $* $(date -u +%T)"; $PY prompt_dom_rh.py "$@" || echo "HF_FAILED $*"; }
case $1 in
fits)
    run padcheck
    run servedref base
    envs="refusal grader"; [ -f $OUT/tv_pairs.pt ] && envs="$envs tv"
    run fit $envs
    [ -f ~/hooks/prand.safetensors ] && { run mech prand; run servedref prand; }
    run cosines
    echo "HF_STAGE_DONE fits $(date -u +%T)";;
arditi)
    run arditi refusal
    [ "${GRADER_A:-0}" = 1 ] && run arditi grader
    [ "${TV_A:-0}" = 1 ] && run arditi tv
    run check ${ROWS:-}
    run cosines
    echo "HF_STAGE_DONE arditi $(date -u +%T)";;
check) run check ${ROWS:-}; echo "HF_STAGE_DONE check $(date -u +%T)";;
*) sed -n 2,7p "$0"; exit 1;;
esac
