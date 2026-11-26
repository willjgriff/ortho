#!/bin/bash
# box G update 2 (sweep/mission_second_model_gemma_update_2.md): the cheap evals of the KL-matched controls, then tier 2 (GSM8K / IFEval) for base and the real rows. Root, from /root/ortho.
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH SM_MODEL=${SM_MODEL:-gemma4_31b} SM_SKIP_TV=1
LOGS=/root/second_model_out/logs
runs() { for run in "$@"; do ./run.py $run > $LOGS/run_$run.log 2>&1 || ./run.py $run > $LOGS/run_$run.retry.log 2>&1 || echo "RUN_FAILED $run"; echo "RUN_DONE $run $(date -u +%T)"; done; }
bash scripts/second_model_box.sh serve prandk_ref
runs prandk_ref_sm_ref_off_harmful prandk_ref_sm_ref_off_harmless prandk_ref_sm_ref_on_harmful prandk_ref_sm_ref_on_harmless
bash scripts/second_model_box.sh serve prandk_gp
runs prandk_gp_sm_g_off_main prandk_gp_sm_g_off_heldout prandk_gp_sm_ref_off_harmful prandk_gp_sm_ref_off_harmless prandk_gp_sm_ref_on_harmful prandk_gp_sm_ref_on_harmless
echo "CONTROLS_DONE $(date -u +%T)"
bash scripts/second_model_box.sh tier2 base refdom gpdom
bash scripts/second_model_box.sh stop
echo "FINISH_DONE $(date -u +%T)"
