#!/bin/bash
# sweep/mission_gemma_envs.md Part B, as root from /root/ortho: bash scripts/gemma_envs_partb.sh <stage> [args]. Logs under /root/gemma_envs_out/logs/.
#   fits <unit> ...          vLLM stopped; second_model.py gefit <unit> ... langdom covrand; first-token KL of every env direction and gpdom at s 0.25 / 0.5 / 1, langdom / covrand at
#                            s 0.25 0.5 1 2 (the control curves); cosines between every direction (gecos)
#   dose <dir> <run> <kind>  the working dose: for s in 0.25 0.5 1, serve <dir> scaled by s as <dir>_sXXX and run its own env's dose check (<run>: a gemma_envs.py run as rhbench,
#                            kind env; or the Grader held-out check, kind grader); prints DOSE lines (the decision is made by gemma_envs_dose.py)
#   serve <row>              serve a row (hook ~/hooks/<row>.safetensors)
cd ~/ortho
export SM_MODEL=gemma4_31b PATH=$HOME/.local/bin:$PATH
LOGS=/root/gemma_envs_out/logs; mkdir -p $LOGS
stop_vllm() { tmux kill-session -t vllm 2>/dev/null; for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done; sleep 5; }
case $1 in
fits)
    shift
    stop_vllm
    .venv/bin/python second_model.py gefit "$@" langdom covrand > $LOGS/hf_fits.log 2>&1 || { echo "FITS_FAILED (see $LOGS/hf_fits.log)"; exit 1; }
    .venv/bin/python second_model.py gekl $(for u in "$@"; do printf '%sdom:0.25,0.5,1 ' $u; done) gpdom:0.25,0.5,1 refdom:1 langdom:0.25,0.5,1,2 covrand:0.25,0.5,1,2 > $LOGS/hf_kl.log 2>&1
    .venv/bin/python second_model.py gecos $(for u in "$@"; do printf '%sdom ' $u; done) gpdom refdom langdom covrand > $LOGS/hf_cos.log 2>&1
    echo "STAGE_DONE fits $(date -u +%T)";;
dose)
    dir=$2; run=$3
    for s in 0.25 0.5 1; do
        name=${dir}_s$(echo $s | tr -d .)
        .venv/bin/python gemma_envs/scale_hook.py $dir $s $name >> $LOGS/dose_$dir.log 2>&1
        bash scripts/gemma_envs_box.sh serve $name >> $LOGS/dose_$dir.log 2>&1 || { echo "SERVE_FAILED $name"; continue; }
        if [ "$run" = grader ]; then GE_ROWS=$name ./run.py ${name}_ge_g_dose > $LOGS/run_${name}_ge_g_dose.log 2>&1; out=$(.venv/bin/python gemma_envs_dose.py $name grader 2>/dev/null | grep DOSE)
        else bash scripts/gemma_envs_box.sh rb $name $run; bash scripts/gemma_envs_box.sh pull > /dev/null; out=$(.venv/bin/python gemma_envs_dose.py $name env $run 2>/dev/null | grep DOSE); fi
        echo "$out $(date -u +%T)"
        echo "$out" | grep -q "works True" && { echo "WORKING_DOSE $dir $s $(date -u +%T)"; exit 0; }
    done
    echo "NO_WORKING_DOSE $dir (kept at s = 1) $(date -u +%T)";;
grid)
    # grid <row> [flags]: GE_SKIP_AGENTIC=1 skips data_entry / overclaim, GE_SKIP_GSMON=1 skips GSM8K-on, GE_SKIP_GRADER=1 skips the Grader runs
    row=$2
    bash scripts/gemma_envs_box.sh serve $row >> $LOGS/grid_$row.log 2>&1 || { echo "SERVE_FAILED $row"; exit 1; }
    echo "GRID_START $row $(date -u +%T)"
    pids=""
    g() { [ "${GE_OWN:-}" = "$1" ] && echo ${1}_gridrest || echo ${1}_grid; }   # GE_OWN=<unit>: the row's own env adds only the ids its dose check did not cover
    bash scripts/gemma_envs_box.sh rb $row $(g mbpp) $(g nishi_rs) nishi_sycophancy_fact_grid nishi_sycophancy_opinion_nlp_grid nishi_sycophancy_opinion_political_grid nishi_world_affecting_approval_grid nishi_world_affecting_reward_grid toolfail_grid & pids="$pids $!"
    if [ "${GE_SKIP_AGENTIC:-0}" != 1 ] || [ "${GE_OWN:-}" = dataentry ]; then bash scripts/gemma_envs_box.sh rb $row $(g dataentry) & pids="$pids $!"; fi
    if [ "${GE_SKIP_AGENTIC:-0}" != 1 ] || [ "${GE_OWN:-}" = overclaim ]; then bash scripts/gemma_envs_box.sh rb $row $(g overclaim) & pids="$pids $!"; fi
    [ "${GE_SKIP_GRADER:-0}" != 1 ] && { GE_ROWS=$row ./run.py ${row}_ge_g_on_main ${row}_ge_g_on_heldout > $LOGS/run_${row}_ge_grader.log 2>&1 || echo "RUN_FAILED grader $row"; } &
    pids="$pids $!"
    LMEVAL_OUT=acts/gemma_envs/lmeval /root/lmeval/bin/python scripts/prompt_dom_lmeval.py cb_$row 500 > $LOGS/lmeval_$row.log 2>&1 & pids="$pids $!"
    [ "${GE_SKIP_GSMON:-0}" != 1 ] && { .venv/bin/python gemma_envs/gsm8k_on.py cb_$row 250 > $LOGS/gsm8k_on_$row.log 2>&1 || echo "RUN_FAILED gsm8k_on $row"; } &
    pids="$pids $!"
    wait $pids
    bash scripts/gemma_envs_box.sh pull > /dev/null
    echo "ROW_DONE grid $row $(date -u +%T)";;
serve) bash scripts/gemma_envs_box.sh serve $2;;
*) sed -n 2,8p "$0"; exit 1;;
esac
