#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/second_model_box.sh <stage> [args], the runs of sweep/mission_second_model.md (SM_MODEL picks the model tag, default gemma4_31b).
# Prints STAGE_DONE / RUN_DONE / RUN_FAILED / *_FAILED lines; tee it to a log under /root/second_model_out/logs/.
#   setup                the serving plugin (scripts/vllm_proj.py as a vLLM general plugin in /root/vllm-env), ~/serve_row.sh, row base's zero hook, and rhbench's copies (rhbench_sync + uv sync)
#   rhbench_sync         rhbench's readable copies of this clone and /root/rh-envs refreshed (no uv sync); run after every code change. The copies are data for root: never executed by root.
#   serve <row>          bash ~/serve_row.sh <row>
#   pilot                row base: the Grader runs (off main/clean/heldout, on main/clean) and refusal runs (off/on x harmful/harmless) as root, the TV runs (on bug mt4 200, on pos 64, off bug 100, off pos 64) as rhbench, concurrently
#   collect [k ...]      TV pair collection as rhbench: sm_tv_collect_c<k> (verifier_bug mt6, source_eval pool chunks of 200); default chunks 0 1 2
#   tier1 <row> ...      per row: serve it, then the Grader runs (off main/heldout, on main/heldout) and refusal runs as root and the TV runs (on bug, on pos; off bug if SM_TV_OFF=1) as rhbench, concurrently
#   refusal <row> ...    per row: serve it, then the four refusal runs as root
#   tier2 <row> ...      per row: serve it, then as rhbench <row>_sm_tv_on_bug_mt6 and as root GSM8K + IFEval through lm-eval (scripts/prompt_dom_lmeval.py)
#   pull_rhbench         copy rhbench's records and logs (terminal_verifier) into this clone's data/inspect and logs (data only)
#   stop                 stop vLLM
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
export SM_MODEL=${SM_MODEL:-gemma4_31b}
RB=/home/rhbench/ortho
LOGS=/root/second_model_out/logs; mkdir -p $LOGS
stop_vllm() { tmux kill-session -t vllm 2>/dev/null; for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done; sleep 5; }
runs() { for run in "$@"; do ./run.py $run > $LOGS/run_$run.log 2>&1 || ./run.py $run > $LOGS/run_$run.retry.log 2>&1 || echo "RUN_FAILED $run"; echo "RUN_DONE $run $(date -u +%T)"; done; }
rb_runs() { for run in "$@"; do rhbench-run bash -c "cd $RB && SM_MODEL=$SM_MODEL ./run.py $run" > $LOGS/rbrun_$run.log 2>&1 || rhbench-run bash -c "cd $RB && SM_MODEL=$SM_MODEL ./run.py $run" > $LOGS/rbrun_$run.retry.log 2>&1 || echo "RUN_FAILED $run (rhbench)"; echo "RUN_DONE $run (rhbench) $(date -u +%T)"; done; }
serve() { curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$1\"" && return 0; bash ~/serve_row.sh $1 || { echo "SERVE_FAILED $1"; return 1; }; echo "SERVING $1, hooked layers: $(grep -c 'vllm_proj: hook on' ~/vllm_$1.log) $(date -u +%T)"; }
pull() { for env in terminal_verifier; do mkdir -p data/inspect/$env logs/$env; [ -d $RB/data/inspect/$env ] && cp -n $RB/data/inspect/$env/*.jsonl data/inspect/$env/ 2>/dev/null; [ -d $RB/logs/$env ] && cp -n $RB/logs/$env/*.eval logs/$env/ 2>/dev/null; done; echo "pulled rhbench records $(date -u +%T)"; }
sync_rb() {
    rsync -a --delete --exclude .git --exclude .venv --exclude data --exclude acts --exclude logs --exclude figures --exclude __pycache__ --exclude '*.pyc' ~/ortho/ $RB/
    rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ /root/rh-envs/ /home/rhbench/rh-envs/
    sed -i 's#/root/rh-envs#/home/rhbench/rh-envs#g' $RB/pyproject.toml $RB/uv.lock
    mkdir -p $RB/data/inspect $RB/logs
    chown -R rhbench:rhbench $RB /home/rhbench/rh-envs
}
case $1 in
setup)
    mkdir -p ~/proj_plugin ~/hooks
    cp scripts/vllm_proj.py ~/proj_plugin/vllm_proj.py
    cp scripts/second_model_serve_row.sh ~/serve_row.sh
    printf '[project]\nname = "vllm-proj-plugin"\nversion = "0.1.0"\ndependencies = []\n\n[project.entry-points."vllm.general_plugins"]\nproj = "vllm_proj:register"\n\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n[tool.setuptools]\npy-modules = ["vllm_proj"]\n' > ~/proj_plugin/pyproject.toml
    uv pip install -q --python /root/vllm-env/bin/python -e ~/proj_plugin
    .venv/bin/python second_model.py hooks
    sync_rb
    rhbench-run bash -c "cd $RB && UV_HTTP_TIMEOUT=900 uv sync" > $LOGS/rhbench_uv_sync.log 2>&1 || { echo "SETUP_FAILED rhbench uv sync, see $LOGS/rhbench_uv_sync.log"; exit 1; }
    rhbench-run bash -c "cd $RB && SM_MODEL=$SM_MODEL ./run.py" > /dev/null 2>&1 || { echo "SETUP_FAILED rhbench run.py list"; exit 1; }
    echo "STAGE_DONE setup $(date -u +%T)";;
rhbench_sync) sync_rb; echo "STAGE_DONE rhbench_sync $(date -u +%T)";;
serve) serve $2;;
pilot)
    serve base || { echo "PILOT_FAILED serve base"; exit 1; }
    rb_runs sm_tv_on_bug sm_tv_on_pos sm_tv_off_bug sm_tv_off_pos &
    pid=$!
    runs sm_g_off_main sm_g_off_clean sm_g_off_heldout sm_ref_off_harmful sm_ref_off_harmless sm_g_on_main sm_g_on_clean sm_ref_on_harmful sm_ref_on_harmless
    wait $pid
    pull
    echo "STAGE_DONE pilot $(date -u +%T)";;
collect)
    shift
    serve base || { echo "COLLECT_FAILED serve base"; exit 1; }
    for k in ${*:-0 1 2}; do rb_runs sm_tv_collect_c$k; pull; echo "CHUNK_DONE $k $(date -u +%T)"; done
    echo "STAGE_DONE collect $(date -u +%T)";;
tier1)
    shift
    for row in "$@"; do
        serve $row || continue
        if [ "$row" = base ]; then g="sm_g_off_main sm_g_off_heldout sm_g_on_main sm_g_on_heldout sm_ref_off_harmful sm_ref_off_harmless sm_ref_on_harmful sm_ref_on_harmless"; tv="sm_tv_on_bug sm_tv_on_pos"; [ "${SM_TV_OFF:-0}" = 1 ] && tv="$tv sm_tv_off_bug sm_tv_off_pos"
        else g="${row}_sm_g_off_main ${row}_sm_g_off_heldout ${row}_sm_g_on_main ${row}_sm_g_on_heldout ${row}_sm_ref_off_harmful ${row}_sm_ref_off_harmless ${row}_sm_ref_on_harmful ${row}_sm_ref_on_harmless"; tv="${row}_sm_tv_on_bug ${row}_sm_tv_on_pos"; [ "${SM_TV_OFF:-0}" = 1 ] && tv="$tv ${row}_sm_tv_off_bug ${row}_sm_tv_off_pos"; fi
        [ "${SM_SKIP_GRADER:-0}" = 1 ] && g=$(echo $g | tr ' ' '\n' | grep -v _g_ | tr '\n' ' ')
        [ "${SM_SKIP_TV:-0}" = 1 ] && tv=""
        [ -n "$tv" ] && { rb_runs $tv & pid=$!; } || pid=""
        runs $g
        [ -n "$pid" ] && wait $pid
        pull
        echo "ROW_DONE tier1 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier1 $* $(date -u +%T)";;
refusal)
    shift
    for row in "$@"; do
        serve $row || continue
        if [ "$row" = base ]; then runs sm_ref_off_harmful sm_ref_off_harmless sm_ref_on_harmful sm_ref_on_harmless; else runs ${row}_sm_ref_off_harmful ${row}_sm_ref_off_harmless ${row}_sm_ref_on_harmful ${row}_sm_ref_on_harmless; fi
        echo "ROW_DONE refusal $row $(date -u +%T)"
    done
    echo "STAGE_DONE refusal $* $(date -u +%T)";;
tier2)
    shift
    for row in "$@"; do
        serve $row || continue
        if [ "$row" = base ]; then tv="sm_tv_on_bug_mt6"; else tv="${row}_sm_tv_on_bug_mt6"; fi
        if [ -x /root/lmeval/bin/python ]; then /root/lmeval/bin/python scripts/prompt_dom_lmeval.py cb_$row ${LMEVAL_LIMIT:-500} > $LOGS/lmeval_$row.log 2>&1 & lpid=$!; else lpid=""; echo "no /root/lmeval venv: GSM8K/IFEval not run for $row"; fi
        rb_runs $tv
        [ -n "$lpid" ] && { wait $lpid && echo "RUN_DONE lmeval $row $(date -u +%T)" || echo "RUN_FAILED lmeval $row (see $LOGS/lmeval_$row.log)"; }
        pull
        echo "ROW_DONE tier2 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier2 $* $(date -u +%T)";;
pull_rhbench) pull;;
stop) stop_vllm; echo "STAGE_DONE stop";;
*) sed -n 2,14p "$0"; exit 1;;
esac
