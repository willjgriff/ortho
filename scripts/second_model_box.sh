#!/bin/bash
# On the GPU box, as root from /root/ortho: PD_MODEL=<tag> bash scripts/second_model_box.sh <stage> [args], the runs of sweep/mission_second_model.md. Prints STAGE_DONE / RUN_DONE / RUN_FAILED lines; tee it to a log.
#   rhbench_sync          refresh the unprivileged user's code copies (/home/rhbench/{ortho,rh-envs}; no .git, .venv, data, acts, logs) after an edit; `rhbench_setup` also runs its uv sync
#   serve <row>           bash scripts/serve_row.sh <row> (MAX_NUM_SEQS, EAGER, EXTRA knobs)
#   base                  the pilot / base runs: refusal x4 and grader x6 as root, tv_bug / tv_pos / tv_bug_off as rhbench, concurrently
#   collect <chunk> ...   the on-policy TV collection chunks (collect_a ...) as rhbench
#   tier1 <row> ...       per row: serve, then grader (off hack/heldout/clean, on hack/heldout/clean) as root and tv_bug / tv_pos (+ tv_bug_off when TV_OFF=1) as rhbench, concurrently; then pull
#   refusal <row> ...     per row: serve, then harmful/harmless x off/on as root
#   tier2 <row> ...       per row: serve, then tv_bug_mt6 as rhbench and GSM8K / IFEval (scripts/prompt_dom_lmeval.py, /root/lmeval) as root
#   pull                  copy rhbench's records and logs (terminal_verifier) into this clone's data/inspect and logs (data only)
#   stop                  stop vLLM
# Row runs are skipped per cell by CELLS (space-separated list of grader_off grader_on tv_on tv_off refusal_off refusal_on; default all): the pilot's gates decide which cells run.
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
export PD_MODEL=${PD_MODEL:?set PD_MODEL}
P=$(.venv/bin/python -c "from models import current; print(current().served_prefix)" 2>/dev/null)
RB=/home/rhbench/ortho
LOGS=/root/second_model_out/logs; mkdir -p $LOGS
CELLS=${CELLS:-grader_off grader_on tv_on tv_off refusal_off refusal_on}
has() { [[ " $CELLS " == *" $1 "* ]]; }
stop_vllm() { tmux kill-session -t vllm 2>/dev/null; for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done; sleep 5; }
runs() { for run in "$@"; do ./run.py $run > $LOGS/run_$run.log 2>&1 || ./run.py $run > $LOGS/run_$run.retry.log 2>&1 || echo "RUN_FAILED $run"; echo "RUN_DONE $run $(date -u +%T)"; done; }
rb_runs() { for run in "$@"; do rhbench-run bash -c "cd $RB && PD_MODEL=$PD_MODEL ./run.py $run" > $LOGS/rbrun_$run.log 2>&1 || rhbench-run bash -c "cd $RB && PD_MODEL=$PD_MODEL ./run.py $run" > $LOGS/rbrun_$run.retry.log 2>&1 || echo "RUN_FAILED $run (rhbench)"; echo "RUN_DONE $run (rhbench) $(date -u +%T)"; done; }
serve() { curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$1\"" && return 0; bash scripts/serve_row.sh $1 || { echo "SERVE_FAILED $1"; return 1; }; echo "SERVING $1 $(date -u +%T)"; }
pull() { for env in terminal_verifier; do mkdir -p data/inspect/$env logs/$env; [ -d $RB/data/inspect/$env ] && cp -n $RB/data/inspect/$env/*.jsonl data/inspect/$env/ 2>/dev/null; [ -d $RB/logs/$env ] && cp -n $RB/logs/$env/*.eval logs/$env/ 2>/dev/null; done; echo "pulled rhbench records $(date -u +%T)"; }
sync_rb() { rsync -a --delete --exclude .git --exclude .venv --exclude data --exclude acts --exclude logs --exclude figures --exclude __pycache__ --exclude '*.pyc' ~/ortho/ $RB/; rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ ~/rh-envs/ /home/rhbench/rh-envs/; mkdir -p $RB/data/inspect $RB/logs; chown -R rhbench:rhbench $RB /home/rhbench/rh-envs; }
grader_runs() {   # hack prompts on the main and held-out families for every row; the clean-prompt control only for base (the pilot's gate), as the mission's Phase 3 lists
    local r=$1 out=""
    has grader_off && { out="$out ${P}${r}_grader_off ${P}${r}_grader_off_heldout"; [ "$r" = base ] && out="$out ${P}${r}_grader_off_clean"; }
    has grader_on && { out="$out ${P}${r}_grader_on ${P}${r}_grader_on_heldout"; [ "$r" = base ] && out="$out ${P}${r}_grader_on_clean"; }
    echo $out; }
tv_runs() { local r=$1 out=""; has tv_on && out="$out ${P}${r}_tv_bug ${P}${r}_tv_pos"; has tv_off && out="$out ${P}${r}_tv_bug_off"; echo $out; }
refusal_runs() { local r=$1 out=""; has refusal_off && out="$out ${P}${r}_harmful_off ${P}${r}_harmless_off"; has refusal_on && out="$out ${P}${r}_harmful_on ${P}${r}_harmless_on"; echo $out; }
case $1 in
rhbench_sync) sync_rb; echo "STAGE_DONE rhbench_sync $(date -u +%T)";;
rhbench_setup) sync_rb; rhbench-run bash -c "cd $RB && UV_HTTP_TIMEOUT=900 uv sync" > ~/rhbench_uv_sync.log 2>&1 || { echo "RHBENCH_SETUP_FAILED uv sync"; exit 1; }; echo "STAGE_DONE rhbench_setup $(date -u +%T)";;
serve) serve $2;;
base)
    serve base || exit 1
    rb_runs $(tv_runs base) & pid=$!
    runs $(refusal_runs base) $(grader_runs base)
    wait $pid; pull; echo "STAGE_DONE base $(date -u +%T)";;
collect) shift; serve base || exit 1; rb_runs $(for c in "$@"; do echo ${P}collect_$c; done); pull; echo "STAGE_DONE collect $* $(date -u +%T)";;
tier1)
    shift
    for row in "$@"; do
        serve $row || continue
        tv=$(tv_runs $row); g=$(grader_runs $row)
        [ -n "$tv" ] && { rb_runs $tv & pid=$!; } || pid=""
        [ -n "$g" ] && runs $g
        [ -n "$pid" ] && wait $pid
        pull; echo "ROW_DONE tier1 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier1 $* $(date -u +%T)";;
refusal) shift; for row in "$@"; do serve $row || continue; runs $(refusal_runs $row); echo "ROW_DONE refusal $row $(date -u +%T)"; done; echo "STAGE_DONE refusal $* $(date -u +%T)";;
tier2)
    shift
    for row in "$@"; do
        serve $row || continue
        [ -x /root/lmeval/bin/python ] && { PD_LMEVAL_OUT=acts/second_model/$PD_MODEL/lmeval /root/lmeval/bin/python scripts/prompt_dom_lmeval.py cb_$row ${LMEVAL_LIMIT:-500} > $LOGS/lmeval_$row.log 2>&1 & lpid=$!; } || lpid=""
        has tv_on && rb_runs ${P}${row}_tv_bug_mt6
        [ -n "$lpid" ] && { wait $lpid && echo "RUN_DONE lmeval $row $(date -u +%T)" || echo "RUN_FAILED lmeval $row"; }
        pull; echo "ROW_DONE tier2 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier2 $* $(date -u +%T)";;
pull) pull;;
stop) stop_vllm; echo "STAGE_DONE stop";;
*) sed -n 2,14p "$0"; exit 1;;
esac
