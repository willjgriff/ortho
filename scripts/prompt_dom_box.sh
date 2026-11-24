#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/prompt_dom_box.sh <stage> [args], the runs of sweep/mission_prompt_dom.md. Prints STAGE_DONE / RUN_DONE / RUN_FAILED / *_FAILED lines; tee it to a log.
#   rhbench_setup        a readable copy of this clone (no .git, .venv, data, acts, logs) at /home/rhbench/ortho for the unprivileged user that runs the TV and MBPP evals (model-written
#                        commands run on the host under sandbox=local), and its own venv (uv sync as rhbench). Rerun after code changes (the copy is data for root: never executed by root).
#   serve <row>          bash ~/serve_row.sh <row> (MAX_NUM_SEQS, MTP, PLAIN env knobs, see scripts/kto_serve_row.sh)
#   tier1 <row> ...      per row: serve it, then the Grader runs as root (<row>_nothink, _nothink_heldout, _main, _heldout) and the TV runs as rhbench (<row>_tv_bug, <row>_tv_pos) concurrently;
#                        row base runs nothink38, nothink38_heldout, main38, heldout38 and pdbase_tv_bug, pdbase_tv_pos. Then pull_rhbench.
#   refusal <row> ...    per row: serve it, then the four refusal test runs (<row>_harmful, _harmless, _harmful_low, _harmless_low) as root
#   tier2 <row> ...      per row: serve it, then as rhbench <row>_tv_bug_mt6, <row>_tv_corr and <row>_mbpp (pdbase_* for base)
#   pull_rhbench         copy rhbench's records and logs (terminal_verifier, mbpp) into this clone's data/inspect and logs (data only)
#   stop                 stop vLLM
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
RB=/home/rhbench/ortho
stop_vllm() { tmux kill-session -t vllm 2>/dev/null; for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done; sleep 5; }
runs() { for run in "$@"; do ./run.py $run > ~/run_$run.log 2>&1 || ./run.py $run > ~/run_$run.retry.log 2>&1 || echo "RUN_FAILED $run"; echo "RUN_DONE $run $(date -u +%T)"; done; }
rb_runs() { for run in "$@"; do rhbench-run bash -c "cd $RB && ./run.py $run" > ~/rbrun_$run.log 2>&1 || rhbench-run bash -c "cd $RB && ./run.py $run" > ~/rbrun_$run.retry.log 2>&1 || echo "RUN_FAILED $run (rhbench)"; echo "RUN_DONE $run (rhbench) $(date -u +%T)"; done; }
serve() { curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$1\"" && return 0; bash ~/serve_row.sh $1 || { echo "SERVE_FAILED $1"; return 1; }; echo "SERVING $1, hooked layers: $(grep -c 'vllm_proj: hook on' ~/vllm_$1.log) $(date -u +%T)"; }
pull() { for env in terminal_verifier mbpp; do mkdir -p data/inspect/$env logs/$env; [ -d $RB/data/inspect/$env ] && cp -n $RB/data/inspect/$env/*.jsonl data/inspect/$env/ 2>/dev/null; [ -d $RB/logs/$env ] && cp -n $RB/logs/$env/*.eval logs/$env/ 2>/dev/null; done; echo "pulled rhbench records $(date -u +%T)"; }
case $1 in
rhbench_setup)
    rsync -a --delete --exclude .git --exclude .venv --exclude data --exclude acts --exclude logs --exclude figures --exclude __pycache__ --exclude '*.pyc' ~/ortho/ $RB/
    mkdir -p $RB/data/inspect $RB/logs
    chown -R rhbench:rhbench $RB
    rhbench-run bash -c "cd $RB && UV_HTTP_TIMEOUT=900 uv sync" > ~/rhbench_uv_sync.log 2>&1 || { echo "RHBENCH_SETUP_FAILED uv sync, see ~/rhbench_uv_sync.log"; exit 1; }
    rhbench-run bash -c "cd $RB && ./run.py" > /dev/null 2>&1 || { echo "RHBENCH_SETUP_FAILED run.py list"; exit 1; }
    echo "STAGE_DONE rhbench_setup $(date -u +%T)";;
rhbench_sync)   # the code copy refreshed after an edit (no uv sync)
    rsync -a --delete --exclude .git --exclude .venv --exclude data --exclude acts --exclude logs --exclude figures --exclude __pycache__ --exclude '*.pyc' ~/ortho/ $RB/
    chown -R rhbench:rhbench $RB; echo "STAGE_DONE rhbench_sync $(date -u +%T)";;
serve) serve $2;;
tier1)
    shift
    for row in "$@"; do
        serve $row || continue
        if [ "$row" = base ]; then g="nothink38 nothink38_heldout main38 heldout38"; tv="pdbase_tv_bug pdbase_tv_pos"; else g="${row}_nothink ${row}_nothink_heldout ${row}_main ${row}_heldout"; tv="${row}_tv_bug ${row}_tv_pos"; fi
        rb_runs $tv &
        pid=$!
        runs $g
        wait $pid
        pull
        echo "ROW_DONE tier1 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier1 $* $(date -u +%T)";;
refusal)
    shift
    for row in "$@"; do
        serve $row || continue
        runs ${row}_harmful ${row}_harmless ${row}_harmful_low ${row}_harmless_low
        echo "ROW_DONE refusal $row $(date -u +%T)"
    done
    echo "STAGE_DONE refusal $* $(date -u +%T)";;
tier2)
    shift
    for row in "$@"; do
        serve $row || continue
        if [ "$row" = base ]; then tv="pdbase_tv_bug_mt6 pdbase_tv_corr pdbase_mbpp"; else tv="${row}_tv_bug_mt6 ${row}_tv_corr ${row}_mbpp"; fi
        /root/lmeval/bin/python scripts/prompt_dom_lmeval.py cb_$row ${LMEVAL_LIMIT:-500} > ~/lmeval_$row.log 2>&1 &   # GSM8K (cot zero-shot) + IFEval as root (regex scoring, no model output executed), concurrent with the rhbench runs
        lpid=$!
        rb_runs $tv
        wait $lpid && echo "RUN_DONE lmeval $row $(date -u +%T)" || echo "RUN_FAILED lmeval $row (see ~/lmeval_$row.log)"
        pull
        echo "ROW_DONE tier2 $row $(date -u +%T)"
    done
    echo "STAGE_DONE tier2 $* $(date -u +%T)";;
phase0)   # serving settings: gprand under the final settings (MTP k=3, max-num-seqs 96): KV cache, acceptance, a short TV + Grader workload; then the equivalence gate against plain settings
    stop_vllm
    serve gprand || { echo "PHASE0_FAILED serve final"; exit 1; }
    grep -E "KV cache size|Maximum concurrency|GPU KV cache|num_gpu_blocks|speculative|Speculative" ~/vllm_gprand.log | head -8
    t0=$(date +%s)
    rhbench-run bash -c "cd $RB && ./run.py pd0_tv" > ~/rbrun_pd0_tv.log 2>&1 &
    pid=$!
    ./run.py pd0_grader > ~/run_pd0_grader.log 2>&1 || echo "RUN_FAILED pd0_grader"
    wait $pid || echo "RUN_FAILED pd0_tv"
    echo "PHASE0 workload (pd0_tv 16 grids mt4 as rhbench + pd0_grader n=8 as root) wall $(( $(date +%s) - t0 )) s $(date -u +%T)"
    curl -s localhost:8000/metrics | grep -E "^vllm:(spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total|kv_cache_usage_perc|num_preemptions_total|prefix_cache_(hits|queries)_total)" | sed 's/{[^}]*}//'
    .venv/bin/python scripts/prompt_dom_served_check.py score cb_gprand final || echo "PHASE0_FAILED score final"
    stop_vllm
    PLAIN=1 serve gprand || { echo "PHASE0_FAILED serve plain"; exit 1; }
    .venv/bin/python scripts/prompt_dom_served_check.py score cb_gprand plain || echo "PHASE0_FAILED score plain"
    .venv/bin/python scripts/prompt_dom_served_check.py compare final plain
    stop_vllm
    echo "STAGE_DONE phase0 $(date -u +%T)";;
phase0b)   # after the gate failed with MTP: gprand without MTP at max-num-seqs 96 scored against plain and against the HF reference (prompt_dom_rh.py servedref gprand), then the workload under the settings given in SETTINGS (e.g. SETTINGS="MTP=0")
    stop_vllm
    MTP=0 serve gprand || { echo "PHASE0B_FAILED serve nomtp"; exit 1; }
    .venv/bin/python scripts/prompt_dom_served_check.py score cb_gprand nomtp || echo "PHASE0B_FAILED score nomtp"
    .venv/bin/python scripts/prompt_dom_served_check.py score cb_gprand nomtp2 || echo "PHASE0B_FAILED score nomtp2"   # the same server again: run-to-run noise
    stop_vllm
    PLAIN=1 serve gprand || { echo "PHASE0B_FAILED serve plain"; exit 1; }
    .venv/bin/python scripts/prompt_dom_served_check.py score cb_gprand plain2 || echo "PHASE0B_FAILED score plain2"
    for pair in "nomtp nomtp2" "plain plain2" "nomtp plain" "final plain" "final hf_gprand" "plain hf_gprand" "nomtp hf_gprand" "plain2 hf_gprand"; do .venv/bin/python scripts/prompt_dom_served_check.py compare $pair; done
    stop_vllm
    echo "STAGE_DONE phase0b $(date -u +%T)";;
phase0_workload)   # the short TV + Grader workload under the current environment's serving knobs (MTP, MAX_NUM_SEQS): KV cache, acceptance, wall time
    stop_vllm
    serve gprand || { echo "PHASE0W_FAILED serve"; exit 1; }
    grep -E "GPU KV cache size|Maximum concurrency" ~/vllm_gprand.log | head -2
    t0=$(date +%s)
    rhbench-run bash -c "cd $RB && ./run.py pd0_tv" > ~/rbrun_pd0_tv.log 2>&1 &
    pid=$!
    ./run.py pd0_grader > ~/run_pd0_grader.log 2>&1 || echo "RUN_FAILED pd0_grader"
    wait $pid || echo "RUN_FAILED pd0_tv"
    echo "PHASE0 workload (pd0_tv 16 grids mt4 as rhbench + pd0_grader n=8 as root) wall $(( $(date +%s) - t0 )) s $(date -u +%T)"
    curl -s localhost:8000/metrics | grep -E "^vllm:(spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total|kv_cache_usage_perc|num_preemptions_total|prefix_cache_(hits|queries)_total|generation_tokens_total|prompt_tokens_total)" | sed 's/{[^}]*}//'
    grep -hoE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%" ~/vllm_gprand.log | sort -t: -k4 -n | tail -3
    echo "STAGE_DONE phase0_workload $(date -u +%T)";;
pull_rhbench) pull;;
stop) stop_vllm; echo "STAGE_DONE stop";;
*) sed -n 2,12p "$0"; exit 1;;
esac
