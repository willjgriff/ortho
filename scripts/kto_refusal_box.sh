#!/bin/bash
# On the GPU box, from ~/ortho: bash scripts/kto_refusal_box.sh <stage>, kto_refusal.py's runs end to end, one stage at a time and in this order. Each prints
# STAGE_DONE <stage> at the end, RUN_DONE / RUN_FAILED per eval and <STAGE>_FAILED when it cannot go on; tee it to a log.
#   setup    the venv, the serving venv (scripts/kto_serve_setup.sh), the weights, and the refusal records, refusal vectors and base grader records it reads (HF eekay/ortho)
#   sample   a smoke eval, then the base model's answers to heretic's 400 harmless train prompts (run.py refusal38_train_harmless): the neutrals
#   train    kto_refusal.py data, prep, the two-step smoke config, every config, export (vLLM stopped)
#   check    kto_refusal.py check: first-token KL, MMLU and the cosines with the DoM directions (vLLM stopped)
#   suite    each config served by ~/serve_row.sh with run.py's refusal and grader side-effect runs, then kto_refusal.py table (which says whether the LoRA is needed)
#   lora     the KTO LoRA baseline: LORA_ENV=refusal train_lora.py, its KL-matched adapter loaded on a cb_base server, its runs, then the table again
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
ROWS="rkto rktopt rktoall rkrand rkrandall"
REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
stop_vllm() { tmux kill-session -t vllm 2>/dev/null; for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done; sleep 10; }
runs() { for run in "$@"; do ./run.py $run > ~/run_$run.log 2>&1 || ./run.py $run > ~/run_$run.retry.log 2>&1 || echo "RUN_FAILED $run"; echo "RUN_DONE $run $(date -u +%T)"; done; }
py() { .venv/bin/python kto_refusal.py "$@" 2>&1 | tee -a ~/kto_refusal.log; return ${PIPESTATUS[0]}; }
case $1 in
setup)
    set -e
    git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"   # uv.lock fetches mechtools over ssh; it is public over https, and the box holds no ssh key
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_HTTP_TIMEOUT=900 uv sync
    UV_HTTP_TIMEOUT=900 bash scripts/kto_serve_setup.sh
    .venv/bin/hf download Qwen/Qwen3.8-27B --revision $REVISION > /dev/null
    .venv/bin/hf download eekay/ortho --repo-type dataset --local-dir . --include "data/inspect/refusal/*" --include "data/vectors/Qwen3.8-27B/refusal_*" --include "data/inspect/grader/2026-11-05T02-38-*"
    echo "STAGE_DONE setup $(date -u +%T)";;
sample)
    bash ~/serve_row.sh base || { echo "SAMPLE_FAILED serve base"; exit 1; }
    ./run.py smoke_refusal > ~/run_smoke_refusal.log 2>&1 || { echo "SAMPLE_FAILED smoke_refusal, see ~/run_smoke_refusal.log"; exit 1; }
    ./run.py refusal38_train_harmless > ~/run_refusal38_train_harmless.log 2>&1 || ./run.py refusal38_train_harmless > ~/run_refusal38_train_harmless.retry.log 2>&1 || { echo "SAMPLE_FAILED refusal38_train_harmless"; exit 1; }
    echo "RUN_DONE refusal38_train_harmless $(date -u +%T)"
    stop_vllm
    echo "STAGE_DONE sample $(date -u +%T)";;
train)
    stop_vllm
    py data || { echo "TRAIN_FAILED data"; exit 1; }
    [ -f acts/kto_refusal/base.pt ] || py prep || { echo "TRAIN_FAILED prep"; exit 1; }
    py run rksmoke || { echo "TRAIN_FAILED smoke"; exit 1; }
    for row in $ROWS; do
        grep -q "\"name\": \"$row\"" acts/kto_refusal/results.jsonl 2>/dev/null || py run $row || { echo "TRAIN_FAILED $row"; exit 1; }
        echo "TRAINED $row $(date -u +%T)"
    done
    py export $ROWS || { echo "TRAIN_FAILED export"; exit 1; }
    echo "STAGE_DONE train $(date -u +%T)";;
check)
    stop_vllm
    py check $ROWS || { echo "CHECK_FAILED"; exit 1; }
    echo "STAGE_DONE check $(date -u +%T)";;
suite)
    for row in $ROWS; do
        [ -f ~/hooks/$row.safetensors ] || { echo "SUITE_FAILED no ~/hooks/$row.safetensors"; exit 1; }
        curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$row\"" || bash ~/serve_row.sh $row || { echo "SUITE_FAILED serve $row"; exit 1; }
        echo "SERVING $row, hooked layers: $(grep -c "vllm_proj: hook on" ~/vllm_$row.log) $(date -u +%T)"
        runs ${row}_harmful ${row}_harmless ${row}_harmful_low ${row}_harmless_low ${row}_none ${row}_clean
    done
    stop_vllm
    py table || { echo "SUITE_FAILED table"; exit 1; }
    echo "STAGE_DONE suite $(date -u +%T)";;
lora)
    stop_vllm
    LORA_ENV=refusal .venv/bin/python train_lora.py > ~/train_lora.log 2>&1 || { echo "LORA_FAILED train, see ~/train_lora.log"; exit 1; }
    out=data/loras/Qwen3.8-27B/refusal_kto_lora
    step=$(.venv/bin/python -c "import json; print(json.load(open('$out/held_out.json'))['kl_matched_step'])")
    adapter=$(printf "%s/step_%04d" "$PWD/$out" "$step")
    echo "LORA_TRAINED, KL-matched adapter $adapter $(date -u +%T)"
    # the base weights as cb_base with LoRA on, from the serving venv of kto_serve_setup.sh (no projection hook, no speculative decoding, the V1 runner)
    tmux new -d -s vllm "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=$HOME/.cache/vllm_lora ~/serve/bin/vllm serve Qwen/Qwen3.8-27B --revision $REVISION --served-model-name cb_base --host 127.0.0.1 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only --max-num-seqs 64 --enable-lora --max-loras 2 --max-lora-rank 16 2>&1 | tee ~/vllm_lora.log"
    for _ in $(seq 240); do curl -sf -m 5 localhost:8000/v1/models | grep -q cb_base && break; tmux has-session -t vllm 2>/dev/null || break; sleep 5; done
    curl -sf -m 5 localhost:8000/v1/models | grep -q cb_base || { echo "LORA_FAILED serve, see ~/vllm_lora.log"; exit 1; }
    curl -sf -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\": \"refusal_kto_lora\", \"lora_path\": \"$adapter\"}" || { echo "LORA_FAILED load adapter"; exit 1; }
    runs refusallora_harmful refusallora_harmless refusallora_harmful_low refusallora_harmless_low refusallora_none refusallora_clean
    final=$(ls -d $PWD/$out/step_* | sort | tail -1)
    echo "LORA_FINAL adapter $final $(date -u +%T)"
    curl -sf -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\": \"refusal_kto_lora_final\", \"lora_path\": \"$final\"}" || { echo "LORA_FAILED load final adapter"; exit 1; }
    runs refusallorafinal_harmful refusallorafinal_harmless refusallorafinal_harmful_low refusallorafinal_harmless_low refusallorafinal_none refusallorafinal_clean
    stop_vllm
    py table || { echo "LORA_FAILED table"; exit 1; }
    echo "STAGE_DONE lora $(date -u +%T)";;
*)
    sed -n 2,10p "$0"; exit 1;;
esac
