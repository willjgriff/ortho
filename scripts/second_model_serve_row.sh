#!/bin/bash
# On the GPU box: bash ~/serve_row.sh <row> serves the second model's base weights as cb_<row> with the hook file ~/hooks/<row>.safetensors through scripts/vllm_proj.py (installed as
# a vLLM plugin in /root/vllm-env), replacing what is served, and waits until it answers. The model is a parameter: SM_MODEL (default gemma4_31b) picks the entry below; a row's hook
# file must exist (row base has a zero hook, written by second_model.py hooks, so base runs through the same patched forward). Installed by scripts/second_model_box.sh setup.
# Knobs: MAX_NUM_SEQS (default 80: at 128 the KV cache ran at 99.8% with 50 requests waiting and 4,234 preemptions in the pilot; the cache holds about 75 of these long requests), MAX_MODEL_LEN (default 131072), GPU_MEMORY_UTILIZATION (default 0.95), PLAIN=1 serves without the plugin (RHABLATION_PROJ unset) for the equivalence check.
row=$1
case ${SM_MODEL:-gemma4_31b} in
gemma4_31b) MODEL=google/gemma-4-31B-it; REVISION=842da3794eaa0b77d5f08bae87a17459d91ff475; FLAGS="--reasoning-parser gemma4 --language-model-only --enable-auto-tool-choice --tool-call-parser gemma4";;   # multimodal checkpoint served text-only; tool parser for the tool-calling envs of sweep/mission_gemma_envs.md (data_entry, revert, hanoi, overclaim)
*) echo "unknown SM_MODEL ${SM_MODEL}"; exit 1;;
esac
tmux kill-session -t vllm 2>/dev/null
for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done
hook=$HOME/hooks/$row.safetensors
[ -f "$hook" ] || { echo "no $hook"; exit 1; }
[ "${PLAIN:-0}" = 1 ] && hook=""
seqs=${MAX_NUM_SEQS:-80}
cache=$HOME/.cache/vllm_sm_$row   # the compile cache does not know about the hook or its layers, so every row keeps its own
tmux new -d -s vllm "export HF_HOME=/workspace/.hf_home VLLM_CACHE_ROOT=$cache RHABLATION_PROJ=$hook VLLM_USE_FLASHINFER_SAMPLER=0; /root/vllm-env/bin/vllm serve $MODEL --revision $REVISION --served-model-name cb_$row --host 127.0.0.1 --port 8000 --max-model-len ${MAX_MODEL_LEN:-131072} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.95} $FLAGS --max-num-seqs $seqs 2>&1 | tee ~/vllm_$row.log"
for _ in $(seq 360); do
    curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$row\"" && { echo "serving cb_$row (${SM_MODEL:-gemma4_31b}, max-num-seqs $seqs, hook ${hook:-none}); hooked layers: $(grep -c 'vllm_proj: hook on' ~/vllm_$row.log)"; exit 0; }
    tmux has-session -t vllm 2>/dev/null || break
    sleep 5
done
echo "cb_$row not serving"; tail -20 ~/vllm_$row.log; exit 1
