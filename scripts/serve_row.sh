#!/bin/bash
# On the GPU box, as root, from ~/ortho: bash scripts/serve_row.sh <row> serves the PD_MODEL (models.py; default qwen38_27b) as cb_<row> with the hook file ~/hooks/<row>.safetensors through
# scripts/vllm_proj.py (none for row base: the same serving path with no hook), replacing what is served, and waits until it answers. Knobs: MAX_NUM_SEQS (default 256; 64 before the pilot throughput check), MAX_MODEL_LEN (131072),
# GPU_MEMORY_UTILIZATION (0.90), EAGER=1 (--enforce-eager, for RHABLATION_PROJ_CHECK=1 runs), EXTRA (more vllm args). The compile cache does not know about the hook, so every row keeps its own.
set -u
cd ~/ortho
row=$1
eval "$(PD_MODEL=${PD_MODEL:-qwen38_27b} .venv/bin/python -c "
from models import current; m = current()
print(f'HF_ID={m.hf_id}; REV={m.revision}; VLLM={m.vllm_bin}; ARCH={m.proj_arch}; ARGS=\"{\" \".join(m.vllm_args)}\"')")"
tmux kill-session -t vllm 2>/dev/null
for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done
hook=""
[ "$row" != base ] && hook=$HOME/hooks/$row.safetensors && [ ! -f "$hook" ] && { echo "no $hook"; exit 1; }
seqs=${MAX_NUM_SEQS:-256}
eager=$([ "${EAGER:-0}" = 1 ] && echo --enforce-eager)
cache=$HOME/.cache/vllm_$row
tmux new -d -s vllm "VLLM_CACHE_ROOT=$cache RHABLATION_PROJ=$hook RHABLATION_PROJ_ARCH=$ARCH RHABLATION_PROJ_CHECK=${RHABLATION_PROJ_CHECK:-0} VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME:-/workspace/.hf_home} $VLLM serve $HF_ID --revision $REV --served-model-name cb_$row --host 127.0.0.1 --port 8000 --max-model-len ${MAX_MODEL_LEN:-131072} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.90} --max-num-seqs $seqs $ARGS $eager ${EXTRA:-} 2>&1 | tee ~/vllm_$row.log"
for _ in $(seq 360); do
    curl -sf -m 5 localhost:8000/v1/models | grep -q "\"cb_$row\"" && { echo "serving cb_$row ($HF_ID, max-num-seqs $seqs, hook ${hook:-none}, hooked layers $(grep -c 'vllm_proj: hook on' ~/vllm_$row.log))"; exit 0; }
    tmux has-session -t vllm 2>/dev/null || break
    sleep 5
done
echo "cb_$row not serving"; tail -20 ~/vllm_$row.log; exit 1
