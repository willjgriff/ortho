#!/bin/bash
# On the GPU box: bash ~/serve_w.sh <row> serves the weight-edited checkpoint ~/models/cb_<row> (kto_orthog.py build) as cb_<row> with no projection hook, through Will's
# vllm_resid_bias plugin (RHABLATION_RESID_BIAS=1: every decoder layer's mlp.down_proj gets the bias the checkpoint carries; zero for base_w and the _wz rows), replacing
# what is served, and waits until it answers. The other flags are scripts/kto_serve_row.sh's, so the hook rows and the edited rows share one serving stack. Installed by scripts/kto_serve_setup.sh.
row=$1
dir=$HOME/models/cb_$row
[ -f "$dir/model.safetensors.index.json" ] && [ -f "$dir/model-resid-bias.safetensors" ] || { echo "no checkpoint at $dir"; exit 1; }
tmux kill-session -t vllm 2>/dev/null
for _ in $(seq 30); do curl -sf -m 2 localhost:8000/v1/models > /dev/null || break; sleep 2; done
cache=$HOME/.cache/vllm_w   # one compile cache for every edited row: the graph (down_proj with a bias) is the same, the weights differ
tmux new -d -s vllm "VLLM_CACHE_ROOT=$cache RHABLATION_RESID_BIAS=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/serve/bin/vllm serve $dir --served-model-name cb_$row --host 127.0.0.1 --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only --max-num-seqs 64 --speculative-config \"{\\\"method\\\": \\\"mtp\\\", \\\"num_speculative_tokens\\\": 3}\" 2>&1 | tee ~/vllm_$row.log"
for _ in $(seq 240); do
    curl -sf -m 5 localhost:8000/v1/models | grep -q "cb_$row" && { echo "serving cb_$row"; grep -c "rhablation_resid_bias" ~/vllm_$row.log; exit 0; }
    tmux has-session -t vllm 2>/dev/null || break
    sleep 5
done
echo "cb_$row not serving"; tail -20 ~/vllm_$row.log; exit 1
