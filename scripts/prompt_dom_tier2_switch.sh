#!/bin/bash
# Reduced tier 2 (user, 2026-11-24 09:50 UTC: hard end 22:01 UTC; tier 2 for base, gparditi, tvpdom and tvarditi only). The running prompt_dom_after_t1.sh chain carries the old
# tier-2 list and must not be edited while bash runs it, so this waits for its refusal stage to finish (STAGE_DONE refusal in its log), stops the chain's process group before its
# tier-2 stage has done more than start a server, stops that server, and runs the reduced tier 2. Run from /root/ortho as root:
#   tmux new -d -s t2switch "bash scripts/prompt_dom_tier2_switch.sh > /root/prompt_dom_out/logs/tier2.log 2>&1"
cd ~/ortho
export PATH=$HOME/.local/bin:$PATH
LOG=/root/prompt_dom_out/logs/after_t1.log
ROWS=${T2_ROWS:-"base gparditi tvpdom tvarditi"}
echo "T2SWITCH waiting for STAGE_DONE refusal $(date -u +%T)"
until grep -qaE "STAGE_DONE refusal|AFTER_T1_DONE|AFTER_T1_EXIT" $LOG; do sleep 1; done
pgid=$(tmux list-panes -t after_t1_launcher -F '#{pane_pid}' 2>/dev/null)
if [ -n "$pgid" ]; then kill -TERM -- -$pgid 2>/dev/null; sleep 3; kill -KILL -- -$pgid 2>/dev/null; echo "CHAIN_STOPPED pgid $pgid $(date -u +%T)"; else echo "CHAIN_GONE $(date -u +%T)"; fi
grep -aE "STAGE_DONE|ROW_DONE|RUN_FAILED|SERVE_FAILED|AFTER_T1" $LOG | tail -12
bash scripts/prompt_dom_box.sh stop
for _ in $(seq 60); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${used:-0}" -lt 5000 ] && break; sleep 2; done   # the server the chain's tier 2 may have started frees the GPU
echo "GPU memory used ${used} MiB $(date -u +%T)"
bash scripts/prompt_dom_box.sh tier2 $ROWS
bash scripts/prompt_dom_box.sh stop
echo "T2_DONE $(date -u +%T)"
