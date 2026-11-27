#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/gemma_envs_box.sh <stage> [args] (sweep/mission_gemma_envs.md). Prints RUN_DONE / RUN_FAILED lines.
#   sync                 rhbench's readable copies of this clone and /root/rh-envs refreshed (scripts/second_model_box.sh rhbench_sync); after every code or config change
#   rb <row> <run> ...   gemma_envs.py runs as rhbench (every env that executes model output) against cb_<row>, log /root/gemma_envs_out/logs/rb_<row>_<first run>.log
#   pull                 copy rhbench's run directories (data/gemma_envs: records, .eval logs, run_config.json, progress.jsonl; data only) to /root/gemma_envs_out/pulled
#   serve <row>          bash ~/serve_row.sh <row> (after refreshing it from scripts/second_model_serve_row.sh)
cd ~/ortho
RB=/home/rhbench/ortho
LOGS=/root/gemma_envs_out/logs; mkdir -p $LOGS
case $1 in
sync) bash scripts/second_model_box.sh rhbench_sync;;
rb) row=$2; shift 2; log=$LOGS/rb_${row}_$1.log
    rhbench-run bash -c "cd $RB && ./gemma_envs.py run $row $*" >> $log 2>&1; echo "RB_DONE $row $* $(date -u +%T)" >> $log;;
pull) mkdir -p /root/gemma_envs_out/pulled; rsync -a --no-links $RB/data/gemma_envs/ /root/gemma_envs_out/pulled/ && find /root/gemma_envs_out/pulled -type f \( -name '*.py' -o -name '*.sh' -o -name '*.ipynb' \) -delete; echo "pulled $(date -u +%T)";;
serve) cp scripts/second_model_serve_row.sh ~/serve_row.sh; bash ~/serve_row.sh $2;;
*) sed -n 2,7p "$0"; exit 1;;
esac
