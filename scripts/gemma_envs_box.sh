#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/gemma_envs_box.sh <stage> [args] (sweep/mission_gemma_envs.md; sweep/mission_model_pilots.md: GE_TAG picks the model,
# default gemma4_31b, the serve script's SM_MODEL). Prints RUN_DONE / RUN_FAILED lines.
#   sync                 rhbench's readable copies of this clone and /root/rh-envs refreshed (scripts/second_model_box.sh rhbench_sync); after every code or config change
#   rb <row> <run> ...   gemma_envs.py runs as rhbench (every env that executes model output) against cb_<row>, log $OUTROOT/logs/rb_<row>_<first run>.log
#   pull                 copy rhbench's run directories (data only) to $OUTROOT/pulled
#   serve <row>          bash ~/serve_row.sh <row> (after refreshing it from scripts/second_model_serve_row.sh)
cd ~/ortho
export GE_TAG=${GE_TAG:-gemma4_31b}
RB=/home/rhbench/ortho
if [ "$GE_TAG" = gemma4_31b ]; then OUTROOT=/root/gemma_envs_out; SRC=$RB/data/gemma_envs; else OUTROOT=/root/mp_out/$GE_TAG; SRC=$RB/data/model_pilots/$GE_TAG; fi
LOGS=$OUTROOT/logs; mkdir -p $LOGS
case $1 in
sync) bash scripts/second_model_box.sh rhbench_sync;;
rb) row=$2; shift 2; log=$LOGS/rb_${row}_$1.log
    rhbench-run bash -c "cd $RB && GE_TAG=$GE_TAG ./gemma_envs.py run $row $*" >> $log 2>&1; echo "RB_DONE $row $* $(date -u +%T)" >> $log;;
pull) mkdir -p $OUTROOT/pulled; [ -d $SRC ] && rsync -a --no-links $SRC/ $OUTROOT/pulled/ && find $OUTROOT/pulled -type f \( -name '*.py' -o -name '*.sh' -o -name '*.ipynb' \) -delete; echo "pulled $(date -u +%T)";;
serve) cp scripts/second_model_serve_row.sh ~/serve_row.sh; SM_MODEL=$GE_TAG bash ~/serve_row.sh $2;;
*) sed -n 2,8p "$0"; exit 1;;
esac
