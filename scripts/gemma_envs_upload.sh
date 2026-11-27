#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/gemma_envs_upload.sh [what ...] (sweep/mission_gemma_envs.md): data-only staging copies under /root/gemma_envs_out/stage/<what>/
# uploaded with the project's scripts/upload_run.sh to second_model/gemma4_31b/envs/<what>/ (public repo). what in: records inspect_logs runs acts vectors box_logs report (default all).
set -u
cd ~/ortho
PROJ=/workspace/llm-reward-hacking-ablation
S=/root/gemma_envs_out/stage
PULLED=/root/gemma_envs_out/pulled
SINCE=2026-11-27T11
what=${*:-records inspect_logs runs acts vectors box_logs report}
bash scripts/gemma_envs_box.sh pull > /dev/null
mkdir -p $S
for w in $what; do
    rm -rf $S/$w; mkdir -p $S/$w
    case $w in
    records) for f in $(cd $PULLED && find . -name records.jsonl); do d=$S/records/$(dirname $f); mkdir -p $(dirname $d); cp $PULLED/$f $d.jsonl; done
             mkdir -p $S/records/root/grader; for f in data/inspect/grader/*.jsonl; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/records/root/grader/; done;;
    inspect_logs) for f in $(cd $PULLED && find . -name '*.eval'); do d=$S/inspect_logs/$(dirname $f); mkdir -p $(dirname $d); cp $PULLED/$f $d.eval; done
             mkdir -p $S/inspect_logs/root/grader; for f in logs/grader/*.eval; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/inspect_logs/root/grader/; done;;
    runs) for f in $(cd $PULLED && find . -name run_config.json -o -name progress.jsonl); do mkdir -p $S/runs/$(dirname $f); cp $PULLED/$f $S/runs/$f; done;;
    acts) [ -d acts/gemma_envs ] && rsync -a --exclude '*.py' acts/gemma_envs/ $S/acts/;;
    vectors) [ -d data/vectors/gemma4_31b/envs ] && rsync -a data/vectors/gemma4_31b/envs/ $S/vectors/; mkdir -p $S/vectors/hooks; cp ~/hooks/*.safetensors $S/vectors/hooks/ 2>/dev/null;;
    box_logs) for f in /root/gemma_envs_out/logs/*.log ~/vllm_*.log; do [ -f "$f" ] && cp $f $S/box_logs/; done;;
    report) cp -r $PROJ/box_report/gemma_envs/. $S/report/; find $S/report -name '*.py' -delete;;
    esac
    [ -z "$(ls -A $S/$w 2>/dev/null)" ] && { echo "nothing to upload for $w"; continue; }
    find $S/$w -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $S/$w, not uploading"; continue; }
    (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/$w --dest-prefix second_model/gemma4_31b/envs) > /root/gemma_envs_out/logs/upload_$w.out 2>&1 && echo "UPLOADED $w" || echo "UPLOAD_FAILED $w (see /root/gemma_envs_out/logs/upload_$w.out)"
done
.venv/bin/python scripts/gemma_envs_verify_upload.py $what
