#!/bin/bash
# On the GPU box, as root from /root/ortho: bash scripts/gemma_envs_upload.sh [what ...]: data-only staging copies under $OUTROOT/stage/<what>/ uploaded with the
# project's scripts/upload_run.sh. GE_TAG picks the model (default gemma4_31b: sweep/mission_gemma_envs.md, second_model/gemma4_31b/envs/<what>/; any other tag:
# sweep/mission_model_pilots.md, model_pilots/<tag>/<what>/ and the report under model_pilots/report/). what in: records inspect_logs runs acts vectors box_logs report private
# (default all). Refusal runs (completions to harmful requests) are never staged publicly: `private` uploads them to the private repo under <prefix>/refusal/.
set -u
cd ~/ortho
export GE_TAG=${GE_TAG:-gemma4_31b}
PROJ=/workspace/llm-reward-hacking-ablation
if [ "$GE_TAG" = gemma4_31b ]; then OUTROOT=/root/gemma_envs_out; DEST=second_model/gemma4_31b/envs; REPORT=$PROJ/box_report/gemma_envs; RDEST=$DEST; SINCE=2026-11-27T11
else OUTROOT=/root/mp_out/$GE_TAG; DEST=model_pilots/$GE_TAG; REPORT=$PROJ/box_report/model_pilots; RDEST=model_pilots; SINCE=2026-11-28T11; fi
S=$OUTROOT/stage
PULLED=$OUTROOT/pulled
what=${*:-records inspect_logs runs acts vectors box_logs report private}
bash scripts/gemma_envs_box.sh pull > /dev/null
mkdir -p $S
for w in $what; do
    rm -rf $S/$w; mkdir -p $S/$w
    case $w in
    records) for f in $(cd $PULLED && find . -name records.jsonl -not -path '*/refusal/*'); do d=$S/records/$(dirname $f); mkdir -p $(dirname $d); cp $PULLED/$f $d.jsonl; done
             [ "$GE_TAG" = gemma4_31b ] && { mkdir -p $S/records/root/grader; for f in data/inspect/grader/*.jsonl; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/records/root/grader/; done; };;
    inspect_logs) for f in $(cd $PULLED && find . -name '*.eval' -not -path '*/refusal/*'); do d=$S/inspect_logs/$(dirname $f); mkdir -p $(dirname $d); cp $PULLED/$f $d.eval; done
             [ "$GE_TAG" = gemma4_31b ] && { mkdir -p $S/inspect_logs/root/grader; for f in logs/grader/*.eval; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/inspect_logs/root/grader/; done; };;
    runs) for f in $(cd $PULLED && find . -name run_config.json -o -name progress.jsonl); do mkdir -p $S/runs/$(dirname $f); cp $PULLED/$f $S/runs/$f; done;;
    acts) [ -d acts/gemma_envs ] && [ "$GE_TAG" = gemma4_31b ] && rsync -a --exclude '*.py' acts/gemma_envs/ $S/acts/;;
    vectors) if [ "$GE_TAG" = gemma4_31b ]; then [ -d data/vectors/gemma4_31b/envs ] && rsync -a data/vectors/gemma4_31b/envs/ $S/vectors/; mkdir -p $S/vectors/hooks; cp ~/hooks/*.safetensors $S/vectors/hooks/ 2>/dev/null
             else mkdir -p $S/vectors/hooks; cp ~/hooks/$GE_TAG/* $S/vectors/hooks/ 2>/dev/null; cp $OUTROOT/hook_check/*.json $S/vectors/hooks/ 2>/dev/null; fi;;
    box_logs) for f in $OUTROOT/logs/*.log ~/vllm_*.log; do [ -f "$f" ] && cp $f $S/box_logs/; done;;
    report) cp -r $REPORT/. $S/report/; find $S/report -name '*.py' -delete;;
    private) for f in $(cd $PULLED && find . -path '*/refusal/*' -type f \( -name '*.jsonl' -o -name '*.eval' -o -name '*.json' \)); do mkdir -p $S/private/$(dirname $f); cp $PULLED/$f $S/private/$f; done;;
    esac
    [ -z "$(ls -A $S/$w 2>/dev/null)" ] && { echo "nothing to upload for $w"; continue; }
    find $S/$w -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $S/$w, not uploading"; continue; }
    if [ $w = private ]; then mv $S/private $S/refusal; (cd $PROJ && bash scripts/upload_run.sh --private --run-dir $S/refusal --dest-prefix $DEST) > $OUTROOT/logs/upload_$w.out 2>&1 && echo "UPLOADED $w (private)" || echo "UPLOAD_FAILED $w"; rm -rf $S/private; mv $S/refusal $S/private; continue; fi
    if [ $w = report ]; then (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/report --dest-prefix $RDEST) > $OUTROOT/logs/upload_$w.out 2>&1 && echo "UPLOADED $w" || echo "UPLOAD_FAILED $w"; continue; fi
    (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/$w --dest-prefix $DEST) > $OUTROOT/logs/upload_$w.out 2>&1 && echo "UPLOADED $w" || echo "UPLOAD_FAILED $w (see $OUTROOT/logs/upload_$w.out)"
done
/venv/main/bin/python scripts/gemma_envs_verify_upload.py $what
