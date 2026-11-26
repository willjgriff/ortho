#!/bin/bash
# On the GPU box, as root, from /root/ortho: bash scripts/second_model_upload.sh [what ...], the uploads of sweep/mission_second_model.md through the project's scripts/upload_run.sh
# (data-only staging copies under /root/second_model_out/stage/<what>/, dest prefix second_model/<tag>/<what>/). what in: acts vectors records inspect_logs box_logs report judge (public,
# default all of them) and private (the refusal records, logs, judge outputs and samples: harmful-request completions, uploaded with --private to the repo named by HF_PRIVATE_REPO as
# second_model/<tag>/private/). SM_MODEL picks the tag (default gemma4_31b).
set -u
cd ~/ortho
TAG=${SM_MODEL:-gemma4_31b}
PROJ=/workspace/llm-reward-hacking-ablation
S=/root/second_model_out/stage
P=/root/second_model_out/private
SINCE=${SINCE:-2026-11-26T10}
what=${*:-acts vectors records inspect_logs box_logs report judge private}
mkdir -p $S $P
for w in $what; do
    if [ $w = private ]; then
        mkdir -p $P/records/refusal $P/inspect_logs/refusal $P/box_logs
        for f in data/inspect/refusal/*.jsonl; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp -n $f $P/records/refusal/; done
        for f in logs/refusal/*.eval; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp -n $f $P/inspect_logs/refusal/; done
        for f in /root/second_model_out/logs/run_*ref_*.log; do [ -f "$f" ] && cp $f $P/box_logs/; done
        [ -d $PROJ/data/refusal_judge ] && { mkdir -p $P/judge; rsync -a --exclude '*.py' $PROJ/data/refusal_judge/ $P/judge/; }
        [ -z "$(ls -A $P 2>/dev/null)" ] && { echo "nothing to upload for private"; continue; }
        find $P -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $P, not uploading"; continue; }
        (cd $PROJ && bash scripts/upload_run.sh --private --run-dir $P --dest-prefix second_model/$TAG) && echo "UPLOADED private -> second_model/$TAG/private (gated repo)" || echo "UPLOAD_FAILED private"
        continue
    fi
    rm -rf $S/$w; mkdir -p $S/$w
    case $w in
    acts) cp acts/second_model/$TAG/*.pt acts/second_model/$TAG/*.json acts/second_model/$TAG/*.md $S/acts/ 2>/dev/null; [ -d acts/prompt_dom/lmeval ] && cp -r acts/prompt_dom/lmeval $S/acts/;;
    vectors) mkdir -p $S/vectors/hooks; cp data/vectors/$TAG/sm_*.safetensors data/vectors/$TAG/sm_*.json $S/vectors/ 2>/dev/null; cp ~/hooks/*.safetensors $S/vectors/hooks/ 2>/dev/null;;
    records) for env in grader terminal_verifier; do mkdir -p $S/records/$env; for f in data/inspect/$env/*.jsonl; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/records/$env/; done; done;;
    inspect_logs) for env in grader terminal_verifier; do mkdir -p $S/inspect_logs/$env; for f in logs/$env/*.eval; do [ -f "$f" ] && [[ $(basename $f) > $SINCE ]] && cp $f $S/inspect_logs/$env/; done; done;;
    box_logs) for f in /root/second_model_out/logs/*.log ~/vllm_*.log; do [ -f "$f" ] && case $(basename $f) in run_*ref_*) ;; *) cp $f $S/box_logs/;; esac; done;;
    report) rm -rf $S/report; cp -r $PROJ/box_report/second_model_$TAG $S/report; find $S/report -name '*.py' -delete;;
    judge) [ -d $PROJ/data/stage3/audit ] && for d in $PROJ/data/stage3/audit/tv_prompt_dom*; do [ -d "$d" ] && { mkdir -p $S/judge/$(basename $d); rsync -a --exclude '*.py' --exclude transcripts --exclude prompts --exclude reading $d/ $S/judge/$(basename $d)/; }; done;;
    esac
    [ -z "$(ls -A $S/$w 2>/dev/null)" ] && { echo "nothing to upload for $w"; continue; }
    find $S/$w -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $S/$w, not uploading"; continue; }
    (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/$w --dest-prefix second_model/$TAG) && echo "UPLOADED $w -> second_model/$TAG/$w" || echo "UPLOAD_FAILED $w"
done
echo "private staging: $(find $P -type f | wc -l) files under $P"
