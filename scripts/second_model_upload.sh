#!/bin/bash
# On the GPU box, as root, from /root/ortho: PD_MODEL=<tag> bash scripts/second_model_upload.sh [what ...], the uploads of sweep/mission_second_model.md through the project's
# scripts/upload_run.sh: data-only staging copies under /root/second_model_out/stage/<what>/, dest prefix second_model/<tag>/<what>. what in: records inspect_logs acts vectors box_logs
# judge report (default all) and private (every refusal record, Inspect log, run log, judge output and the refusal samples, uploaded with --private to HF_PRIVATE_REPO under second_model/<tag>/refusal).
set -u
cd ~/ortho
TAG=${PD_MODEL:?set PD_MODEL}
PROJ=/workspace/llm-reward-hacking-ablation
S=/root/second_model_out/stage
P=/root/second_model_out/private_stage/refusal
ACTS=acts/second_model/$TAG
what=${*:-records inspect_logs acts vectors box_logs judge report private}
mkdir -p $S $P
for w in $what; do
    if [ $w = private ]; then
        mkdir -p $P/records $P/inspect_logs $P/box_logs $P/judge
        cp data/inspect/refusal/*.jsonl $P/records/ 2>/dev/null; cp logs/refusal/*.eval $P/inspect_logs/ 2>/dev/null
        cp /root/second_model_out/logs/run_*harmful* /root/second_model_out/logs/run_*harmless* $P/box_logs/ 2>/dev/null
        cp $ACTS/refusal_samples.md $P/ 2>/dev/null
        [ -d /root/second_model_out/judge/refusal ] && rsync -a --exclude '*.py' /root/second_model_out/judge/refusal/ $P/judge/
        find $P -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $P, not uploading"; continue; }
        (cd $PROJ && bash scripts/upload_run.sh --private --run-dir $P --dest-prefix second_model/$TAG) && echo "UPLOADED private refusal -> second_model/$TAG/refusal (gated repo)" || echo "UPLOAD_FAILED private"
        continue
    fi
    rm -rf $S/$w; mkdir -p $S/$w
    case $w in
    records) for env in grader terminal_verifier; do mkdir -p $S/records/$env; cp data/inspect/$env/*.jsonl $S/records/$env/ 2>/dev/null; done;;
    inspect_logs) for env in grader terminal_verifier; do mkdir -p $S/inspect_logs/$env; cp logs/$env/*.eval $S/inspect_logs/$env/ 2>/dev/null; done;;
    acts) cp $ACTS/*.pt $ACTS/*.json $ACTS/*.md $S/acts/ 2>/dev/null; rm -f $S/acts/refusal_samples.md; [ -d $ACTS/lmeval ] && cp -r $ACTS/lmeval $S/acts/;;
    vectors) mkdir -p $S/vectors/hooks; cp data/vectors/$TAG/*.safetensors data/vectors/$TAG/*.json $S/vectors/ 2>/dev/null; cp ~/hooks/*.safetensors $S/vectors/hooks/ 2>/dev/null;;
    box_logs) cp /root/second_model_out/logs/*.log ~/vllm_*.log ~/rhbench_uv_sync.log $S/box_logs/ 2>/dev/null; rm -f $S/box_logs/run_*harmful* $S/box_logs/run_*harmless*;;
    judge) [ -d /root/second_model_out/judge/tv ] && rsync -a --exclude '*.py' /root/second_model_out/judge/tv/ $S/judge/tv/;;
    report) rm -rf $S/report; cp -r $PROJ/box_report/second_model_$TAG $S/report;;
    esac
    [ -z "$(ls -A $S/$w 2>/dev/null)" ] && { echo "nothing to upload for $w"; continue; }
    find $S/$w -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $S/$w, not uploading"; continue; }
    (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/$w --dest-prefix second_model/$TAG) && echo "UPLOADED $w -> second_model/$TAG/$w" || echo "UPLOAD_FAILED $w"
done
echo "private staging: $(find $P -type f 2>/dev/null | wc -l) files under $P"
