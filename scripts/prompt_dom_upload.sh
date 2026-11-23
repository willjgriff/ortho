#!/bin/bash
# On the GPU box, as root, from /root/ortho: bash scripts/prompt_dom_upload.sh [what ...], the uploads of sweep/mission_prompt_dom.md through the project's scripts/upload_run.sh
# (data-only staging copies under /root/prompt_dom_out/stage/<what>/, dest prefix prompt_dom/<what>/). what in: acts vectors records inspect_logs box_logs families report (default all).
# Private items (the refusal records and logs of refarditi / zrand) are staged under /root/prompt_dom_out/private_staging/ and uploaded with --private only when the Hub reports the
# private repo as private (upload_run.sh refuses otherwise; see LOG.md "Contradictions").
set -u
cd ~/ortho
PROJ=/workspace/llm-reward-hacking-ablation
S=/root/prompt_dom_out/stage
P=/root/prompt_dom_out/private_staging
SINCE=${SINCE:-2026-11-23T20}
what=${*:-acts vectors records inspect_logs box_logs families report}
mkdir -p $S $P
for w in $what; do
    rm -rf $S/$w; mkdir -p $S/$w
    case $w in
    acts) cp acts/prompt_dom/*.pt acts/prompt_dom/*.json acts/prompt_dom/*.md $S/acts/ 2>/dev/null;;
    vectors) mkdir -p $S/vectors/hooks; cp data/vectors/Qwen3.8-27B/pd_*.safetensors data/vectors/Qwen3.8-27B/pd_*.json $S/vectors/ 2>/dev/null; cp ~/hooks/*.safetensors $S/vectors/hooks/ 2>/dev/null;;
    records)   # this box's records: logs created from SINCE; refusal records go to the private staging
        for env in grader terminal_verifier mbpp; do mkdir -p $S/records/$env; for f in data/inspect/$env/*.jsonl; do [[ $(basename $f) > $SINCE ]] && cp $f $S/records/$env/; done; done
        mkdir -p $P/records/refusal; for f in data/inspect/refusal/*.jsonl; do [[ $(basename $f) > $SINCE ]] && cp $f $P/records/refusal/; done;;
    inspect_logs)
        for env in grader terminal_verifier mbpp; do mkdir -p $S/inspect_logs/$env; for f in logs/$env/*.eval; do [[ $(basename $f) > $SINCE ]] && cp $f $S/inspect_logs/$env/; done; done
        mkdir -p $P/inspect_logs/refusal; for f in logs/refusal/*.eval; do [[ $(basename $f) > $SINCE ]] && cp $f $P/inspect_logs/refusal/; done;;
    box_logs) cp ~/run_*.log ~/rbrun_*.log ~/vllm_*.log ~/rhbench_uv_sync.log /root/prompt_dom_out/logs/*.log $S/box_logs/ 2>/dev/null; for f in ~/run_refarditi_* ~/run_zrand_harm* ~/run_zrand_harmless*; do [ -f "$f" ] && mv $S/box_logs/$(basename $f) $P/ 2>/dev/null; done;;
    families) cp prompt_dom/grader_fit_families.yaml $S/families/;;
    report) rm -rf $S/report; cp -r $PROJ/box_report/prompt_dom $S/report;;
    esac
    [ -z "$(ls -A $S/$w 2>/dev/null)" ] && { echo "nothing to upload for $w"; continue; }
    find $S/$w -name '*.py' -o -name '*.sh' -o -name '*.ipynb' | grep -q . && { echo "code in $S/$w, not uploading"; continue; }
    if [ $w = report ]; then (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/report --dest-prefix prompt_dom) && echo "UPLOADED report -> prompt_dom/report" || echo "UPLOAD_FAILED report"
    else (cd $PROJ && bash scripts/upload_run.sh --run-dir $S/$w --dest-prefix prompt_dom) && echo "UPLOADED $w -> prompt_dom/$w" || echo "UPLOAD_FAILED $w"; fi
done
echo "private staging: $(find $P -type f | wc -l) files under $P (uploaded only with --private when the repo is private)"
