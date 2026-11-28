#!/bin/bash
# sweep/mission_model_pilots.md: the 30-minute upload loop (tmux uploader, as root from /root/ortho). Each cycle uploads every model tag listed in
# /root/mp_out/tags (one per line: the current model and any not yet final-uploaded) with scripts/gemma_envs_upload.sh, and logs to /root/mp_out/uploader.log.
while true; do
    for tag in $(cat /root/mp_out/tags 2>/dev/null); do
        echo "=== $(date -u +%FT%TZ) $tag" >> /root/mp_out/uploader.log
        GE_TAG=$tag bash scripts/gemma_envs_upload.sh >> /root/mp_out/uploader.log 2>&1
    done
    sleep ${UPLOAD_EVERY:-1800}
done
