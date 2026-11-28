#!/bin/bash
# sweep/mission_model_pilots.md: SUMMARY.md = /root/mp_out/summary_head.md + per model (in mission order) section_head.md, screen.md, section_tail.md + /root/mp_out/final_table.md
P=/workspace/llm-reward-hacking-ablation/box_report/model_pilots
{ cat /root/mp_out/summary_head.md
  for t in nanbeige43_33b k2h_32b k2h_mova_36b muse_glimmer_30b k2h_7b; do
    d=/root/mp_out/$t; [ -f $d/section_head.md ] || continue
    echo; cat $d/section_head.md; echo; [ -f $d/screen.md ] && cat $d/screen.md; [ -f $d/section_tail.md ] && cat $d/section_tail.md
  done
  [ -f /root/mp_out/final_table.md ] && { echo; cat /root/mp_out/final_table.md; }
} > $P/SUMMARY.md
echo "SUMMARY.md $(wc -l < $P/SUMMARY.md) lines"
