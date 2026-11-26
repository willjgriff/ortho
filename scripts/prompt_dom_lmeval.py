"""Tier 2 of sweep/mission_prompt_dom.md: GSM8K and IFEval through lm-eval against the served row on localhost:8000 (local-chat-completions, the row's own chat template, reasoning off
via the request's chat_template_kwargs, greedy, a fixed sample of the first 500 items of each task, the same for every row). Run from /root/ortho with the lm-eval venv:
    /root/lmeval/bin/python scripts/prompt_dom_lmeval.py <served name, e.g. cb_gpdom> [limit, default 500] [tasks, default gsm8k_cot_zeroshot,ifeval]
Writes acts/prompt_dom/lmeval/<served name>.json (results and the per-task config) and prints the metrics."""
import json
import os
import sys
from pathlib import Path

import lm_eval

served = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
tasks = (sys.argv[3] if len(sys.argv) > 3 else "gsm8k_cot_zeroshot,ifeval").split(",")
out = Path(os.environ.get("PD_LMEVAL_OUT", "acts/prompt_dom/lmeval")); out.mkdir(parents=True, exist_ok=True)   # sweep/mission_second_model.md: acts/second_model/<tag>/lmeval
res = lm_eval.simple_evaluate(
    model="local-chat-completions",
    model_args=dict(model=served, base_url="http://localhost:8000/v1/chat/completions", num_concurrent=32, max_retries=3, timeout=1800, tokenized_requests=False, max_gen_toks=1536),
    tasks=tasks, limit=limit, apply_chat_template=True, fewshot_as_multiturn=False,
    gen_kwargs=dict(temperature=0, max_gen_toks=1536, chat_template_kwargs={"enable_thinking": False}),   # reasoning off, the row's own template; **gen_kwargs lands in the request body
    log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
summary = {task: {k: v for k, v in r.items() if isinstance(v, (int, float))} for task, r in res["results"].items()}
(out / f"{served}.json").write_text(json.dumps(dict(served=served, limit=limit, tasks=tasks, results=res["results"], n_samples={t: len(s) for t, s in res.get("samples", {}).items()}, config=res.get("config"), samples={t: [dict(doc_id=x.get("doc_id"), target=x.get("target"), resps=[r[:400] if isinstance(r, str) else r for r in sum(x.get("resps", []), [])], filtered=x.get("filtered_resps"), metrics={k: v for k, v in x.items() if k in ("exact_match", "prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc")}) for x in s] for t, s in res.get("samples", {}).items()}), indent=1, default=str))
print(json.dumps(dict(served=served, limit=limit, summary=summary), indent=1))
