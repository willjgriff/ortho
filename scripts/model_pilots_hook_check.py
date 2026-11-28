#!/usr/bin/env python3
"""sweep/mission_model_pilots.md hook check: 5 fixed prompts, greedy (temperature 0), 48 tokens, sent one at a time to the served model through
/v1/chat/completions with the model's thinking kwargs, plus the top-5 logprobs of the first 8 tokens. Saves one JSON per served row; `compare`
prints whether two rows give the same text (and the max first-8-token logprob difference). Stdlib only.
    model_pilots_hook_check.py save <served name> <out.json> '<chat_template_kwargs json>' [extra_body json]
    model_pilots_hook_check.py compare <a.json> <b.json>"""
import json, sys, urllib.request

PROMPTS = ["What is the capital of Australia? Answer in one sentence.", "Write a Python function that returns the n-th Fibonacci number.",
           "Explain in two sentences why the sky is blue.", "List three prime numbers greater than 50.", "Translate 'good morning, how are you?' into French."]

def chat(model, prompt, kwargs, extra):
    body = dict(model=model, messages=[dict(role="user", content=prompt)], temperature=0, max_tokens=48, logprobs=True, top_logprobs=5, chat_template_kwargs=kwargs, **extra)
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    m = r["message"]
    lp = [[t["token"], t["logprob"]] for t in (r.get("logprobs") or {}).get("content", [])[:8]]
    return dict(prompt=prompt, reasoning=m.get("reasoning_content") or m.get("reasoning"), content=m.get("content"), first_logprobs=lp)

if sys.argv[1] == "save":
    model, out, kwargs = sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    res = [chat(model, p, kwargs, extra) for p in PROMPTS]
    json.dump(dict(model=model, kwargs=kwargs, extra=extra, results=res), open(out, "w"), indent=1)
    for r in res: print(json.dumps(r)[:400])
else:
    a, b = (json.load(open(x))["results"] for x in sys.argv[2:4])
    same = 0
    for x, y in zip(a, b):
        tx, ty = (x["reasoning"] or "") + "|" + (x["content"] or ""), (y["reasoning"] or "") + "|" + (y["content"] or "")
        d = max([abs(p[1] - q[1]) for p, q in zip(x["first_logprobs"], y["first_logprobs"]) if p[0] == q[0]] or [float("nan")])
        same += tx == ty
        print(f"{'SAME' if tx == ty else 'DIFF'}  max|dlogprob| first 8 (same tokens) {d:.4f}  {x['prompt'][:40]!r}\n   a: {tx[:160]!r}\n   b: {ty[:160]!r}")
    print(f"HOOKCHECK {sys.argv[2]} vs {sys.argv[3]}: {same}/{len(a)} identical")
