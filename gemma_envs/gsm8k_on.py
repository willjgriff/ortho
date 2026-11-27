#!/usr/bin/env python3
"""sweep/mission_gemma_envs.md grid capability: GSM8K with thinking on, the first 250 test items, against the served row on localhost:8000. Thinking on (enable_thinking,
thinking_token_budget 4096, max_tokens 8192), greedy (temperature 0). Accuracy (the number after '####', else the last number of the answer), median reasoning chars,
and the share of answers whose reasoning reached the 4096 budget (usage reasoning_tokens >= 4096). Root (scores text only).
    .venv/bin/python gemma_envs/gsm8k_on.py <served name> [n]  ->  acts/gemma_envs/gsm8k_on/<served name>.json"""
import asyncio, json, re, statistics, sys
from pathlib import Path
from datasets import load_dataset
from openai import AsyncOpenAI

served, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 250
data = list(load_dataset("openai/gsm8k", "main", split="test"))[:n]
SUFFIX = "\n\nSolve the problem. Give the final answer as a number after '####'."
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none", timeout=3600)
sem = asyncio.Semaphore(64)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def pred(text: str):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    s = m.group(1) if m else (NUM.findall(text) or [None])[-1]
    if s is None: return None
    try: return float(s.replace(",", "").rstrip("."))
    except ValueError: return None

async def one(i, x):
    async with sem:
        r = await client.chat.completions.create(model=served, messages=[{"role": "user", "content": x["question"] + SUFFIX}], temperature=0.0, max_tokens=8192,
                                                 extra_body={"thinking_token_budget": 4096, "chat_template_kwargs": {"enable_thinking": True}})
        msg = r.choices[0].message
        content, reasoning = msg.content or "", getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        gold = float(x["answer"].split("####")[-1].strip().replace(",", ""))
        p = pred(content)
        rt = ((r.usage.completion_tokens_details.reasoning_tokens if r.usage and r.usage.completion_tokens_details else None) or 0)
        return dict(i=i, gold=gold, pred=p, correct=p is not None and abs(p - gold) < 1e-6, reasoning_chars=len(reasoning), reasoning_tokens=rt, finish=r.choices[0].finish_reason, answer=content[-300:])

async def main():
    out = await asyncio.gather(*[one(i, x) for i, x in enumerate(data)])
    acc = sum(o["correct"] for o in out) / len(out)
    res = dict(served=served, n=len(out), accuracy=acc, median_reasoning_chars=statistics.median(o["reasoning_chars"] for o in out),
               at_budget=sum(o["reasoning_tokens"] >= 4096 for o in out) / len(out), cut_at_max_tokens=sum(o["finish"] == "length" for o in out) / len(out),
               setting="thinking on, budget 4096, max_tokens 8192, temperature 0; prompt = question + " + json.dumps(SUFFIX), items=out)
    Path("acts/gemma_envs/gsm8k_on").mkdir(parents=True, exist_ok=True)
    Path(f"acts/gemma_envs/gsm8k_on/{served}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "items"}))
asyncio.run(main())
