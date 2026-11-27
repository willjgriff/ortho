#!/usr/bin/env python3
"""sweep/mission_gemma_envs.md Phase 2, langdom: base Gemma (served cb_base, thinking off, greedy) translates heretic's 400 harmless train prompts into French.
Output acts/gemma_envs/langdom_french.json: [{i, en, fr, finish_reason}]. Usage: .venv/bin/python gemma_envs/translate.py"""
import json, asyncio
from pathlib import Path
from openai import AsyncOpenAI
from rhenvs.common import CONFIGS
import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
PROMPTS = json.loads((CONFIGS / "refusal" / "prompts.json").read_text())["harmless"][SPLIT][:400]
INSTR = "Translate the following text into French. Reply with the French translation only, nothing else.\n\n{text}"
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
sem = asyncio.Semaphore(32)
async def one(i, text):
    async with sem:
        r = await client.chat.completions.create(model="cb_base", messages=[{"role": "user", "content": INSTR.format(text=text)}], temperature=0.0, max_tokens=1024,
                                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return dict(i=i, en=text, fr=r.choices[0].message.content.strip(), finish_reason=r.choices[0].finish_reason)
async def main():
    out = await asyncio.gather(*[one(i, p) for i, p in enumerate(PROMPTS)])
    Path("acts/gemma_envs").mkdir(parents=True, exist_ok=True)
    Path("acts/gemma_envs/langdom_french.json" if SPLIT == "train" else f"acts/gemma_envs/langdom_french_{SPLIT}.json").write_text(json.dumps(dict(instruction=INSTR, model="cb_base (base, zero hook)", sampling="temperature 0, max_tokens 1024, enable_thinking false", items=out), indent=1, ensure_ascii=False))
    print(len(out), "translated;", sum(o["finish_reason"] != "stop" for o in out), "not stopped")
asyncio.run(main())
