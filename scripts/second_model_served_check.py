"""The served-vs-HF gate of sweep/mission_second_model.md (phase 2 step 1): teacher-forced log-probs of the stored base answers that `second_model.py served <row>` scored in HF
(acts/second_model/<tag>/served_hf_<row>.json, which carries the ids) through the served row on localhost:8000 (prompt_logprobs, every token), and the comparison. Run from /root/ortho:
    .venv/bin/python scripts/second_model_served_check.py score cb_<row> <tag>       -> acts/second_model/<tag>/served_<tag>.json
    .venv/bin/python scripts/second_model_served_check.py compare <tag a> <tag b>   -> mean |difference| per token; the floor is base's own served-vs-HF (compare base hf_base), tolerance 2 x floor
Tags: hf_<row> are the HF references (written by second_model.py served)."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

OUT = Path(f"acts/second_model/{os.environ.get('SM_MODEL', 'gemma4_31b')}")
URL = "http://localhost:8000"

def score(model: str, ids: list[int]) -> list[float]:
    r = requests.post(f"{URL}/v1/completions", json=dict(model=model, prompt=ids, max_tokens=1, temperature=0, prompt_logprobs=0), timeout=1800)
    assert r.status_code == 200, r.text[:500]
    lp = r.json()["choices"][0]["prompt_logprobs"]
    return [lp[j][str(ids[j])]["logprob"] for j in range(1, len(ids))]

if sys.argv[1] == "score":
    model, tag = sys.argv[2], sys.argv[3]
    ref = json.loads((OUT / f"served_hf_{sys.argv[4] if len(sys.argv) > 4 else 'base'}.json").read_text())   # the item list (ids) of an HF reference
    xs = ref["items"]
    with ThreadPoolExecutor(6) as ex: lps = list(ex.map(lambda x: score(model, x["ids"]), xs))
    (OUT / f"served_{tag}.json").write_text(json.dumps(dict(model=model, items=[dict(key=x["key"], env=x["env"], n=x["n"]) for x in xs], logps=lps)))
    print(f"{tag}: {len(xs)} rollouts, {sum(map(len, lps))} tokens scored on {model}; mean log-prob {np.mean(np.concatenate(lps)):.4f}")
elif sys.argv[1] == "compare":
    a, b = [json.loads((OUT / f"served_{tag}.json").read_text()) for tag in sys.argv[2:4]]
    assert [x["key"] for x in a["items"]] == [x["key"] for x in b["items"]]
    d = np.concatenate([np.array(p) - np.array(q) for p, q in zip(a["logps"], b["logps"])])
    per_env = {env: float(np.abs(np.concatenate([np.array(p) - np.array(q) for x, p, q in zip(a["items"], a["logps"], b["logps"]) if x["env"] == env])).mean()) for env in sorted({x["env"] for x in a["items"]})}
    floor_path = OUT / "served_compare_base_vs_hf_base.json"
    floor = json.loads(floor_path.read_text())["mean_abs"] if floor_path.exists() and sys.argv[2:4] != ["base", "hf_base"] else None
    res = dict(a=sys.argv[2], b=sys.argv[3], tokens=int(d.size), mean_abs=float(np.abs(d).mean()), signed=float(d.mean()), max_abs=float(np.abs(d).max()), per_env=per_env, floor=floor, tol=2 * floor if floor else None, passed=bool(np.abs(d).mean() <= 2 * floor) if floor else None)
    (OUT / f"served_compare_{sys.argv[2]}_vs_{sys.argv[3]}.json").write_text(json.dumps(res, indent=1))
    print(f"{sys.argv[2]} vs {sys.argv[3]}: {d.size} tokens, mean |diff| {res['mean_abs']:.5f} nats/token (signed {res['signed']:+.5f}, max {res['max_abs']:.3f}; per env {per_env})" + (f"; floor {floor:.5f}, tol {2 * floor:.5f}: {'PASS' if res['passed'] else 'FAIL'}" if floor else " (this is the floor)"))
else:
    print(__doc__)
