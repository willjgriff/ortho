"""Phase 0 of sweep/mission_prompt_dom.md, the equivalence gate: teacher-forced log-probs of 30 stored base rollouts (10 TV, 10 Grader, 10 refusal-harmless, from the records) through the
served row on localhost:8000, every token of the stored ids (prompt_logprobs). score <served name> <tag> saves acts/prompt_dom/served_<tag>.json; compare <tag a> <tag b> prints the mean
|difference| per token against the tolerance (2 x the bf16 floor of kto_orthog.py's base_w, 0.011577 nats/token). Also reads the server's /metrics for the spec-decoding acceptance.
    python scripts/prompt_dom_served_check.py score cb_gprand final
    python scripts/prompt_dom_served_check.py score cb_gprand plain
    python scripts/prompt_dom_served_check.py compare final plain"""
import json
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

OUT = Path("acts/prompt_dom")
FLOOR = 0.011576792751558266   # base_w's served-vs-HF mean |diff| per token (kto_orthog, part 2)
TOL = 2 * FLOOR
N_EACH = 10
URL = "http://localhost:8000"

def pick(env: str, keep) -> list[dict]:
    rs = [r for p in sorted((Path("data/inspect") / env).glob("*.jsonl")) for r in map(json.loads, p.open()) if r["model"] == "vllm/cb_base" and r["condition"] == "none" and r.get("ids") and keep(r)]
    rs = sorted(rs, key=lambda r: zlib.crc32(r["rollout_id"].encode()))[:N_EACH]
    assert len(rs) == N_EACH, (env, len(rs))
    return [dict(key=r["rollout_id"], env=env, ids=r["ids"]) for r in rs]

def items() -> list[dict]:
    return pick("terminal_verifier", lambda r: r["config_id"] == "qwen3.8_sweep" and r["mode"] == "verifier_bug" and len(r["ids"]) <= 12000) + pick("grader", lambda r: r["config_id"] == "qwen3.8_low" and r["labels"]["prompts"] == "hack") + pick("refusal", lambda r: r["config_id"] == "nothink" and r["labels"]["prompts"] == "harmless" and r["labels"]["split"] == "test")

def score(model: str, x: dict) -> list[float]:
    r = requests.post(f"{URL}/v1/completions", json=dict(model=model, prompt=x["ids"], max_tokens=1, temperature=0, prompt_logprobs=0), timeout=1800)
    assert r.status_code == 200, r.text[:500]
    lp = r.json()["choices"][0]["prompt_logprobs"]
    return [lp[j][str(x["ids"][j])]["logprob"] for j in range(1, len(x["ids"]))]

def metrics() -> dict:
    text = requests.get(f"{URL}/metrics", timeout=10).text
    out = {}
    for key in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_drafts_total", "vllm:kv_cache_usage_perc", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
        m = re.findall(rf"^{re.escape(key)}(?:\{{[^}}]*\}})? (\S+)$", text, re.M)
        if m: out[key] = sum(float(v) for v in m)
    if out.get("vllm:spec_decode_num_draft_tokens_total"): out["spec_acceptance_per_draft_token"] = out["vllm:spec_decode_num_accepted_tokens_total"] / out["vllm:spec_decode_num_draft_tokens_total"]
    if out.get("vllm:spec_decode_num_drafts_total"): out["accepted_tokens_per_draft"] = out["vllm:spec_decode_num_accepted_tokens_total"] / out["vllm:spec_decode_num_drafts_total"]
    return out

if sys.argv[1] == "score":
    model, tag = sys.argv[2], sys.argv[3]
    xs = items()
    with ThreadPoolExecutor(6) as ex: lps = list(ex.map(lambda x: score(model, x), xs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"served_{tag}.json").write_text(json.dumps(dict(model=model, items=[dict(key=x["key"], env=x["env"], n=len(x["ids"])) for x in xs], logps=lps, metrics=metrics())))
    print(f"{tag}: {len(xs)} rollouts, {sum(map(len, lps))} tokens scored on {model}; mean log-prob {np.mean(np.concatenate(lps)):.4f}; metrics {metrics()}")
elif sys.argv[1] == "compare":
    a, b = [json.loads((OUT / f"served_{tag}.json").read_text()) for tag in sys.argv[2:4]]
    assert [x["key"] for x in a["items"]] == [x["key"] for x in b["items"]]
    d = np.concatenate([np.array(p) - np.array(q) for p, q in zip(a["logps"], b["logps"])])
    per_env = {env: float(np.abs(np.concatenate([np.array(p) - np.array(q) for x, p, q in zip(a["items"], a["logps"], b["logps"]) if x["env"] == env])).mean()) for env in ("terminal_verifier", "grader", "refusal")}
    res = dict(a=sys.argv[2], b=sys.argv[3], tokens=int(d.size), mean_abs=float(np.abs(d).mean()), signed=float(d.mean()), max_abs=float(np.abs(d).max()), per_env=per_env, floor=FLOOR, tol=TOL, passed=bool(np.abs(d).mean() <= TOL))
    (OUT / f"served_compare_{sys.argv[2]}_vs_{sys.argv[3]}.json").write_text(json.dumps(res, indent=1))
    print(f"{sys.argv[2]} vs {sys.argv[3]}: {d.size} tokens, mean |diff| {res['mean_abs']:.5f} nats/token (signed {res['signed']:+.5f}, max {res['max_abs']:.3f}; per env {per_env}); tol {TOL:.4f}: {'PASS' if res['passed'] else 'FAIL'}")
else:
    print(__doc__)
