"""Verifies the gemma_envs uploads (sweep/mission_gemma_envs.md): every file of /root/gemma_envs_out/stage/<what>/ must be in the public repo under second_model/<tag>/envs/<what>/ with the same size.
Run as root from /root/ortho (the token is read into this process only): .venv/bin/python scripts/gemma_envs_verify_upload.py [what ...]"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

TAG = os.environ.get("GE_TAG", os.environ.get("SM_MODEL", "gemma4_31b"))   # sweep/mission_model_pilots.md: any other tag stages under /root/mp_out/<tag>/stage, uploads under model_pilots/<tag>/
env = dict(line.strip().split("=", 1) for line in Path("/root/.config/rhablation/upload.env").read_text().splitlines() if "=" in line and not line.startswith("#"))
env = {k: v.strip().strip('"').strip("'") for k, v in env.items()}
api = HfApi(token=env["HF_TOKEN"])
GEMMA = TAG == "gemma4_31b"
STAGE = Path("/root/gemma_envs_out/stage") if GEMMA else Path(f"/root/mp_out/{TAG}/stage")
PRIVATE = Path("/nonexistent") if GEMMA else STAGE / "private"
DEST = f"second_model/{TAG}/envs" if GEMMA else f"model_pilots/{TAG}"
whats = sys.argv[1:] or sorted(p.name for p in STAGE.iterdir() if p.is_dir()) 

def remote(repo: str, prefix: str) -> dict[str, int]:
    return {f.path[len(prefix) + 1:]: f.size for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix, recursive=True) if hasattr(f, "size") and f.size is not None}

bad = 0
for what in whats:
    if what == "private": local_dir, repo, prefix = PRIVATE, env["HF_PRIVATE_REPO"], (f"second_model/{TAG}/private" if GEMMA else f"{DEST}/refusal")
    elif what == "report" and not GEMMA: local_dir, repo, prefix = STAGE / what, env["HF_UPLOAD_REPO"], "model_pilots/report"
    else: local_dir, repo, prefix = STAGE / what, env["HF_UPLOAD_REPO"], f"{DEST}/{what}"
    local = {str(p.relative_to(local_dir)): p.stat().st_size for p in local_dir.rglob("*") if p.is_file()} if local_dir.exists() else {}
    if not local: print(f"{what}: nothing staged"); continue
    try: rem = remote(repo, prefix)
    except Exception as e: print(f"{what}: cannot list {repo}/{prefix}: {e}"); bad += 1; continue
    missing = sorted(k for k in local if k not in rem)
    wrong = sorted(k for k in local if k in rem and rem[k] != local[k])
    print(f"{what}: {len(local)} local files, {len(rem)} remote under {prefix} ({sum(rem.values()) / 1e6:.1f} MB); missing {len(missing)}, size mismatch {len(wrong)}" + ("  OK" if not missing and not wrong else "  **FAIL**"))
    for k in missing[:5]: print(f"  missing: {k}")
    for k in wrong[:5]: print(f"  size: {k} local {local[k]} remote {rem[k]}")
    bad += bool(missing or wrong)
print("VERIFY_OK" if not bad else f"VERIFY_FAILED {bad}")
