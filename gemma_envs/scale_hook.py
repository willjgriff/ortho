#!/usr/bin/env python3
"""~/hooks/<name>.safetensors = ~/hooks/<row>.safetensors with the hook scaled by s (scripts/vllm_proj.py and second_model.py read `scale`:
h <- h - s((h.u) - c)u). Usage: .venv/bin/python gemma_envs/scale_hook.py <row> <s> <name>"""
import sys
from pathlib import Path
import torch as t
from safetensors.torch import load_file, save_file
row, s, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
H = Path.home() / "hooks"
h = load_file(H / f"{row}.safetensors")
save_file({k: v for k, v in h.items() if k != "scale"} | dict(scale=t.tensor(s)), H / f"{name}.safetensors")
print(f"{H / f'{name}.safetensors'}: {row} at scale {s}")
