"""The model a mission runs on, chosen by tag (sweep/mission_second_model.md): PD_MODEL=<tag> in the environment, default the Qwen3.8-27B of every earlier mission. One place for what
differs between models: the HF id and revision, the decoder-layer count and width, the template's token ids and how thinking is switched, the layer windows (mapped from Qwen's 64
layers by depth fraction: R window every third layer from round(0.47 N) to round(0.94 N), Arditi candidates below round(0.8 N), every layer 0..N-1), the rhenvs configs per env and
setting, the HF loader and the vLLM serving flags. prompt_dom_rh.py, run.py and scripts/serve_row.sh read it."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    hf_id: str
    revision: str
    n_layers: int
    d_model: int
    think_open: int
    think_close: int
    newline: int
    im_start: int
    im_end: int
    off_tail: list[int]            # the ids a reasoning-off prompt ends with (the answer's first token follows); its last id is the "last prompt token"
    on_tail: list[int]             # the ids a reasoning-on generation prompt ends with
    off_kwargs: dict               # chat_template_kwargs for reasoning off
    on_kwargs: dict                # chat_template_kwargs for reasoning on (the TV config adds its own)
    configs: dict                  # (env, setting) -> rhenvs config name, env in grader / terminal_verifier / refusal, setting in off / on
    loader: str                    # transformers Auto class
    layers_attr: str               # dotted path from the model to the decoder-layer ModuleList
    hidden_attr: str               # dotted path from the config to hidden_size
    served_prefix: str             # run.py instance prefix for this model's runs ("" for Qwen: stage 11's names)
    vllm_bin: str
    vllm_args: list[str]
    proj_arch: str                 # scripts/vllm_proj.py's RHABLATION_PROJ_ARCH
    vectors_dir: str               # data/vectors/<vectors_dir>
    notes: str = ""

    @property
    def step(self) -> list[int]: return list(range(round(0.47 * self.n_layers), round(0.94 * self.n_layers) + 1, 3))
    @property
    def all(self) -> list[int]: return list(range(self.n_layers))
    @property
    def arditi_layers(self) -> list[int]: return list(range(round(0.8 * self.n_layers)))


MODELS = {
    "qwen38_27b": ModelSpec(tag="qwen38_27b", hf_id="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", n_layers=64, d_model=5120, think_open=248068, think_close=248069, newline=198, im_start=248045, im_end=248046,
                            off_tail=[248068, 271, 248069, 271], on_tail=[248068, 198], off_kwargs={"enable_thinking": False}, on_kwargs={"reasoning_effort": "low"},
                            configs={("grader", "off"): "nothink", ("grader", "on"): "qwen3.8_low", ("terminal_verifier", "on"): "qwen3.8_sweep", ("refusal", "off"): "nothink", ("refusal", "on"): "qwen3.8_low"},
                            loader="AutoModelForImageTextToText", layers_attr="model.language_model.layers", hidden_attr="text_config.hidden_size", served_prefix="", vllm_bin=os.path.expanduser("~/serve/bin/vllm"),
                            vllm_args=["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml", "--language-model-only"], proj_arch="qwen", vectors_dir="Qwen3.8-27B",
                            notes="stage 11's model; windows 30-60 by 3 / 0-50 / 0-63"),
    "nemotron35_30b": ModelSpec(tag="nemotron35_30b", hf_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9", n_layers=52, d_model=2688, think_open=12, think_close=13, newline=1010, im_start=10, im_end=11,
                                off_tail=[1010, 12, 13], on_tail=[12, 1010], off_kwargs={"enable_thinking": False}, on_kwargs={"enable_thinking": True},
                                configs={("grader", "off"): "nemotron35_30b_off", ("grader", "on"): "nemotron35_30b_on", ("terminal_verifier", "on"): "nemotron35_30b_on", ("terminal_verifier", "off"): "nemotron35_30b_off", ("refusal", "off"): "nemotron35_30b_off", ("refusal", "on"): "nemotron35_30b_on"},
                                loader="AutoModelForCausalLM", layers_attr="model.layers", hidden_attr="hidden_size", served_prefix="nm_", vllm_bin="/root/vllm-env/bin/vllm",
                                vllm_args=["--reasoning-parser", "nemotron_v3"], proj_arch="nemotron_h", vectors_dir="nemotron35_30b",
                                notes="hybrid Mamba-2 / MoE / attention, every block a layer (config layers_block_type); R window 24-48 by 3, Arditi layers 0-41, every layer 0-51; thinking by enable_thinking, budget by vLLM's thinking_token_budget"),
}


def current() -> ModelSpec:
    tag = os.environ.get("PD_MODEL", "qwen38_27b")
    assert tag in MODELS, f"PD_MODEL {tag} is one of {list(MODELS)}"
    return MODELS[tag]


def getattr_path(obj, path: str):
    for part in path.split("."): obj = getattr(obj, part)
    return obj


if __name__ == "__main__":
    for tag, m in MODELS.items():
        print(f"{tag}: {m.hf_id} @ {m.revision[:8]}, {m.n_layers} layers x {m.d_model}; R window {m.step}; Arditi layers 0-{m.arditi_layers[-1]}; configs {m.configs}")
