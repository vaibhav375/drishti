"""Local LLM backend shim — the ONE place a model is loaded and called.

Two interchangeable local backends, both free and offline once weights are
cached (no paid API, per project preference):

  * mlx          — Apple Silicon (the M1 dev machine). Default when mlx_lm
                   is importable. Model: mlx-community/Qwen2.5-3B-Instruct-4bit.
  * transformers — CPU fallback for Linux hosts (e.g. the Hugging Face Space
                   deployment, where MLX does not exist). Model:
                   Qwen/Qwen2.5-1.5B-Instruct — same model family, smaller so
                   CPU generation stays in the ~10-30 s range.

Override with DRISHTI_LLM_BACKEND=mlx|transformers|none and DRISHTI_HF_MODEL.
"none" (small hosts, e.g. the 512 MB Render deployment) makes chat() raise
LLMUnavailable; callers degrade honestly instead of faking output.
Decoding is greedy on both backends. Groundedness is NOT this module's job:
report/verify.py checks every output independently, whichever backend ran.
"""
from __future__ import annotations

import os
import threading

MLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
HF_MODEL = os.environ.get("DRISHTI_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

class LLMUnavailable(RuntimeError):
    """No local model on this host (DRISHTI_LLM_BACKEND=none)."""


_cache: dict[str, tuple] = {}
_lock = threading.Lock()  # one load / one generation at a time (CPU-bound anyway)


def backend() -> str:
    forced = os.environ.get("DRISHTI_LLM_BACKEND", "").strip().lower()
    if forced in ("mlx", "transformers", "none"):
        return forced
    try:
        import mlx_lm  # noqa: F401
        return "mlx"
    except ImportError:
        return "transformers"


def available() -> bool:
    return backend() != "none"


def model_label() -> str:
    return MLX_MODEL if backend() == "mlx" else HF_MODEL


def _load(name: str, kind: str):
    if name not in _cache:
        if kind == "mlx":
            from mlx_lm import load
            _cache[name] = load(name)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
            model.eval()
            _cache[name] = (model, tok)
    return _cache[name]


def chat(messages: list[dict], max_tokens: int = 300, model_name: str | None = None) -> str:
    """Run a chat-formatted prompt through the active local backend."""
    kind = backend()
    if kind == "none":
        raise LLMUnavailable("no local LLM on this host (DRISHTI_LLM_BACKEND=none)")
    with _lock:
        if kind == "mlx":
            from mlx_lm import generate as mlx_generate
            model, tok = _load(model_name or MLX_MODEL, "mlx")
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
            return mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)

        import torch
        model, tok = _load(HF_MODEL, "transformers")
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
