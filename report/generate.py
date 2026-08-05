"""Grounded report generation (F4.1).

Per handoff §5/§7: feed structured detections to a local LLM that writes
analyst narrative. Kept fully local (MLX, a small quantized model) per
explicit user preference — no paid API, and it matches the project's own
"runs offline on a laptop" claim. Confirmed viable on the actual target
hardware (8 GB RAM, M1 MacBook Air — this session runs natively on it,
not a remote sandbox): a 3B-class 4-bit model (~1.6 GB on disk) loads
and generates without issue.

This module does NOT itself guarantee groundedness — `report/verify.py`
checks the output afterward, independent of this module (§9: never trust
an LLM's own claim about its output; verify separately). Generation
refuses outright on an empty facts payload rather than ever prompting
the model with nothing to ground it in.
"""
from __future__ import annotations

from report.facts import ReportFacts

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

SYSTEM_PROMPT = (
    "You are a space domain awareness analyst writing a short, factual "
    "situation report. Use ONLY the facts listed below the object header "
    "— do not invent numbers, dates, or claims not explicitly present in "
    "them. Every quantitative statement in your report must directly "
    "restate a number from the facts. "
    "CRITICAL: do NOT characterize how threatening, significant, concerning, "
    "or severe the object is in your own words. If a 'Threat level' fact is "
    "given (low / moderate / high), you may state exactly that level and "
    "nothing stronger — never call a low or moderate object 'significant', "
    "'notable', 'high concern', or similar. A low threat level means the "
    "object is unremarkable; say so plainly. "
    "Write 3-5 plain-prose sentences, no bullet points, no speculation "
    "beyond what the facts support, no operational recommendations (this is "
    "a decision-support summary, not a targeting or action directive)."
)

_model_cache: dict[str, tuple] = {}


def _load_model(model_name: str = DEFAULT_MODEL):
    if model_name not in _model_cache:
        from mlx_lm import load
        _model_cache[model_name] = load(model_name)
    return _model_cache[model_name]


def generate_report(
    facts: ReportFacts,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 300,
) -> str:
    if not facts.facts:
        raise ValueError("no facts provided — refuse to generate an ungrounded report")

    from mlx_lm import generate as mlx_generate

    model, tokenizer = _load_model(model_name)
    user_prompt = (
        f"Object: {facts.object_name or facts.norad_id} (NORAD {facts.norad_id})\n\n"
        f"Facts:\n{facts.as_prompt_block()}\n\nWrite the report."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _strip_echoed_facts(text: str) -> str:
    """The small local model sometimes echoes the bulleted facts block back
    before writing prose. Drop the bullet lines and the facts header, keep
    only the narrative — otherwise the echoed block (esp. the ISO timestamp)
    pollutes verification with spurious numbers."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("-") or s.lower().startswith("watch cycle findings"):
            continue
        kept.append(s)
    return " ".join(kept).strip()


WATCH_SYSTEM_PROMPT = (
    "You are a space domain awareness analyst writing a short watch-cycle "
    "brief — a summary of what changed since the last cycle. Use ONLY the "
    "facts listed; do not invent numbers, objects, or events. Every "
    "quantitative statement must restate a number from the facts. Do NOT "
    "add severity or alarm language beyond the significance labels given "
    "(priority / elevated / routine). Open with the counts, then summarize "
    "the notable findings in 2-4 plain sentences. No recommendations, no "
    "speculation — this is a factual change summary, not a directive."
)


def generate_watch_brief(finding_dicts: list[dict], run_at: str,
                         model_name: str = DEFAULT_MODEL, max_tokens: int = 260) -> str:
    """Grounded narrative for a watch cycle. `finding_dicts`:
    {'summary','significance'}. Returns "" if there are no findings."""
    if not finding_dicts:
        return ""
    from mlx_lm import generate as mlx_generate

    from report.facts import watch_facts
    facts = watch_facts(finding_dicts, run_at)
    model, tokenizer = _load_model(model_name)
    user_prompt = f"Watch cycle findings:\n{facts.as_prompt_block()}\n\nWrite the brief."
    messages = [
        {"role": "system", "content": WATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return _strip_echoed_facts(raw)
