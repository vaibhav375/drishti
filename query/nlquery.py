"""Natural-language catalogue query (roadmap feature 2).

The user types plain English ("Chinese payloads in graveyard orbit", "most
anomalous LEO objects"); the local LLM's ONLY job is to translate that into
a constrained, validated filter spec — it never answers the question or
invents data. Execution of the spec is 100% deterministic over the real
catalogue (see web/app.py::query). So the LLM cannot hallucinate results:
the worst it can do is produce a filter, which we validate and then run
against real rows.

This is the honest way to put an LLM in the query path — intent parsing,
not answer generation. The interpreted spec is always shown back to the
user (`describe`) so the translation is transparent, never a black box.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

REGIMES = {"LEO", "MEO", "GEO", "HEO", "GRAVEYARD"}
TYPES = {"PAY", "R/B", "DEB"}
SORTS = {"anomaly", "altitude_desc", "altitude_asc"}
MAX_LIMIT = 200

# Regime → SQL over SATCAT apogee/perigee (coarse, no propagation), reused
# from the catalogue browser's bands.
_REGIME_SQL = {
    "LEO":  "apogee_km < 2000",
    "MEO":  "perigee_km >= 2000 AND apogee_km < 34000",
    "GEO":  "apogee_km BETWEEN 34000 AND 38000 AND perigee_km > 33000",
    "HEO":  "apogee_km > 35000 AND perigee_km < 2000",
    "GRAVEYARD": "perigee_km > 35936",   # > GEO + ~150 km
}


@dataclass(frozen=True)
class QuerySpec:
    regime: Optional[str] = None
    object_type: Optional[str] = None
    country: Optional[str] = None
    name_contains: Optional[str] = None
    decayed: Optional[bool] = None
    min_inclination: Optional[float] = None
    max_inclination: Optional[float] = None
    sort: Optional[str] = None
    limit: int = 50

    def where(self) -> tuple[list[str], list]:
        """Deterministic SQL WHERE clauses + params over SATCAT."""
        clauses, params = [], []
        if self.decayed is True:
            clauses.append("decay_date IS NOT NULL")
        elif self.decayed is False or self.decayed is None:
            clauses.append("decay_date IS NULL")
        if self.regime in _REGIME_SQL:
            clauses.append(_REGIME_SQL[self.regime])
        if self.object_type in TYPES:
            clauses.append("object_type = ?"); params.append(self.object_type)
        if self.country:
            clauses.append("UPPER(country) = ?"); params.append(self.country.upper())
        if self.name_contains:
            clauses.append("name LIKE ?"); params.append(f"%{self.name_contains.upper()}%")
        if self.min_inclination is not None:
            clauses.append("inclination >= ?"); params.append(self.min_inclination)
        if self.max_inclination is not None:
            clauses.append("inclination <= ?"); params.append(self.max_inclination)
        return clauses, params

    def describe(self) -> str:
        parts = []
        if self.object_type:
            parts.append({"PAY": "payloads", "R/B": "rocket bodies", "DEB": "debris"}[self.object_type])
        else:
            parts.append("objects")
        if self.regime:
            parts.append(f"in {self.regime}")
        if self.country:
            parts.append(f"owned by {self.country.upper()}")
        if self.name_contains:
            parts.append(f"named like “{self.name_contains}”")
        if self.min_inclination is not None or self.max_inclination is not None:
            lo = self.min_inclination if self.min_inclination is not None else 0
            hi = self.max_inclination if self.max_inclination is not None else 180
            parts.append(f"inclination {lo:g}–{hi:g}°")
        if self.decayed is True:
            parts.append("(decayed / re-entered)")
        base = " ".join(parts)
        if self.sort == "anomaly":
            base += ", ranked by statistical anomaly"
        elif self.sort == "altitude_desc":
            base += ", highest altitude first"
        elif self.sort == "altitude_asc":
            base += ", lowest altitude first"
        return f"{base} · top {self.limit}"


SCHEMA_PROMPT = (
    "You translate a user's plain-English request about the satellite catalogue into a JSON "
    "filter. Output ONLY a JSON object, no prose. Allowed keys (omit any you don't need):\n"
    '  "regime": one of LEO, MEO, GEO, HEO, GRAVEYARD\n'
    '  "object_type": one of PAY (payload/satellite), R/B (rocket body), DEB (debris)\n'
    '  "country": an owner code, e.g. PRC (China), CIS (Russia), US, JPN, IND, ESA\n'
    '  "name_contains": a substring of the object name, e.g. COSMOS, STARLINK\n'
    '  "decayed": true for re-entered objects only\n'
    '  "min_inclination", "max_inclination": degrees\n'
    '  "sort": one of anomaly, altitude_desc, altitude_asc\n'
    '  "limit": integer up to 200\n'
    "Examples:\n"
    'Q: Chinese payloads in graveyard orbit -> {"regime":"GEO","object_type":"PAY","country":"PRC"}\n'
    'Q: most anomalous objects in low earth orbit -> {"regime":"LEO","sort":"anomaly","limit":30}\n'
    'Q: decayed Cosmos debris -> {"object_type":"DEB","name_contains":"COSMOS","decayed":true}\n'
    'Q: sun-synchronous satellites near 98 degrees -> {"object_type":"PAY","min_inclination":97,"max_inclination":100}\n'
)


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def _coerce_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def spec_from_json(d: dict) -> QuerySpec:
    """Validate/coerce a raw dict into a QuerySpec — unknown keys ignored,
    invalid values dropped, limit clamped. Pure; testable without the LLM."""
    regime = str(d.get("regime", "")).upper() or None
    otype = str(d.get("object_type", "")).upper().replace("RB", "R/B") or None
    sort = str(d.get("sort", "")).lower() or None
    lim = d.get("limit")
    try:
        lim = max(1, min(MAX_LIMIT, int(lim))) if lim is not None else 50
    except (TypeError, ValueError):
        lim = 50
    return QuerySpec(
        regime=regime if regime in REGIMES else None,
        object_type=otype if otype in TYPES else None,
        country=(str(d["country"]).strip() or None) if d.get("country") else None,
        name_contains=(str(d["name_contains"]).strip() or None) if d.get("name_contains") else None,
        decayed=_coerce_bool(d.get("decayed")),
        min_inclination=_coerce_float(d.get("min_inclination")),
        max_inclination=_coerce_float(d.get("max_inclination")),
        sort=sort if sort in SORTS else None,
        limit=lim,
    )


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_query(text: str, model_name: str = DEFAULT_MODEL) -> Optional[QuerySpec]:
    """LLM: plain English → validated QuerySpec. Returns None if the model
    produced no parseable JSON (caller falls back to a name search)."""
    from mlx_lm import generate as mlx_generate

    from report.generate import _load_model
    model, tokenizer = _load_model(model_name)
    messages = [
        {"role": "system", "content": SCHEMA_PROMPT},
        {"role": "user", "content": f"Q: {text} ->"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    d = _extract_json(raw)
    return spec_from_json(d) if d is not None else None
