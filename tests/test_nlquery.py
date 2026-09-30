"""Tests for natural-language query parsing (query/nlquery.py).

Offline — covers `spec_from_json` (validation/coercion) and the
deterministic SQL/describe, independent of the LLM. The LLM step
(parse_query) is validated live against real English queries.
"""
from __future__ import annotations

from query.nlquery import QuerySpec, spec_from_json


def test_valid_fields_parsed():
    s = spec_from_json({"regime": "geo", "object_type": "PAY", "country": "prc", "limit": 30})
    assert s.regime == "GEO"
    assert s.object_type == "PAY"
    assert s.country == "prc"
    assert s.limit == 30


def test_invalid_values_dropped():
    s = spec_from_json({"regime": "MOON", "object_type": "SPACESHIP", "sort": "vibes"})
    assert s.regime is None
    assert s.object_type is None
    assert s.sort is None


def test_unknown_keys_ignored_and_limit_clamped():
    s = spec_from_json({"colour": "blue", "limit": 99999})
    assert s.limit == 200  # clamped to MAX_LIMIT


def test_rb_normalized():
    assert spec_from_json({"object_type": "RB"}).object_type == "R/B"


def test_decayed_coercion():
    assert spec_from_json({"decayed": "true"}).decayed is True
    assert spec_from_json({"decayed": False}).decayed is False


def test_where_builds_deterministic_sql():
    s = QuerySpec(regime="GEO", object_type="PAY", country="PRC", decayed=False)
    clauses, params = s.where()
    assert "decay_date IS NULL" in clauses
    assert "object_type = ?" in clauses and "PRC" in params
    assert any("apogee_km BETWEEN 34000 AND 38000" in c for c in clauses)


def test_where_decayed_true_flips_clause():
    clauses, _ = QuerySpec(decayed=True).where()
    assert "decay_date IS NOT NULL" in clauses
    assert "decay_date IS NULL" not in clauses


def test_inclination_range_in_where():
    s = QuerySpec(min_inclination=97, max_inclination=100)
    clauses, params = s.where()
    assert "inclination >= ?" in clauses and 97 in params
    assert "inclination <= ?" in clauses and 100 in params


def test_describe_is_human_readable():
    s = QuerySpec(regime="LEO", object_type="PAY", sort="anomaly", limit=30)
    d = s.describe()
    assert "payloads" in d and "LEO" in d and "anomaly" in d and "top 30" in d
