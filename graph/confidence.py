from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


SIGNAL_WEIGHTS = {
    "ner_consistency": 0.20,
    "doc_frequency": 0.15,
    "user_feedback": 0.30,
    "name_structure": 0.20,
    "context_coherence": 0.15,
}

LOWERCASE_BLACKLIST: set[str] = set()
UPPERCASE_BLACKLIST: set[str] = set()


def compute_name_structure_score(name: str, label: str) -> float:
    score = 0.5
    if len(name) < 2 or len(name) > 50:
        score -= 0.3
    if any(c in name for c in "{}|=$/\\<>"):
        score -= 0.4
    if any(c.isdigit() for c in name) and label in ("PER", "ORG", "LOC"):
        score -= 0.3
    if name[0].isupper() and len(name) > 2:
        score += 0.4
    elif name[0].islower() and len(name) > 3:
        score -= 0.3
    if all(c.isalpha() or c in "- _.':" for c in name) and len(name) > 2:
        score += 0.2
    if label == "LOC" and name[0].isupper():
        score += 0.2
    if label == "ORG" and any(w[0].isupper() for w in name.split()):
        score += 0.2
    return max(0.0, min(1.0, score))


def compute_confidence(
    name: str,
    label: str,
    doc_count: int = 1,
    user_feedback: float = 0.0,
    context_entities: int = 0,
    active_rules: list[dict] | None = None,
) -> float:
    if doc_count > 1:
        ner_score = min(1.0, 0.3 + (doc_count - 1) * 0.1)
    else:
        ner_score = 0.3

    doc_freq_score = min(1.0, doc_count * 0.15)

    structure_score = compute_name_structure_score(name, label)

    context_score = min(1.0, context_entities * 0.1)

    # Apply learned rules
    rule_penalty = 0.0
    if active_rules:
        for rule in active_rules:
            if _matches_rule(name, label, rule):
                if rule.get("rejection_rate", 0) > 0.7 and rule.get("auto_apply"):
                    rule_penalty = max(rule_penalty, rule["rejection_rate"] * 0.4)

    total = (
        SIGNAL_WEIGHTS["ner_consistency"] * ner_score
        + SIGNAL_WEIGHTS["doc_frequency"] * doc_freq_score
        + SIGNAL_WEIGHTS["user_feedback"] * user_feedback
        + SIGNAL_WEIGHTS["name_structure"] * structure_score
        + SIGNAL_WEIGHTS["context_coherence"] * context_score
        - rule_penalty
    )

    return max(0.0, min(1.0, total))


def _matches_rule(name: str, label: str, rule: dict) -> bool:
    pattern = rule.get("pattern_type", "")
    if pattern == "lowercase_word_over_3_chars" and label == rule.get("entity_label"):
        return name[0].islower() and len(name) > 3
    if pattern == "short_word" and label == rule.get("entity_label"):
        return 1 < len(name) <= 3
    if pattern == "contains_special_chars":
        return any(c in name for c in "{}|=$/\\<>")
    if pattern == "has_digits" and label in ("PER", "ORG"):
        return any(c.isdigit() for c in name)
    return False


def analyze_pattern(corrections: list[dict]) -> dict | None:
    if len(corrections) < 5:
        return None

    total = len(corrections)
    total_weighted = sum(c.get("samples", 1) for c in corrections)

    lowercase_long = [c for c in corrections
                      if c.get("original_text", "")[0].islower()
                      and len(c.get("original_text", "")) > 3]
    ll_weight = sum(c.get("samples", 1) for c in lowercase_long)
    if len(lowercase_long) >= max(3, total * 0.5):
        return {
            "pattern_type": "lowercase_word_over_3_chars",
            "entity_label": _most_common_label(corrections),
            "samples": ll_weight,
            "rejection_rate": ll_weight / total_weighted if total_weighted else 1.0,
        }

    short_words = [c for c in corrections
                   if 1 < len(c.get("original_text", "")) <= 3]
    sw_weight = sum(c.get("samples", 1) for c in short_words)
    if len(short_words) >= max(3, total * 0.5):
        return {
            "pattern_type": "short_word",
            "entity_label": _most_common_label(corrections),
            "samples": sw_weight,
            "rejection_rate": sw_weight / total_weighted if total_weighted else 1.0,
        }

    special_chars = [c for c in corrections
                     if any(ch in c.get("original_text", "")
                            for ch in "{}|=$/\\<>")]
    sc_weight = sum(c.get("samples", 1) for c in special_chars)
    if special_chars:
        return {
            "pattern_type": "contains_special_chars",
            "entity_label": _most_common_label(corrections),
            "samples": sc_weight,
            "rejection_rate": sc_weight / total_weighted if total_weighted else 1.0,
        }

    return None


def _most_common_label(corrections: list[dict]) -> str:
    from collections import Counter
    labels = [c.get("entity_type", c.get("entity_label", "PER")) for c in corrections]
    return Counter(labels).most_common(1)[0][0]
