from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

from arango.collection import StandardCollection
from arango.database import StandardDatabase


class CorrectionStore:
    def __init__(self, db: StandardDatabase):
        self._db = db
        self._ensure_collections()

    def _ensure_collections(self):
        for name in ["EntityCorrection", "LearnedRule"]:
            if not self._db.has_collection(name):
                self._db.create_collection(name)

    @property
    def corrections(self) -> StandardCollection:
        return self._db.collection("EntityCorrection")

    @property
    def rules(self) -> StandardCollection:
        return self._db.collection("LearnedRule")

    def log_correction(self, original_text: str, entity_type: str,
                       action: str, reason: str = "", source: str = "",
                       applied_rules: list[str] | None = None):
        key = f"corr_{uuid.uuid4().hex[:12]}"
        self.corrections.insert({
            "_key": key,
            "original_text": original_text,
            "entity_type": entity_type,
            "action": action,
            "reason": reason,
            "source": source,
            "applied_rules": applied_rules or [],
            "timestamp": time.time(),
        }, overwrite=True)
        return key

    def get_recent_corrections(self, limit: int = 50,
                               action: str | None = None) -> list[dict]:
        filters = []
        if action:
            filters.append(f'FILTER c.action == "{action}"')
        aql = f"""
        FOR c IN EntityCorrection
            {''.join(filters)}
            SORT c.timestamp DESC
            LIMIT {limit}
            RETURN c
        """
        cursor = self._db.aql.execute(aql)
        return [doc for doc in cursor]

    def get_entity_feedback(self, entity_key: str) -> dict | None:
        aql = """
        FOR c IN EntityCorrection
            FILTER c.original_text == @key
            SORT c.timestamp DESC
            LIMIT 1
            RETURN c
        """
        cursor = self._db.aql.execute(aql, bind_vars={"key": entity_key})
        for doc in cursor:
            return doc
        return None

    def get_rules(self, auto_apply_only: bool = False) -> list[dict]:
        filters = ["FILTER r.auto_apply == true"] if auto_apply_only else []
        aql = f"""
        FOR r IN LearnedRule
            {''.join(filters)}
            SORT r.samples DESC
            RETURN r
        """
        cursor = self._db.aql.execute(aql)
        return [doc for doc in cursor]

    def upsert_rule(self, pattern_type: str, entity_label: str,
                    samples: int, rejection_rate: float) -> str:
        key = f"rule_{pattern_type}_{entity_label.lower()}"
        existing = self.rules.get(key)
        if existing:
            new_auto = rejection_rate > 0.7 and (existing.get("samples", 0) + samples >= 8)
            self.rules.update_match(
                {"_key": key},
                {
                    "samples": existing.get("samples", 0) + samples,
                    "rejection_rate": rejection_rate,
                    "auto_apply": new_auto,
                    "last_applied": time.time(),
                },
            )
        else:
            auto = rejection_rate > 0.7 and samples >= 8
            self.rules.insert({
                "_key": key,
                "pattern_type": pattern_type,
                "entity_label": entity_label,
                "samples": samples,
                "rejection_rate": rejection_rate,
                "auto_apply": auto,
                "created": time.time(),
                "last_applied": time.time(),
            }, overwrite=True)
        return key

    def get_matching_rules(self, name: str, label: str) -> list[dict]:
        rules = self.get_rules(auto_apply_only=True)
        from graph.confidence import _matches_rule
        return [r for r in rules if _matches_rule(name, label, r)]

    def delete_rule(self, rule_key: str):
        self.rules.delete(rule_key, ignore_missing=True)

    def get_correction_stats(self) -> dict:
        aql = """
        FOR c IN EntityCorrection
            COLLECT action = c.action WITH COUNT INTO count
            RETURN {action, count}
        """
        cursor = self._db.aql.execute(aql)
        actions = {doc["action"]: doc["count"] for doc in cursor}
        return {
            "total": sum(actions.values()),
            "confirmed": actions.get("confirmed", 0),
            "denied": actions.get("denied", 0),
            "renamed": actions.get("renamed", 0),
            "auto_denied": actions.get("auto_denied", 0),
            "rules": len(self.get_rules()),
        }

    def get_aggregated_denied(self) -> list[dict]:
        """Aggregate all denied corrections by (original_text, entity_type)."""
        aql = """
        FOR c IN EntityCorrection
            FILTER c.action == "denied"
            COLLECT text = c.original_text, type = c.entity_type
            AGGREGATE count = COUNT(1)
            SORT count DESC
            RETURN {original_text: text, entity_type: type, samples: count}
        """
        cursor = self._db.aql.execute(aql)
        return [doc for doc in cursor]
