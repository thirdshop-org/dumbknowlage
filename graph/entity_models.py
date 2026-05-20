from __future__ import annotations

from graph.models import sanitize_key


ENTITY_CONFIDENCE_LOW = 0.40
ENTITY_CONFIDENCE_HIGH = 0.75


class EntityBase:
    collection: str = ""

    def __init__(self, name: str, mentions: int = 1, confidence: float = 0.5, user_feedback: float = 0.0):
        self._key = sanitize_key(name)
        self.name = name
        self.mentions = mentions
        self.confidence = round(confidence, 2)
        self.user_feedback = user_feedback

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "name": self.name,
            "mentions": self.mentions,
            "confidence": self.confidence,
            "user_feedback": self.user_feedback,
        }

    @property
    def needs_review(self) -> bool:
        return ENTITY_CONFIDENCE_LOW <= self.confidence < ENTITY_CONFIDENCE_HIGH

    @property
    def is_valid(self) -> bool:
        return self.confidence >= ENTITY_CONFIDENCE_LOW


class PersonNode(EntityBase):
    collection = "Person"

    def __init__(self, name: str, title: str = "", mentions: int = 1, confidence: float = 0.5):
        super().__init__(name, mentions, confidence)
        self.title = title

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["title"] = self.title
        return d


class OrganizationNode(EntityBase):
    collection = "Organization"

    def __init__(self, name: str, domain: str = "", mentions: int = 1, confidence: float = 0.5):
        super().__init__(name, mentions, confidence)
        self.domain = domain

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["domain"] = self.domain
        return d


class LocationNode(EntityBase):
    collection = "Location"

    def __init__(self, name: str, loc_type: str = "", mentions: int = 1, confidence: float = 0.5):
        super().__init__(name, mentions, confidence)
        self.loc_type = loc_type

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["loc_type"] = self.loc_type
        return d


class EventNode(EntityBase):
    collection = "Event"

    def __init__(self, name: str, date: str = "", mentions: int = 1, confidence: float = 0.5):
        super().__init__(name, mentions, confidence)
        self.date = date

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["date"] = self.date
        return d


ENTITY_CLASSES = {
    "PER": PersonNode,
    "PERSON": PersonNode,
    "ORG": OrganizationNode,
    "ORGANIZATION": OrganizationNode,
    "LOC": LocationNode,
    "GPE": LocationNode,
    "EVENT": EventNode,
}


def entity_from_label(label: str, text: str,
                      doc_count: int = 1,
                      user_feedback: float = 0.0,
                      context_entities: int = 0,
                      active_rules: list[dict] | None = None) -> EntityBase | None:
    cls = ENTITY_CLASSES.get(label.upper())
    if cls is None:
        return None

    from graph.confidence import compute_confidence
    confidence = compute_confidence(
        name=text, label=label,
        doc_count=doc_count,
        user_feedback=user_feedback,
        context_entities=context_entities,
        active_rules=active_rules or [],
    )

    entity = cls(name=text, confidence=confidence, user_feedback=user_feedback)
    return entity
