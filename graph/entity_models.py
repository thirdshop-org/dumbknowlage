from __future__ import annotations

from graph.models import sanitize_key


class EntityBase:
    collection: str = ""

    def __init__(self, name: str, mentions: int = 1):
        self._key = sanitize_key(name)
        self.name = name
        self.mentions = mentions

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "name": self.name,
            "mentions": self.mentions,
        }


class PersonNode(EntityBase):
    collection = "Person"

    def __init__(self, name: str, title: str = "", mentions: int = 1):
        super().__init__(name, mentions)
        self.title = title

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["title"] = self.title
        return d


class OrganizationNode(EntityBase):
    collection = "Organization"

    def __init__(self, name: str, domain: str = "", mentions: int = 1):
        super().__init__(name, mentions)
        self.domain = domain

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["domain"] = self.domain
        return d


class LocationNode(EntityBase):
    collection = "Location"

    def __init__(self, name: str, loc_type: str = "", mentions: int = 1):
        super().__init__(name, mentions)
        self.loc_type = loc_type

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["loc_type"] = self.loc_type
        return d


class EventNode(EntityBase):
    collection = "Event"

    def __init__(self, name: str, date: str = "", mentions: int = 1):
        super().__init__(name, mentions)
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


def entity_from_label(label: str, text: str) -> EntityBase | None:
    cls = ENTITY_CLASSES.get(label.upper())
    if cls is None:
        return None
    return cls(name=text)
