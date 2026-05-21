from __future__ import annotations

import re
from typing import Any


def sanitize_key(s: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9\-_:.@]', '_', s)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_').lower()
    return sanitized or 'unknown'


class WordNode:
    def __init__(self, lemma: str, pos: str = "", language: str = "fr",
                 frequency: int = 1, is_entity: bool = False, entity_label: str = ""):
        self._key = sanitize_key(lemma)
        self.lemma = lemma.lower()
        self.pos = pos
        self.language = language
        self.frequency = frequency
        self.is_entity = is_entity
        self.entity_label = entity_label

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "lemma": self.lemma,
            "pos": self.pos,
            "language": self.language,
            "frequency": self.frequency,
            "is_entity": self.is_entity,
            "entity_label": self.entity_label,
        }


class SentenceNode:
    def __init__(self, text: str, timestamp: float, session_id: str,
                 language: str = "fr", chunk_index: int = 0):
        import hashlib
        self._key = hashlib.md5(text.encode()).hexdigest()[:12]
        self.text = text
        self.timestamp = timestamp
        self.session_id = session_id
        self.language = language
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "text": self.text,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "language": self.language,
            "chunk_index": self.chunk_index,
        }


class TopicNode:
    def __init__(self, label: str, weight: float = 1.0, keywords: list[str] | None = None):
        import hashlib
        self._key = hashlib.md5(label.encode()).hexdigest()[:12]
        self.label = label
        self.weight = weight
        self.keywords = keywords or []

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "label": self.label,
            "weight": self.weight,
            "keywords": self.keywords,
        }


class DocumentNode:
    def __init__(self, filename: str, session_id: str = "",
                 title: str = "", author: str = "", pages: int = 0,
                 file_type: str = "", word_count: int = 0):
        import hashlib
        self._key = hashlib.md5(session_id.encode()).hexdigest()[:12]
        self.filename = filename
        self.session_id = session_id
        self.title = title
        self.author = author
        self.pages = pages
        self.file_type = file_type
        self.word_count = word_count

    def to_dict(self) -> dict:
        return {
            "_key": self._key,
            "filename": self.filename,
            "session_id": self.session_id,
            "title": self.title,
            "author": self.author,
            "pages": self.pages,
            "file_type": self.file_type,
            "word_count": self.word_count,
        }


class Edge:
    def __init__(self, collection: str, _from: str, _to: str,
                 relation_type: str, weight: float = 1.0, **extra):
        self.collection = collection
        self._from = _from
        self._to = _to
        self.relation_type = relation_type
        self.weight = weight
        self.extra = extra

    def to_dict(self) -> dict:
        d = {
            "_from": self._from,
            "_to": self._to,
            "relation_type": self.relation_type,
            "weight": self.weight,
        }
        d.update(self.extra)
        return d
