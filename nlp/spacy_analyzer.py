from __future__ import annotations

import spacy
from spacy.language import Language


class SpacyAnalyzer:
    def __init__(self):
        self._nlp_fr: Language | None = None
        self._nlp_en: Language | None = None

    def load(self, lang: str = "fr"):
        if lang == "fr":
            if self._nlp_fr is None:
                self._nlp_fr = spacy.load("fr_core_news_lg")
            return self._nlp_fr
        else:
            if self._nlp_en is None:
                self._nlp_en = spacy.load("en_core_web_lg")
            return self._nlp_en

    def analyze(self, text: str, lang: str = "fr") -> dict:
        nlp = self.load(lang)
        doc = nlp(text)

        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "shape": token.shape_,
            })

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        # Syntactic relations : sujet, verbe, objet
        relations = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubj:pass", "dobj", "iobj", "obj", "ROOT", "acl", "advcl"):
                relations.append({
                    "word": token.text,
                    "lemma": token.lemma_,
                    "dep": token.dep_,
                    "head_text": token.head.text,
                    "head_lemma": token.head.lemma_,
                    "head_pos": token.head.pos_,
                })

        return {
            "tokens": tokens,
            "entities": entities,
            "relations": relations,
            "sentences": [sent.text for sent in doc.sents],
        }
