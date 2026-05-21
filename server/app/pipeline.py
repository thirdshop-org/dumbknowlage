from __future__ import annotations

from typing import Any

from config import config
from storage.sqlite_store import SQLiteStore


def run_nlp_pipeline(
    store: SQLiteStore,
    session_id: str,
    chunks: list[dict],
    lang: str,
    build_graph: bool,
    doc_metadata: dict | None = None,
):
    from nlp.spacy_analyzer import SpacyAnalyzer
    from nlp.camembert_analyzer import CamembertAnalyzer
    from nlp.extractor import (
        extract_frequencies,
        extract_co_occurrences,
        compute_tfidf,
        detect_burst_topics,
    )
    from rag.indexer import index_session

    full_text = " ".join(c["text"] for c in chunks)

    # spaCy
    spacy_analyzer = SpacyAnalyzer()
    spacy_result = spacy_analyzer.analyze(full_text, lang=lang)
    store.save_analysis(session_id, "spacy", spacy_result)

    # Frequencies + co-occurrences
    frequencies = extract_frequencies(spacy_result["tokens"])
    co_occurrences = extract_co_occurrences(spacy_result["tokens"])
    store.save_analysis(session_id, "frequencies", {"frequencies": frequencies})
    store.save_analysis(session_id, "co_occurrences", {"co_occurrences": co_occurrences})

    # TF-IDF
    chunk_texts = [c["text"] for c in chunks]
    tfidf_result = compute_tfidf(chunk_texts)
    store.save_analysis(session_id, "tfidf", {"tfidf": tfidf_result})

    # Burst topics
    burst_topics = detect_burst_topics(chunk_texts)
    store.save_analysis(session_id, "burst_topics", {"burst_topics": burst_topics})

    # CamemBERT
    camembert = CamembertAnalyzer()
    topics = camembert.extract_topics(full_text)
    store.save_analysis(session_id, "camembert_topics", {"topics": topics})

    # Graph
    if build_graph:
        build_graph_from_analysis(
            session_id, spacy_result, chunks,
            co_occurrences, topics, doc_metadata,
        )

    # ChromaDB index
    try:
        index_session(session_id)
    except Exception:
        pass


def build_graph_from_analysis(
    session_id: str,
    spacy_result: dict[str, Any],
    chunks: list[dict],
    co_occurrences: list[tuple[tuple[str, str], int]],
    topics: list[dict],
    doc_metadata: dict | None = None,
):
    from graph.arango_client import GraphManager
    from graph.entity_models import entity_from_label
    from graph.models import DocumentNode, SentenceNode, TopicNode, WordNode, sanitize_key

    gm = GraphManager()
    if not gm.connect():
        return

    # Document node
    if doc_metadata:
        doc_node = DocumentNode(
            filename=doc_metadata.get("filename", f"session_{session_id}"),
            session_id=session_id,
            title=doc_metadata.get("title", ""),
            author=doc_metadata.get("author", ""),
            pages=doc_metadata.get("pages", 0),
            file_type=doc_metadata.get("file_type", ""),
            word_count=doc_metadata.get("word_count", 0),
        )
    else:
        doc_node = DocumentNode(
            filename=f"session_{session_id}",
            session_id=session_id,
        )
    gm.insert_document(doc_node)

    # Words
    word_keys = {}
    for t in spacy_result["tokens"]:
        if t["is_punct"] or t["is_stop"]:
            continue
        is_entity = any(e["text"].lower() == t["text"].lower() for e in spacy_result["entities"])
        entity_label = ""
        if is_entity:
            for e in spacy_result["entities"]:
                if e["text"].lower() == t["text"].lower():
                    entity_label = e["label"]
                    break
        w = WordNode(
            lemma=t["lemma"],
            pos=t["pos"],
            language="fr",
            is_entity=is_entity,
            entity_label=entity_label,
        )
        gm.upsert_word(w)
        word_keys[t["lemma"]] = w._key

    # Dependency relations
    for rel in spacy_result.get("relations", []):
        if rel["lemma"] in word_keys and rel["head_lemma"] in word_keys:
            gm.create_dependency(rel["lemma"], rel["head_lemma"], rel["dep"])

    # Co-occurrences
    for (a, b), count in co_occurrences:
        gm.create_co_occurrence(a, b, count)

    # Typed entities
    entity_map: dict[str, tuple[str, str]] = {}

    active_rules: list[dict] = []
    try:
        store = gm.get_correction_store()
        active_rules = store.get_rules(auto_apply_only=True)
    except Exception:
        pass

    for ent in spacy_result["entities"]:
        entity = entity_from_label(ent["label"], ent["text"],
                                   active_rules=active_rules)
        if entity is None:
            continue
        gm.upsert_entity(entity)
        gm.create_appears_in(entity.collection, entity._key, doc_node._key)
        entity_map[ent["text"].lower()] = (entity.collection, entity._key)
        word_key = sanitize_key(ent["text"])
        if word_key in word_keys:
            gm.create_entity_edge(
                "Word", word_key,
                entity.collection, entity._key,
                gm.cfg.edge_is_similar,
                "IS_ENTITY",
            )

    # Infer entity relations from dependencies
    for rel in spacy_result.get("relations", []):
        word_lower = rel["lemma"]
        head_lower = rel["head_lemma"]
        w_info = entity_map.get(word_lower)
        h_info = entity_map.get(head_lower)
        if w_info and h_info:
            w_type, w_key = w_info
            h_type, h_key = h_info
            if w_type == "Person" and h_type == "Organization":
                gm.create_works_for(w_key, h_key)
            elif w_type == "Organization" and h_type == "Person":
                gm.create_works_for(h_key, w_key)
            if h_type == "Location":
                gm.create_located_in(w_type, w_key, h_key)
            elif w_type == "Location":
                gm.create_located_in(h_type, h_key, w_key)
            if w_type != h_type:
                gm.create_related_to(w_type, w_key, h_type, h_key, weight=0.5)

    # Co-occurrence entities in same chunk → RELATED_TO
    if entity_map:
        seen_pairs = set()
        for c in chunks:
            chunk_entities = []
            for e_text, (e_type, e_key) in entity_map.items():
                if e_text in c["text"].lower():
                    chunk_entities.append((e_type, e_key))
            for i in range(len(chunk_entities)):
                for j in range(i + 1, len(chunk_entities)):
                    t1, k1 = chunk_entities[i]
                    t2, k2 = chunk_entities[j]
                    pair = (k1, k2) if k1 < k2 else (k2, k1)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        gm.create_related_to(t1, k1, t2, k2, weight=1.0)

    # Sentences
    sentences = []
    for c in chunks:
        sn = SentenceNode(
            text=c["text"],
            timestamp=c.get("start_time", 0),
            session_id=session_id,
            chunk_index=c["chunk_index"],
        )
        gm.insert_sentence(sn)
        sentences.append(sn)
        for t in spacy_result["tokens"]:
            if t["lemma"] in word_keys:
                gm.create_sentence_word_link(sn._key, t["lemma"])

    # Topics
    if topics:
        for i, t in enumerate(topics):
            label = t["sentence"][:50]
            tn = TopicNode(label=label, weight=1.0 / (i + 1))
            gm.upsert_topic(tn)
            topic_text = t["sentence"].lower().strip()
            for sn in sentences:
                if topic_text in sn.text.lower():
                    gm.create_sentence_topic_link(sn._key, tn._key)

    gm.close()
