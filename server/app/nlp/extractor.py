from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config


def extract_frequencies(tokens: list[dict], field: str = "lemma") -> list[tuple[str, int]]:
    words = [t[field] for t in tokens if not t.get("is_punct", False) and not t.get("is_stop", False) and t.get(field, "").strip()]
    counter = Counter(words)
    return counter.most_common(config.extraction.top_n_frequencies)


def extract_co_occurrences(
    tokens: list[dict],
    window: int | None = None,
    field: str = "lemma",
) -> list[tuple[tuple[str, str], int]]:
    w = window or config.extraction.co_occurrence_window
    words = [t[field] for t in tokens
             if not t.get("is_punct", False)
             and not t.get("is_stop", False)
             and t.get(field, "").strip()]
    co_occur: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(len(words)):
        for j in range(i + 1, min(i + w + 1, len(words))):
            if words[i] != words[j]:
                pair = tuple(sorted((words[i], words[j])))
                co_occur[pair] += 1
    return sorted(co_occur.items(), key=lambda x: -x[1])[:50]


def compute_tfidf(documents: list[str]) -> list[dict]:
    if len(documents) < 2:
        return []
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=["le", "la", "les", "de", "des", "du", "un", "une", "et", "est",
                     "sont", "dans", "pour", "sur", "avec", "que", "qui", "pas", "l", "d"],
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).flatten()
    top_indices = scores.argsort()[::-1][:20]
    return [{"term": feature_names[i], "score": float(scores[i])} for i in top_indices]


def detect_burst_topics(
    chunk_texts: list[str],
    window_size: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    ws = window_size or config.extraction.burst_window_size
    th = threshold or config.extraction.burst_threshold

    if len(chunk_texts) < ws:
        return []

    # Tokenise chaque chunk
    chunk_words = []
    for txt in chunk_texts:
        words = [w.lower().strip(".,!?;:()[]\"'") for w in txt.split()]
        chunk_words.append([w for w in words if len(w) > 2])

    # Fréquence globale
    all_words = [w for cw in chunk_words for w in cw]
    global_counts = Counter(all_words)
    global_total = len(all_words)

    if global_total == 0:
        return []

    bursts = []
    for i in range(len(chunk_words) - ws + 1):
        window = chunk_words[i : i + ws]
        window_words = [w for cw in window for w in cw]
        window_counts = Counter(window_words)
        window_total = len(window_words)

        if window_total == 0:
            continue

        for word, count in window_counts.items():
            global_freq = global_counts[word] / global_total
            local_freq = count / window_total
            if global_freq > 0:
                burst_score = (local_freq - global_freq) / (global_freq + 1e-8)
            else:
                burst_score = local_freq

            if burst_score > th:
                bursts.append({
                    "word": word,
                    "burst_score": float(burst_score),
                    "window_start": i,
                    "local_frequency": count,
                })

    bursts.sort(key=lambda x: -x["burst_score"])
    seen_words = set()
    unique_bursts = []
    for b in bursts:
        if b["word"] not in seen_words:
            seen_words.add(b["word"])
            unique_bursts.append(b)
            if len(unique_bursts) >= config.extraction.top_n_hot_topics:
                break

    return unique_bursts
