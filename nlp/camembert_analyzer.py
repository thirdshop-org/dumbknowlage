from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from config import config

logging.getLogger("transformers").setLevel(logging.ERROR)


class CamembertAnalyzer:
    def __init__(self):
        self.model_name = config.nlp.camembert_model
        self.device = torch.device(config.nlp.device)
        self._tokenizer = None
        self._model = None
        self._load_error = None

    def load(self):
        if self._load_error:
            return
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    ignore_mismatched_sizes=True,
                ).to(self.device)
                self._model.eval()
            except Exception as e:
                self._load_error = str(e)

    def _is_available(self) -> bool:
        self.load()
        return self._model is not None

    def get_embedding(self, text: str) -> np.ndarray:
        if not self._is_available():
            return np.array([], dtype=np.float32)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

    def extract_topics(self, text: str, top_k: int = 3) -> list[dict]:
        if not self._is_available():
            return []
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if len(s.strip()) > 10]

        topics = []
        for sent in sentences:
            try:
                inputs = self._tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                topics.append({
                    "sentence": sent,
                    "embedding": cls_vec.tolist(),
                })
            except Exception:
                continue

        return topics[:top_k]
