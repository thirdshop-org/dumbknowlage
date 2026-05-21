from __future__ import annotations

from typing import Callable

import numpy as np

from audio.chunker import chunk_audio
from config import config
from transcription.whisper_model import WhisperModel


class Transcriber:
    def __init__(self, model_name: str | None = None):
        self.model = WhisperModel(model_name)
        self._context: list[str] = []

    def transcribe_chunks(
        self,
        audio: np.ndarray,
        language: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict]:
        chunks = chunk_audio(audio, config.chunk.sample_rate)
        results: list[dict] = []

        for idx, (chunk, start_time, end_time) in enumerate(chunks):
            result = self.model.transcribe(chunk, language=language)
            text = result.get("text", "").strip()

            self._context.append(text)

            entry = {
                "chunk_index": idx,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "segments": result.get("segments", []),
                "language": result.get("language", language),
            }
            results.append(entry)

            if progress_callback:
                progress_callback(idx + 1, len(chunks), text)

        return results

    def reset_context(self):
        self._context.clear()
