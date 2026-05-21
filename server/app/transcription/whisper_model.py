from __future__ import annotations

import numpy as np
import whisper
from config import config


class WhisperModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.whisper.model
        self._model = None

    def load(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio: np.ndarray, language: str | None = None, **kwargs) -> dict:
        self.load()
        opts = {
            "beam_size": config.whisper.beam_size,
            "temperature": config.whisper.temperature,
            "condition_on_previous_text": config.whisper.condition_on_previous_text,
        }
        if language:
            opts["language"] = language
        opts.update(kwargs)
        result = self._model.transcribe(audio, **opts)
        return result

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
