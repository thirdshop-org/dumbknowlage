from __future__ import annotations

import numpy as np
import torch
import whisper
from config import config


def _fix_whisper_dtype():
    """Patch whisper audio to ensure float32 consistency on CPU."""
    import whisper.audio as wa
    _orig_log_mel = wa.log_mel_spectrogram

    def _patched_log_mel(audio, n_mels=None, padding=None, device=None):
        if n_mels is None:
            n_mels = wa.N_MELS
        if padding is None:
            padding = wa.N_SAMPLES
        result = _orig_log_mel(audio, n_mels, padding, device)
        return result.to(dtype=torch.float32)

    wa.log_mel_spectrogram = _patched_log_mel


class WhisperModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.whisper.model
        self._model = None

    def load(self):
        if self._model is None:
            torch.set_default_dtype(torch.float32)
            _fix_whisper_dtype()
            self._model = whisper.load_model(self.model_name)
            self._model = self._model.float()

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
