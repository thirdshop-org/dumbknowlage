import numpy as np
from config import config


def chunk_audio(audio: np.ndarray, sample_rate: int | None = None) -> list[tuple[np.ndarray, float, float]]:
    sr = sample_rate or config.audio.sample_rate
    chunk_len = int(config.chunk.chunk_duration * sr)
    overlap_len = int(config.chunk.overlap * sr)
    stride = chunk_len - overlap_len

    chunks: list[tuple[np.ndarray, float, float]] = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_len, len(audio))
        chunk = audio[start:end]
        if len(chunk) < sr * 0.5:
            break
        start_time = start / sr
        end_time = end / sr
        chunks.append((chunk, start_time, end_time))
        start += stride

    return chunks


def merge_transcriptions(segments: list[dict]) -> str:
    return " ".join(seg["text"].strip() for seg in segments)
