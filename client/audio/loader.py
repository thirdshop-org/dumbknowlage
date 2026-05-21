import numpy as np
from config import config


def load_audio(file_path: str, sample_rate: int | None = None) -> np.ndarray:
    import soundfile as sf

    sr = sample_rate or config.audio.sample_rate
    audio, orig_sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * sr / orig_sr))
    return audio.astype(np.float32)
