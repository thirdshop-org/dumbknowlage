import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from config import config


def get_device_sample_rate() -> int:
    device_info = sd.query_devices(kind="input")
    return int(device_info["default_samplerate"])


def resample_to(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import scipy.signal
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    num_samples = int(duration * target_sr)
    return scipy.signal.resample(audio, num_samples)


class AudioRecorder:
    def __init__(self):
        self.device_sample_rate = get_device_sample_rate()
        self.target_sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _record_worker(self):
        def callback(indata, frames, time_info, status):
            self._queue.put(indata.copy().flatten())

        with sd.InputStream(
            samplerate=self.device_sample_rate,
            channels=self.channels,
            callback=callback,
            blocksize=config.audio.chunk_size,
        ):
            self._stop_event.wait()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def read_chunk(self) -> np.ndarray | None:
        frames = []
        while not self._queue.empty():
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not frames:
            return None
        audio = np.concatenate(frames)
        return resample_to(audio, self.device_sample_rate, self.target_sample_rate)

    def read_all(self) -> np.ndarray:
        frames = []
        while not self._queue.empty():
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not frames:
            return np.array([], dtype=np.float32)
        return resample_to(np.concatenate(frames), self.device_sample_rate, self.target_sample_rate)

    def record_duration(self, duration: float, callback: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
        frames = []

        def record_callback(indata, frames_count, time_info, status):
            chunk = indata.copy().flatten()
            frames.append(chunk)
            if callback:
                callback(chunk)

        with sd.InputStream(
            samplerate=self.device_sample_rate,
            channels=self.channels,
            callback=record_callback,
            blocksize=config.audio.chunk_size,
        ):
            sd.sleep(int(duration * 1000))

        audio = np.concatenate(frames)
        return resample_to(audio, self.device_sample_rate, self.target_sample_rate)
