import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from client.config import config as client_config


class ChunkUploader(QThread):
    chunk_sent = Signal(str)
    chunk_error = Signal(str)

    def __init__(self, wav_path):
        super().__init__()
        self.wav_path = wav_path

    def run(self):
        from client.api_client import ApiClient
        try:
            client = ApiClient()
            client.transcribe(self.wav_path)
            Path(self.wav_path).unlink(missing_ok=True)
            self.chunk_sent.emit("ok")
        except Exception as e:
            self.chunk_error.emit(str(e))


class RecorderPanel(QWidget):
    recording_done = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self._chunk_count = 0
        self._start_time = 0.0
        self._audio_frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recorder = None
        self._record_thread: threading.Thread | None = None
        self._chunk_timer = QTimer()
        self._chunk_timer.timeout.connect(self._send_chunk)
        self._chunk_interval_ms = 60000
        self._uploaders: list[ChunkUploader] = []
        self._stopping = False

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(320, 120)

        self.container = QFrame()
        self.container.setStyleSheet(
            "QFrame { background: rgba(40, 40, 50, 220); border-radius: 12px; }"
        )

        layout = QVBoxLayout(self.container)

        row = QHBoxLayout()
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(12, 12)
        row.addWidget(self.status_dot)

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.timer_label, stretch=1)

        self.chunks_label = QLabel("0 sent")
        self.chunks_label.setStyleSheet("color: #888; font-size: 11px;")
        row.addWidget(self.chunks_label)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 6px; padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        self.stop_btn.clicked.connect(self.stop_recording)
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        self.status_label = QLabel("Streaming...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)
        self.setLayout(outer)

    def _center_on_screen(self):
        screen = self.screen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.center().x() - self.width() // 2
            y = geo.top() + 40
            self.move(x, y)

    def start_recording(self):
        from client.audio.recorder import AudioRecorder

        self._audio_frames = []
        self._chunk_count = 0
        self._start_time = time.time()
        self.is_recording = True

        self._recorder = AudioRecorder()
        self._recorder.start()

        self._record_thread = threading.Thread(target=self._record_worker, daemon=True)
        self._record_thread.start()

        self.status_dot.setStyleSheet("background: red; border-radius: 6px;")
        self.timer_label.setText("00:00")
        self.chunks_label.setText("0 sent")
        self.status_label.setText("Streaming...")

        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._update_ui)
        self._ui_timer.start(200)

        self._chunk_timer.start(self._chunk_interval_ms)

    def _record_worker(self):
        while self.is_recording:
            chunk = self._recorder.read_chunk()
            if chunk is not None:
                with self._lock:
                    self._audio_frames.append(chunk)
            time.sleep(0.05)

    def _send_chunk(self):
        with self._lock:
            if not self._audio_frames:
                return
            frames = self._audio_frames[:]
            self._audio_frames.clear()

        if not frames:
            return

        audio = np.concatenate(frames)
        self._chunk_count += 1

        tmp = Path(tempfile.mkdtemp()) / f"chunk_{int(time.time())}.wav"
        import soundfile as sf
        sf.write(str(tmp), audio, client_config.audio.sample_rate)

        self.status_label.setText(f"Sending chunk {self._chunk_count}...")
        self.chunks_label.setText(f"{self._chunk_count} sent")

        uploader = ChunkUploader(str(tmp))
        uploader.chunk_sent.connect(lambda _, u=uploader: self._on_chunk_done(u))
        uploader.chunk_error.connect(lambda _, u=uploader: self._on_chunk_done(u))
        self._uploaders.append(uploader)
        uploader.start()

    def _on_chunk_done(self, uploader):
        self._uploaders = [u for u in self._uploaders if u is not uploader]
        if not self._uploaders:
            self.status_label.setText("Streaming...")

    def stop_recording(self):
        self._stopping = True
        self.is_recording = False
        self._ui_timer.stop()
        self._chunk_timer.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Finalizing...")

        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=5)

        self._recorder.stop()

        with self._lock:
            if self._audio_frames:
                audio = np.concatenate(self._audio_frames)
                self._audio_frames.clear()
                self._chunk_count += 1
                tmp = Path(tempfile.mkdtemp()) / f"chunk_final_{int(time.time())}.wav"
                import soundfile as sf
                sf.write(str(tmp), audio, client_config.audio.sample_rate)
                uploader = ChunkUploader(str(tmp))
                self._uploaders.append(uploader)
                uploader.finished.connect(self._on_stop_done)
                uploader.start()
            else:
                self._on_stop_done()

    def _on_stop_done(self):
        if not self._stopping:
            return
        self._uploaders.clear()
        self.recording_done.emit(str(self._chunk_count))
        self.status_label.setText(f"Done: {self._chunk_count} chunks")
        QTimer.singleShot(2000, self.hide)

    def _update_ui(self):
        elapsed = time.time() - self._start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.timer_label.setText(f"{mins:02d}:{secs:02d}")
        if int(elapsed) % 2 == 0:
            self.status_dot.setStyleSheet("background: red; border-radius: 6px;")
        else:
            self.status_dot.setStyleSheet("background: rgba(255, 0, 0, 40); border-radius: 6px;")
