import queue
import sys
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


class UploadWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, wav_path: str, language: str = "fr"):
        super().__init__()
        self.wav_path = wav_path
        self.language = language

    def run(self):
        from client.api_client import ApiClient
        try:
            client = ApiClient()
            result = client.transcribe(self.wav_path, language=self.language)
            Path(self.wav_path).unlink(missing_ok=True)
            self.done.emit(result.get("session_id", ""))
        except Exception as e:
            self.error.emit(str(e))


class RecorderPanel(QWidget):
    recording_done = Signal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self._audio_frames = []
        self._lock = threading.Lock()
        self._recording_thread: threading.Thread | None = None
        self._start_time = 0.0

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(300, 110)

        self.container = QFrame()
        self.container.setStyleSheet(
            "QFrame { background: rgba(40, 40, 50, 220); border-radius: 12px; }"
        )

        layout = QVBoxLayout(self.container)

        row = QHBoxLayout()
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(12, 12)
        self.status_dot.setStyleSheet("background: red; border-radius: 6px;")
        row.addWidget(self.status_dot)

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet(
            "color: white; font-size: 18px; font-weight: bold;"
        )
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.timer_label, stretch=1)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 6px; padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        self.stop_btn.clicked.connect(self.stop_recording)
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        self.status_label = QLabel("Recording...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)
        self.setLayout(outer)

    def _setup_timer(self):
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._timer.setInterval(200)

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
        self._start_time = time.time()
        self.is_recording = True

        self.recorder = AudioRecorder()
        self.status_dot.setStyleSheet("background: red; border-radius: 6px;")
        self.timer_label.setText("00:00")
        self.status_label.setText("Recording...")

        self._recording_thread = threading.Thread(
            target=self._record_worker, daemon=True
        )
        self._recording_thread.start()

        self._timer.start()
        self._center_on_screen()
        self.show()
        self.raise_()

    def _record_worker(self):
        self.recorder.start()
        while self.is_recording:
            chunk = self.recorder.read_chunk()
            if chunk is not None:
                with self._lock:
                    self._audio_frames.append(chunk)
            time.sleep(0.05)
        remaining = self.recorder.read_all()
        with self._lock:
            if len(remaining) > 0:
                self._audio_frames.append(remaining)

    def stop_recording(self):
        self.is_recording = False
        self._timer.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Processing...")

        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=5)

        with self._lock:
            if not self._audio_frames:
                self.status_label.setText("No audio captured")
                QTimer.singleShot(2000, self.hide)
                return
            audio = np.concatenate(self._audio_frames)
            self._audio_frames = []

        self.status_label.setText("Saving...")
        self.recorder.stop()

        tmp = Path(tempfile.mkdtemp()) / f"recording_{int(time.time())}.wav"
        import soundfile as sf
        sf.write(str(tmp), audio, client_config.audio.sample_rate)

        self.status_label.setText("Uploading...")
        self.worker = UploadWorker(str(tmp), language="fr")
        self.worker.done.connect(self._on_upload_done)
        self.worker.error.connect(self._on_upload_error)
        self.worker.start()

    def _on_upload_done(self, session_id: str):
        self.recording_done.emit(session_id)
        self.status_label.setText(f"OK: {session_id}")
        QTimer.singleShot(2000, self.hide)

    def _on_upload_error(self, err: str):
        self.status_label.setText(f"Error: {err}")
        self.stop_btn.setEnabled(True)
        QTimer.singleShot(5000, self.hide)

    def _update_timer(self):
        elapsed = time.time() - self._start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.timer_label.setText(f"{mins:02d}:{secs:02d}")
        if int(elapsed) % 2 == 0:
            self.status_dot.setStyleSheet("background: red; border-radius: 6px;")
        else:
            self.status_dot.setStyleSheet(
                "background: rgba(255, 0, 0, 40); border-radius: 6px;"
            )
