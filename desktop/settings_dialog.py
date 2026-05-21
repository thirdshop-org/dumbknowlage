from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from client.api_client import ApiClient
from client.config import config


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self.api = ApiClient()

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout()

        form = QFormLayout()

        self.server_url = QLineEdit()
        self.server_url.setPlaceholderText("https://dumbknowlage.thirdshop.fr")
        form.addRow("Server URL:", self.server_url)

        self.default_lang = QComboBox()
        self.default_lang.addItems(["fr", "en", "de", "es", "it"])
        form.addRow("Default language:", self.default_lang)

        self.default_duration = QLineEdit()
        self.default_duration.setPlaceholderText("30")
        form.addRow("Recording duration (s):", self.default_duration)

        self.ocr_lang = QLineEdit()
        self.ocr_lang.setPlaceholderText("fra")
        form.addRow("OCR language (tesseract):", self.ocr_lang)

        layout.addLayout(form)

        status_label = QLabel()
        layout.addWidget(status_label)

        btn_row = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(test_btn)

        btn_row.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        self.setLayout(layout)

    def _load_settings(self):
        self.server_url.setText(config.server_url)
        idx = self.default_lang.findText(config.default_language)
        if idx >= 0:
            self.default_lang.setCurrentIndex(idx)
        self.default_duration.setText(str(config.default_duration))

    def _test_connection(self):
        import os
        os.environ["NOTES_GRAPH_SERVER"] = self.server_url.text().strip()
        from client.api_client import ApiClient
        try:
            api = ApiClient()
            h = api.health()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Connection OK",
                f"Server: {h.get('status')}\n"
                f"ArangoDB: {h.get('arango')}\n"
                f"ChromaDB: {h.get('chroma')}\n"
                f"Ollama: {h.get('ollama')}\n"
                f"Whisper: {h.get('whisper_model')}",
            )
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Connection Failed", str(e))

    def _save_settings(self):
        import os
        os.environ["NOTES_GRAPH_SERVER"] = self.server_url.text().strip()
        config.server_url = self.server_url.text().strip()
        config.default_language = self.default_lang.currentText()
        try:
            config.default_duration = float(self.default_duration.text())
        except ValueError:
            pass
        self.accept()
