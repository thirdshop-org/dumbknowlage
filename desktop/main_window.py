import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from client.api_client import ApiClient


class SearchWorker(QThread):
    finished = Signal(list)

    def __init__(self, client, query, top_k=5):
        super().__init__()
        self.client = client
        self.query = query
        self.top_k = top_k

    def run(self):
        try:
            results = self.client.search(self.query, self.top_k)
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit([{"error": str(e)}])


class SessionsWorker(QThread):
    finished = Signal(list)

    def __init__(self, client):
        super().__init__()
        self.client = client

    def run(self):
        try:
            sessions = self.client.list_sessions()
            self.finished.emit(sessions)
        except Exception as e:
            self.finished.emit([{"error": str(e)}])


class EntitiesWorker(QThread):
    finished = Signal(list)

    def __init__(self, client, query=""):
        super().__init__()
        self.client = client
        self.query = query

    def run(self):
        try:
            entities = self.client.list_entities(q=self.query)
            self.finished.emit(entities)
        except Exception as e:
            self.finished.emit([{"error": str(e)}])


class IngestWorker(QThread):
    finished = Signal(str)
    progress = Signal(str)

    def __init__(self, client, file_paths, language="fr"):
        super().__init__()
        self.client = client
        self.file_paths = file_paths
        self.language = language

    def run(self):
        for path in self.file_paths:
            p = Path(path)
            ext = p.suffix.lower()
            self.progress.emit(f"Ingesting {p.name}...")
            try:
                if ext in (".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"):
                    text = p.read_text(encoding="utf-8", errors="replace")
                    result = self.client.ingest(text, filename=p.name, language=self.language)
                elif ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
                    result = self.client.transcribe(str(p), language=self.language)
                elif ext == ".pdf":
                    import fitz
                    doc = fitz.open(str(p))
                    text = "\n\n".join(page.get_text() for page in doc)
                    doc.close()
                    result = self.client.ingest(text, filename=p.name, language=self.language)
                elif ext == ".docx":
                    from docx import Document as DocxDocument
                    doc = DocxDocument(str(p))
                    text = "\n".join(para.text for para in doc.paragraphs)
                    result = self.client.ingest(text, filename=p.name, language=self.language)
                else:
                    self.progress.emit(f"Unsupported format: {ext}")
                    continue
                self.progress.emit(f"OK: {p.name} ({result.get('chunks_count', '?')} chunks)")
            except Exception as e:
                self.progress.emit(f"Error: {p.name}: {e}")
        self.finished.emit("Done")


class SearchTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcripts and documents...")
        self.search_input.returnPressed.connect(self.do_search)
        row.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.do_search)
        row.addWidget(self.search_btn)

        top_k_label = QLabel("Top:")
        row.addWidget(top_k_label)
        self.top_k_input = QLineEdit("5")
        self.top_k_input.setFixedWidth(40)
        row.addWidget(self.top_k_input)
        row.addStretch()
        layout.addLayout(row)

        self.results_list = QListWidget()
        self.results_list.setWordWrap(True)
        self.results_list.itemDoubleClicked.connect(self._show_result_detail)
        layout.addWidget(self.results_list)

        self.setLayout(layout)

    def do_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        self.results_list.clear()
        self.results_list.addItem("Searching...")
        self.worker = SearchWorker(self.client, query, int(self.top_k_input.text() or "5"))
        self.worker.finished.connect(self._on_results)
        self.worker.start()

    def _on_results(self, results):
        self.results_list.clear()
        if not results:
            self.results_list.addItem("No results found.")
            return
        if isinstance(results[0], dict) and "error" in results[0]:
            self.results_list.addItem(f"Error: {results[0]['error']}")
            return
        for r in results:
            score = r.get("score", 0)
            source = r.get("source", "?")
            stype = r.get("source_type", "?")
            text = r.get("text", "")[:200]
            item = QListWidgetItem(f"[{score:.3f}] ({source}) [{stype}]\n{text}")
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.results_list.addItem(item)

    def _show_result_detail(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            QMessageBox.information(self, "Result Detail", data.get("text", ""))


class SessionsTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(refresh_btn)
        row.addStretch()
        layout.addLayout(row)

        self.sessions_list = QListWidget()
        self.sessions_list.setWordWrap(True)
        self.sessions_list.itemDoubleClicked.connect(self._show_session)
        layout.addWidget(self.sessions_list)

        self.setLayout(layout)
        self.refresh()

    def refresh(self):
        self.sessions_list.clear()
        self.sessions_list.addItem("Loading...")
        self.worker = SessionsWorker(self.client)
        self.worker.finished.connect(self._on_sessions)
        self.worker.start()

    def _on_sessions(self, sessions):
        self.sessions_list.clear()
        if not sessions:
            self.sessions_list.addItem("No sessions.")
            return
        if isinstance(sessions[0], dict) and "error" in sessions[0]:
            self.sessions_list.addItem(f"Error: {sessions[0]['error']}")
            return
        for s in sessions:
            sid = s["id"]
            source = s.get("source", "?")
            dur = s.get("duration", "-")
            dur_str = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-"
            created = s.get("created_at", "")[:19]
            item = QListWidgetItem(f"{sid}: {source} ({dur_str}) {created}")
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.sessions_list.addItem(item)

    def _show_session(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        try:
            detail = self.client.get_session(data["id"])
            chunks = detail.get("chunks", [])
            texts = "\n\n---\n\n".join(
                c["text"][:300] for c in chunks
            )
            QMessageBox.information(self, f"Session {data['id']}", texts or "(empty)")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class EntitiesTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter entities...")
        self.search_input.returnPressed.connect(self.refresh)
        row.addWidget(self.search_input)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        row.addWidget(refresh_btn)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self._confirm_selected)
        row.addWidget(confirm_btn)

        deny_btn = QPushButton("Deny")
        deny_btn.clicked.connect(self._deny_selected)
        row.addWidget(deny_btn)

        row.addStretch()
        layout.addLayout(row)

        self.entities_list = QListWidget()
        self.entities_list.setWordWrap(True)
        self.entities_list.itemDoubleClicked.connect(self._show_detail)
        layout.addWidget(self.entities_list)

        self.setLayout(layout)
        self.refresh()

    def refresh(self):
        self.entities_list.clear()
        self.entities_list.addItem("Loading...")
        self.worker = EntitiesWorker(self.client, self.search_input.text().strip())
        self.worker.finished.connect(self._on_entities)
        self.worker.start()

    def _on_entities(self, entities):
        self.entities_list.clear()
        if not entities:
            self.entities_list.addItem("No entities.")
            return
        if isinstance(entities[0], dict) and "error" in entities[0]:
            self.entities_list.addItem(f"Error: {entities[0]['error']}")
            return
        for e in entities:
            etype = e.get("_type", e.get("type", "?"))
            name = e.get("name", "?")
            conf = e.get("confidence", "?")
            feedback = e.get("user_feedback", 0)
            fb = "✓" if feedback == 1 else "✗" if feedback == -1 else " "
            item = QListWidgetItem(f"[{fb}] {etype}: {name} (conf: {conf})")
            item.setData(Qt.ItemDataRole.UserRole, e)
            self.entities_list.addItem(item)

    def _selected_entity(self):
        item = self.entities_list.currentItem()
        if not item:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return None
        return data.get("_type", data.get("type", "")), data.get("_key", data.get("key", ""))

    def _confirm_selected(self):
        ent = self._selected_entity()
        if not ent:
            return
        try:
            self.client.confirm_entity(*ent)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _deny_selected(self):
        ent = self._selected_entity()
        if not ent:
            return
        try:
            self.client.deny_entity(*ent)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _show_detail(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        try:
            etype = data.get("_type", data.get("type", ""))
            ekey = data.get("_key", data.get("key", ""))
            detail = self.client.get_entity(etype, ekey)
            text = json.dumps(detail, indent=2, ensure_ascii=False)[:3000]
            QMessageBox.information(self, f"Entity: {data.get('name', '?')}", text)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class DragDropArea(QWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("Drop files here\n\nSupported: .txt .md .pdf .docx .mp3 .wav .m4a")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 8px; padding: 20px;"
        )
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.files_dropped.emit(paths)


class IngestTab(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        label = QLabel("Drop documents or audio files to ingest:")
        layout.addWidget(label)

        self.drop_area = DragDropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_area)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _on_files_dropped(self, paths):
        self.log.append(f"Dropped {len(paths)} file(s)")
        self.worker = IngestWorker(self.client, paths)
        self.worker.progress.connect(lambda msg: self.log.append(msg))
        self.worker.finished.connect(lambda: self.log.append("--- Done ---"))
        self.worker.start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = ApiClient()
        self.setWindowTitle("notes-graph")
        self.resize(900, 700)

        self._setup_ui()

    def _setup_ui(self):
        tabs = QTabWidget()
        tabs.addTab(SearchTab(self.client), "Search")
        tabs.addTab(SessionsTab(self.client), "Sessions")
        tabs.addTab(EntitiesTab(self.client), "Entities")
        tabs.addTab(IngestTab(self.client), "Ingest")

        self.setCentralWidget(tabs)
