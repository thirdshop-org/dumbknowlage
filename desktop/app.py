import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

from desktop.main_window import MainWindow
from desktop.recorder_panel import RecorderPanel
from desktop.settings_dialog import SettingsDialog


class TrayApp:
    def __init__(self):
        self.main_window = MainWindow()
        self.recorder_panel = RecorderPanel()
        self.recorder_panel.recording_done.connect(self._on_recording_done)

        self._setup_tray()

    def _setup_tray(self):
        icon = self._make_icon()
        self.tray = QSystemTrayIcon(icon)
        self.tray.setToolTip("notes-graph")

        menu = QMenu()

        show_act = QAction("Show Window", menu)
        show_act.triggered.connect(self.show_window)
        menu.addAction(show_act)

        self.record_act = QAction("Start Recording", menu)
        self.record_act.triggered.connect(self.toggle_recording)
        menu.addAction(self.record_act)

        ocr_act = QAction("OCR Screenshot", menu)
        ocr_act.triggered.connect(self.run_ocr)
        menu.addAction(ocr_act)

        menu.addSeparator()

        settings_act = QAction("Settings", menu)
        settings_act.triggered.connect(self.show_settings)
        menu.addAction(settings_act)

        quit_act = QAction("Quit", menu)
        quit_act.triggered.connect(self.quit_app)
        menu.addAction(quit_act)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _make_icon(self):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.darkBlue)
        return QIcon(pixmap)

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window()

    def show_window(self):
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def toggle_recording(self):
        if self.recorder_panel.is_recording:
            self.recorder_panel.stop_recording()
            self.record_act.setText("Start Recording")
        else:
            self.recorder_panel.start_recording()
            self.record_act.setText("Stop Recording")

    def run_ocr(self):
        try:
            if sys.platform == "win32":
                from desktop.ocr import ocr_screenshot_windows as _ocr
            else:
                from desktop.ocr import ocr_screenshot_linux as _ocr
            text = _ocr()
            self.main_window.statusBar().showMessage(
                f"OCR: {len(text)} chars copied to clipboard"
            )
        except Exception as e:
            QMessageBox.warning(self.main_window, "OCR Error", str(e))

    def show_settings(self):
        dlg = SettingsDialog(self.main_window)
        dlg.exec()

    def _on_recording_done(self, session_id: str):
        self.record_act.setText("Start Recording")
        self.tray.showMessage(
            "Recording Complete",
            f"Session {session_id} transcribed and indexed",
            QSystemTrayIcon.MessageIcon.Information,
            4000,
        )

    def quit_app(self):
        if self.recorder_panel.is_recording:
            self.recorder_panel.stop_recording()
        self.tray.hide()
        QApplication.quit()


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("notes-graph")
    app.setOrganizationName("notes-graph")

    tray_app = TrayApp()

    if "--show" in sys.argv:
        tray_app.show_window()

    sys.exit(app.exec())
