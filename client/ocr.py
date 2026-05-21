"""Screenshot → OCR → Presse-papier (optionnel: ingestion via serveur)"""

import subprocess
import tempfile
import time
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter

from rich.console import Console

console = Console()


def capture_screenshot(region: bool = False) -> Path:
    out = Path(tempfile.mkdtemp()) / "ocr_screenshot.png"
    cmd = ["spectacle", "-b", "-n", "-o", str(out)]
    if region:
        cmd.insert(2, "-r")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"Screenshot failed: {result.stderr}")
    return out


def ocr_image(image_path: Path, lang: str = "fra") -> str:
    img = Image.open(image_path)
    img = img.filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def copy_to_clipboard(text: str):
    subprocess.run(
        ["qdbus6", "org.kde.Klipper", "/klipper", "setClipboardContents", text],
        capture_output=True,
    )
