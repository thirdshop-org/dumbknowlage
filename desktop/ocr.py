"""Cross-platform OCR: screenshot → text → clipboard."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter


def _copy_to_clipboard(text: str):
    if sys.platform == "win32":
        import pyperclip
        pyperclip.copy(text)
    else:
        subprocess.run(
            ["qdbus6", "org.kde.Klipper", "/klipper", "setClipboardContents", text],
            capture_output=True,
        )


def _enhance_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _ocr_image(img: Image.Image, lang: str = "fra") -> str:
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def ocr_screenshot_linux(region: bool = False, lang: str = "fra") -> str:
    """Take screenshot via spectacle (KDE), OCR it, copy to clipboard."""
    tmp = Path(tempfile.mkdtemp()) / "ocr_screenshot.png"
    cmd = ["spectacle", "-b", "-n", "-o", str(tmp)]
    if region:
        cmd.insert(2, "-r")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"Screenshot failed: {result.stderr}")
    img = Image.open(tmp)
    text = _ocr_image(img, lang)
    _copy_to_clipboard(text)
    tmp.unlink(missing_ok=True)
    return text


def ocr_screenshot_windows(lang: str = "fra") -> str:
    """Take screenshot via mss, OCR it, copy to clipboard."""
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    text = _ocr_image(img, lang)
    _copy_to_clipboard(text)
    return text
