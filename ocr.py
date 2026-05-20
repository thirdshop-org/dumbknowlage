#!/usr/bin/env python3
"""Screenshot → OCR → Presse-papier"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter


def capture_screen(region: bool = False) -> Path:
    out = Path(tempfile.mkdtemp()) / "ocr_screenshot.png"
    cmd = ["spectacle", "-b", "-n", "-o", str(out)]
    if region:
        cmd.insert(2, "-r")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"Échec screenshot: {result.stderr}")
    return out


def enhance_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    return img


def ocr_image(image: Image.Image, lang: str) -> str:
    custom_config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text.strip()


def copy_to_clipboard(text: str):
    subprocess.run(
        ["qdbus6", "org.kde.klipper", "/klipper", "setClipboardContents", text],
        capture_output=True, timeout=5,
    )


def main():
    parser = argparse.ArgumentParser(description="Screenshot → OCR → Presse-papier")
    parser.add_argument("--region", "-r", action="store_true", help="Sélectionner une zone")
    parser.add_argument("--image", "-i", type=Path, help="OCR depuis un fichier image")
    parser.add_argument("--lang", "-l", default="fra+eng", help="Langue OCR (défaut: fra+eng)")
    parser.add_argument("--no-clipboard", "-n", action="store_true", help="Ne pas copier dans le presse-papier")
    parser.add_argument("--preprocess", "-p", action="store_true", help="Améliorer l'image avant OCR")
    args = parser.parse_args()

    if args.image:
        if not args.image.exists():
            print(f"Fichier introuvable: {args.image}", file=sys.stderr)
            sys.exit(1)
        path = args.image
    else:
        print("Capture d'écran...", end=" ", flush=True)
        path = capture_screen(region=args.region)
        print("✓")

    print("OCR...", end=" ", flush=True)
    img = enhance_image(path) if args.preprocess else Image.open(path)
    text = ocr_image(img, args.lang)
    print("✓")

    if not text:
        print("Aucun texte détecté.")
        sys.exit(0)

    print(f"\n{'─' * 50}")
    print(text)
    print(f"{'─' * 50}")
    print(f"{len(text)} caractères")

    if not args.no_clipboard:
        copy_to_clipboard(text)
        print("✓ Copié dans le presse-papier")

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
