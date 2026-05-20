#!/usr/bin/env python3
"""Screenshot → OCR → Presse-papier (optionnel: ingestion NLP + graphe)"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter

from rich.console import Console
from rich.panel import Panel

console = Console()


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


def ingest_text(text: str, lang: str, build_graph: bool):
    from document.reader import chunk_text
    from main import _run_nlp_pipeline
    from storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    store.connect()

    session_id = store.create_session(
        source=f"ocr_{time.strftime('%Y%m%d_%H%M%S')}",
        language=lang,
        model="ocr/tesseract",
    )

    chunks = chunk_text(text)
    store.insert_chunks_batch(session_id, chunks)

    console.print(f"\n[bold cyan]Ingestion OCR (session: {session_id})...[/]")
    _run_nlp_pipeline(
        store, session_id, chunks, lang.split("+")[0],
        build_graph=build_graph,
    )

    console.print(f"\n[bold green]✓ Texte OCR ingéré ! Session ID: {session_id}[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Screenshot → OCR → Presse-papier (ou ingestion NLP)",
    )
    parser.add_argument("--region", "-r", action="store_true", help="Sélectionner une zone")
    parser.add_argument("--image", "-i", type=Path, help="OCR depuis un fichier image")
    parser.add_argument("--lang", "-l", default="fra+eng", help="Langue OCR (défaut: fra+eng)")
    parser.add_argument("--no-clipboard", "-n", action="store_true", help="Ne pas copier dans le presse-papier")
    parser.add_argument("--preprocess", "-p", action="store_true", help="Améliorer l'image avant OCR")
    parser.add_argument("--ingest", action="store_true", help="Ingérer le texte dans la base mémoire (NLP + graphe)")
    parser.add_argument("--build-graph", action="store_true", help="Construire le graphe ArangoDB")
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

    if args.ingest:
        ingest_text(text, args.lang, args.build_graph)

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
