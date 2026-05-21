from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import config
from api_client import ApiClient
from sync_manager import SyncManager

console = Console()
api = ApiClient()
sync_mgr = SyncManager()

# ─── Health ──────────────────────────────────────────────────────────────────


def cmd_health(args: argparse.Namespace):
    try:
        h = api.health()
        console.print(Panel(f"[bold green]Serveur OK[/]"))
        console.print(f"  Statut: {h.get('status')}")
        console.print(f"  ArangoDB: {'✓' if h.get('arango') else '✗'}")
        console.print(f"  ChromaDB: {'✓' if h.get('chroma') else '✗'}")
        console.print(f"  Ollama: {'✓' if h.get('ollama') else '✗'}")
        console.print(f"  Whisper: {h.get('whisper_model', '?')}")
    except Exception as e:
        console.print(f"[red]Serveur injoignable: {e}[/]")
        pending = sync_mgr.list_pending()
        if pending:
            console.print(f"[yellow]→ {len(pending)} éléments en attente de synchronisation[/]")

# ─── Record ──────────────────────────────────────────────────────────────────


def cmd_record(args: argparse.Namespace):
    from audio.loader import load_audio
    from audio.recorder import AudioRecorder
    from audio.chunker import chunk_audio

    duration = args.duration or config.default_duration
    lang = args.language or config.default_language

    recorder = AudioRecorder()
    console.print(f"[bold yellow]Enregistrement de {duration}s...[/]")
    audio = recorder.record_duration(duration)

    # Save temp file
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = f"/tmp/record_{ts}.wav"
    import soundfile as sf
    sf.write(tmp_path, audio, 16000)

    # Send to server
    try:
        result = api.transcribe(tmp_path, language=lang, build_graph=not args.no_graph)
        console.print(f"[green]✓ Session:[/] {result['session_id']} ({result['chunks_count']} chunks)")
    except Exception as e:
        console.print(f"[red]Serveur injoignable: {e}[/]")
        # Save for later sync
        sync_mgr.save_pending({"action": "transcribe", "audio_path": tmp_path, "language": lang}, "audio")
        console.print(f"[yellow]→ Sauvegardé pour synchronisation ultérieure[/]")

# ─── Ingest ──────────────────────────────────────────────────────────────────


INJECT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".docx"}


def cmd_ingest(args: argparse.Namespace):
    from document.reader import get_metadata, read_file

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]Introuvable: {path}[/]")
        sys.exit(1)

    lang = args.language or config.default_language

    if path.is_dir():
        supported = []
        for p in path.rglob("*") if args.recursive else path.glob("*"):
            if p.suffix.lower() in INJECT_EXTENSIONS and p.is_file():
                supported.append(p)
        supported.sort()

        if not supported:
            console.print(f"[yellow]Aucun document trouvé dans {path}[/]")
            return

        console.print(f"[bold]Dossier:[/] {path} ({len(supported)} fichiers)")
        for fp in supported:
            try:
                text = read_file(fp)
                meta = get_metadata(fp, text)
                result = api.ingest(text, filename=str(fp), language=lang,
                                    build_graph=not args.no_graph)
                console.print(f"  [green]✓[/] {fp.name} → {result['session_id']}")
            except Exception as e:
                console.print(f"  [red]✗[/] {fp.name}: {e}")
    else:
        if path.suffix.lower() not in INJECT_EXTENSIONS:
            console.print(f"[red]Format non supporté: {path.suffix}[/]")
            sys.exit(1)
        try:
            text = read_file(path)
            meta = get_metadata(path, text)
            result = api.ingest(text, filename=str(path), language=lang,
                                build_graph=not args.no_graph)
            console.print(f"[green]✓ Session:[/] {result['session_id']} ({result['chunks_count']} chunks)")
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/]")
            sync_mgr.save_pending({"action": "ingest", "text": text, "filename": str(path), "language": lang}, "document")

# ─── OCR ─────────────────────────────────────────────────────────────────────


def cmd_ocr(args: argparse.Namespace):
    from ocr import capture_screenshot, ocr_image, copy_to_clipboard

    lang = args.language or "fra"
    region = args.region

    console.print("[yellow]Capture d'écran...[/]")
    img_path = capture_screenshot(region=region)

    console.print("[yellow]OCR...[/]")
    text = ocr_image(img_path, lang=lang)
    console.print(f"[green]Texte extrait ({len(text)} chars)[/]")

    if not args.no_clipboard:
        copy_to_clipboard(text)
        console.print("[dim]Copié dans le presse-papier[/]")

    if args.ingest:
        if args.no_graph:
            build_graph = False
        else:
            build_graph = True
        try:
            result = api.ingest(text, filename=f"ocr_{time.strftime('%Y%m%d_%H%M%S')}",
                                language=lang[:2], build_graph=build_graph)
            console.print(f"[green]✓ Session OCR:[/] {result['session_id']}")
        except Exception as e:
            console.print(f"[red]Erreur serveur: {e}[/]")
            sync_mgr.save_pending({"action": "ingest", "text": text,
                                    "filename": f"ocr_{time.strftime('%Y%m%d_%H%M%S')}",
                                    "language": lang[:2]}, "ocr")
    else:
        console.print(f"\n[bold]Texte OCR:[/]\n{text[:500]}")

# ─── Sessions ────────────────────────────────────────────────────────────────


def cmd_sessions(args: argparse.Namespace):
    try:
        sessions = api.list_sessions()
        if not sessions:
            console.print("[yellow]Aucune session[/]")
            return
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Source", style="white")
        table.add_column("Durée", style="yellow")
        table.add_column("Langue", style="green")
        table.add_column("Date", style="dim")
        for s in sessions:
            dur = f"{s['duration']:.1f}s" if s.get("duration") else "-"
            table.add_row(s["id"], str(s.get("source", ""))[:30], dur,
                           s.get("language", "-"), s.get("created_at", "")[:19])
        console.print(table)
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")

# ─── Search ──────────────────────────────────────────────────────────────────


def cmd_search(args: argparse.Namespace):
    try:
        results = api.search(args.query, top_k=args.top_k)
        if not results:
            console.print("[yellow]Aucun résultat[/]")
            return
        for r in results:
            console.print(f"\n[bold][{r['score']:.3f}][/] [dim]({r['source']})[/] [{r['source_type']}]")
            console.print(f"  {r['text'][:200]}")
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")

# ─── Entities ────────────────────────────────────────────────────────────────


def cmd_entities(args: argparse.Namespace):
    try:
        entities = api.list_entities(type=args.type, q=args.query,
                                      limit=args.limit, offset=args.offset)
        if not entities:
            console.print("[yellow]Aucune entité[/]")
            return
        table = Table(title=f"Entités ({args.type or 'tous'})")
        table.add_column("Type", style="cyan")
        table.add_column("Nom", style="white")
        table.add_column("Confiance", style="yellow")
        for e in entities:
            table.add_row(e.get("_type", ""), e.get("name", "")[:40],
                           str(e.get("confidence", "?")))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")

# ─── Entity confirm/deny/rename ──────────────────────────────────────────────


def cmd_entity_confirm(args: argparse.Namespace):
    try:
        entities = api.list_entities(q=args.name)
        for e in entities:
            api.confirm_entity(e["_type"], e["_key"])
            console.print(f"[green]✓ Confirmé[/] {e.get('_type')}: {e.get('name')}")
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")


def cmd_entity_deny(args: argparse.Namespace):
    try:
        entities = api.list_entities(q=args.name)
        for e in entities:
            api.deny_entity(e["_type"], e["_key"], reason=args.reason)
            console.print(f"[red]✗ Refusée[/] {e.get('_type')}: {e.get('name')}")
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")


def cmd_entity_rename(args: argparse.Namespace):
    try:
        entities = api.list_entities(q=args.name)
        for e in entities:
            api.rename_entity(e["_type"], e["_key"], args.new_name)
            console.print(f"[green]✓ Renommée[/] {e.get('name')} → {args.new_name}")
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")

# ─── Sync ────────────────────────────────────────────────────────────────────


def cmd_sync(args: argparse.Namespace):
    pending = sync_mgr.list_pending()
    if not pending:
        console.print("[green]Rien à synchroniser[/]")
        return
    console.print(f"[yellow]{len(pending)} élément(s) en attente...[/]")
    ok, fail = sync_mgr.sync_all(api)
    console.print(f"[green]{ok} synchronisés[/]" + (f", [red]{fail} échoués[/]" if fail else ""))

# ─── Rules / Corrections ─────────────────────────────────────────────────────


def cmd_rules(args: argparse.Namespace):
    try:
        rules = api.list_rules()
        if not rules:
            console.print("[yellow]Aucune règle[/]")
            return
        table = Table(title="Règles apprises")
        table.add_column("Pattern", style="cyan")
        table.add_column("Label", style="green")
        table.add_column("Échantillons", style="yellow")
        table.add_column("Rejet", style="red")
        table.add_column("Auto", style="blue")
        for r in rules:
            table.add_row(
                r.get("pattern_type", ""),
                r.get("entity_label", ""),
                str(r.get("samples", 0)),
                f"{r.get('rejection_rate', 0):.0%}",
                "✓" if r.get("auto_apply") else "✗",
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Erreur: {e}[/]")


# ─── CLI setup ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="notes-graph client — Enregistrement, OCR, ingestion",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("health", help="Vérifier la connexion au serveur")

    p = sub.add_parser("record", help="Enregistrer depuis le micro")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--language", "-l", default=None)
    p.add_argument("--no-graph", action="store_true", help="Ne pas construire le graphe")

    p = sub.add_parser("ingest", help="Ingérer un document")
    p.add_argument("file", help="Chemin du fichier ou dossier")
    p.add_argument("--language", "-l", default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("ocr", help="Capture + OCR + envoi")
    p.add_argument("--region", action="store_true", help="Sélectionner une zone")
    p.add_argument("--language", default=None, help="Langue OCR (fra, eng)")
    p.add_argument("--ingest", action="store_true", help="Envoyer au serveur")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--no-clipboard", action="store_true")

    p = sub.add_parser("sessions", help="Lister les sessions")

    p = sub.add_parser("search", help="Recherche sémantique")
    p.add_argument("query", help="Requête")
    p.add_argument("--top-k", type=int, default=5)

    p = sub.add_parser("entities", help="Lister les entités")
    p.add_argument("--type", default=None, choices=["Person", "Organization", "Location", "Event"])
    p.add_argument("--query", "-q", default="", help="Filtrer par nom")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)

    p = sub.add_parser("confirm", help="Confirmer une entité")
    p.add_argument("name")

    p = sub.add_parser("deny", help="Refuser une entité")
    p.add_argument("name")
    p.add_argument("--reason", "-r", default="")

    p = sub.add_parser("rename", help="Renommer une entité")
    p.add_argument("name")
    p.add_argument("new_name")

    p = sub.add_parser("sync", help="Synchroniser le cache local")

    p = sub.add_parser("rules", help="Afficher les règles apprises")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cmds = {
        "health": cmd_health,
        "record": cmd_record,
        "ingest": cmd_ingest,
        "ocr": cmd_ocr,
        "sessions": cmd_sessions,
        "search": cmd_search,
        "entities": cmd_entities,
        "confirm": cmd_entity_confirm,
        "deny": cmd_entity_deny,
        "rename": cmd_entity_rename,
        "sync": cmd_sync,
        "rules": cmd_rules,
    }
    fn = cmds.get(args.command)
    if fn:
        fn(args)


if __name__ == "__main__":
    main()
