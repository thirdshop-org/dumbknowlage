from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from audio.loader import load_audio
from audio.recorder import AudioRecorder
from config import config
from storage.json_exporter import build_export_payload, export_analysis, export_session
from storage.sqlite_store import SQLiteStore

console = Console()


def cmd_record(args: argparse.Namespace):
    """Enregistrer l'audio depuis le micro et transcrire en temps réel."""
    from transcription.transcriber import Transcriber

    store = SQLiteStore()
    store.connect()

    transcriber = Transcriber(model_name=args.model)
    recorder = AudioRecorder()

    duration = args.duration
    lang = args.language or "fr"

    console.print(Panel(f"[bold yellow]Enregistrement de {duration}s[/]"))

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold green]Enregistrement..."),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("record", total=None)
        audio = recorder.record_duration(duration)

    console.print(f"[green]✓ Enregistré:[/] {duration}s [dim]| {len(audio)/16000:.1f} samples[/]")
    console.print(f"[dim]Modèle Whisper: {transcriber.model.model_name} | Langue: {lang}[/]")

    session_id = store.create_session(
        source=f"microphone_{time.strftime('%Y%m%d_%H%M%S')}",
        language=lang,
        model=transcriber.model.model_name,
        duration=duration,
    )

    console.print("[green]Transcription en cours...[/]")

    chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcription par chunks...", total=None)

        def on_progress(current, total, text):
            progress.update(task, description=f"[cyan]Chunk {current}/{total}: {text[:50]}...")
            progress.update(task, total=total, completed=current)

        chunks = transcriber.transcribe_chunks(audio, language=lang, progress_callback=on_progress)

    # Print results
    console.print("\n[bold green]Transcription terminée ![/]")
    for c in chunks:
        console.print(f"  [{c['chunk_index']}] [dim]{c['start_time']:.1f}s-{c['end_time']:.1f}s[/] {c['text']}")

    store.insert_chunks_batch(session_id, chunks)
    console.print(f"\n[bold]Session ID:[/] {session_id}")

    if args.no_save:
        return

    # Analyse NLP automatique
    _run_nlp_pipeline(store, session_id, chunks, lang, args.build_graph)


def cmd_transcribe(args: argparse.Namespace):
    """Transcrire un fichier audio."""
    from transcription.transcriber import Transcriber

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]Fichier introuvable: {file_path}[/]")
        sys.exit(1)

    store = SQLiteStore()
    store.connect()

    lang = args.language or "fr"
    transcriber = Transcriber(model_name=args.model)

    console.print(f"[yellow]Chargement:[/] {file_path}")
    audio = load_audio(str(file_path))

    duration = len(audio) / config.audio.sample_rate
    console.print(f"[yellow]Durée:[/] {duration:.1f}s | [yellow]Modèle:[/] {transcriber.model.model_name}")

    session_id = store.create_session(
        source=str(file_path),
        language=lang,
        model=transcriber.model.model_name,
        duration=duration,
    )

    chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcription par chunks...", total=None)

        def on_progress(current, total, text):
            progress.update(task, description=f"[cyan]Chunk {current}/{total}: {text[:50]}...")
            progress.update(task, total=total, completed=current)

        chunks = transcriber.transcribe_chunks(audio, language=lang, progress_callback=on_progress)

    console.print("\n[bold green]Transcription terminée ![/]")
    for c in chunks:
        console.print(f"  [{c['chunk_index']}] [dim]{c['start_time']:.1f}s-{c['end_time']:.1f}s[/] {c['text']}")

    store.insert_chunks_batch(session_id, chunks)
    console.print(f"\n[bold]Session ID:[/] {session_id}")

    # Analyse NLP
    _run_nlp_pipeline(store, session_id, chunks, lang, args.build_graph)


def _run_nlp_pipeline(
    store: SQLiteStore,
    session_id: str,
    chunks: list[dict],
    lang: str,
    build_graph: bool,
    doc_metadata: dict | None = None,
):
    console.print("\n[bold cyan]Analyse NLP...[/]")

    full_text = " ".join(c["text"] for c in chunks)

    # spaCy
    console.print("  [dim]→ Analyse spaCy (POS, entités, dépendances)...[/]")
    from nlp.spacy_analyzer import SpacyAnalyzer

    spacy_analyzer = SpacyAnalyzer()
    spacy_result = spacy_analyzer.analyze(full_text, lang=lang)
    store.save_analysis(session_id, "spacy", spacy_result)

    # Extraction fréquences
    console.print("  [dim]→ Extraction fréquences et co-occurrences...[/]")
    from nlp.extractor import (
        compute_tfidf,
        detect_burst_topics,
        extract_co_occurrences,
        extract_frequencies,
    )

    frequencies = extract_frequencies(spacy_result["tokens"])
    co_occurrences = extract_co_occurrences(spacy_result["tokens"])

    store.save_analysis(session_id, "frequencies", {"frequencies": frequencies})
    store.save_analysis(session_id, "co_occurrences", {"co_occurrences": co_occurrences})

    # TF-IDF
    chunk_texts = [c["text"] for c in chunks]
    tfidf_result = compute_tfidf(chunk_texts)
    store.save_analysis(session_id, "tfidf", {"tfidf": tfidf_result})

    # Burst detection (hot topics)
    burst_topics = detect_burst_topics(chunk_texts)
    store.save_analysis(session_id, "burst_topics", {"burst_topics": burst_topics})

    # CamemBERT (sujets profonds)
    console.print("  [dim]→ Analyse CamemBERT (embeddings, sujets)...[/]")
    from nlp.camembert_analyzer import CamembertAnalyzer

    camembert = CamembertAnalyzer()
    topics = camembert.extract_topics(full_text)
    store.save_analysis(session_id, "camembert_topics", {"topics": topics})

    # Affichage
    _display_analysis(frequencies, co_occurrences, tfidf_result, burst_topics, topics)

    # Graphe
    if build_graph:
        _build_graph(session_id, spacy_result, chunks, co_occurrences, topics, doc_metadata)


def _display_analysis(frequencies, co_occurrences, tfidf, burst_topics, topics):
    # Fréquences
    table = Table(title="Top mots fréquents")
    table.add_column("Mot", style="cyan")
    table.add_column("Fréquence", style="yellow")
    for word, count in frequencies[:10]:
        table.add_row(word, str(count))
    console.print(table)

    # Co-occurrences
    if co_occurrences:
        table2 = Table(title="Top co-occurrences")
        table2.add_column("Paire", style="cyan")
        table2.add_column("Fréquence", style="yellow")
        for (a, b), count in co_occurrences[:10]:
            table2.add_row(f"{a} ↔ {b}", str(count))
        console.print(table2)

    # TF-IDF
    if tfidf:
        table3 = Table(title="Top termes TF-IDF")
        table3.add_column("Terme", style="cyan")
        table3.add_column("Score", style="yellow")
        for item in tfidf[:10]:
            table3.add_row(item["term"], f"{item['score']:.3f}")
        console.print(table3)

    # Burst topics
    if burst_topics:
        table4 = Table(title="Hot topics détectés (burst)")
        table4.add_column("Mot", style="red")
        table4.add_column("Score burst", style="yellow")
        table4.add_column("Fenêtre", style="dim")
        for bt in burst_topics[:5]:
            table4.add_row(bt["word"], f"{bt['burst_score']:.2f}", str(bt["window_start"]))
        console.print(table4)

    # Topics CamemBERT
    if topics:
        console.print("[bold]Sujets extraits (CamemBERT):[/]")
        for t in topics:
            console.print(f"  • [cyan]{t['sentence'][:80]}...[/]")


def _build_graph(session_id, spacy_result, chunks, co_occurrences, topics,
                 doc_metadata: dict | None = None):
    console.print("  [bold yellow]Construction du graphe ArangoDB...[/]")
    from graph.arango_client import GraphManager
    from graph.entity_models import entity_from_label
    from graph.models import DocumentNode, SentenceNode, TopicNode, WordNode, sanitize_key

    gm = GraphManager()
    if not gm.connect():
        console.print("[red]Impossible de connecter ArangoDB. Vérifiez docker-compose.[/]")
        return

    # Document node
    if doc_metadata:
        doc_node = DocumentNode(
            filename=doc_metadata.get("filename", f"session_{session_id}"),
            session_id=session_id,
            title=doc_metadata.get("title", ""),
            author=doc_metadata.get("author", ""),
            pages=doc_metadata.get("pages", 0),
            file_type=doc_metadata.get("file_type", ""),
            word_count=doc_metadata.get("word_count", 0),
        )
    else:
        doc_node = DocumentNode(
            filename=f"session_{session_id}",
            session_id=session_id,
        )
    gm.insert_document(doc_node)

    # Words
    word_keys = {}
    for t in spacy_result["tokens"]:
        if t["is_punct"] or t["is_stop"]:
            continue
        is_entity = any(e["text"].lower() == t["text"].lower() for e in spacy_result["entities"])
        entity_label = ""
        if is_entity:
            for e in spacy_result["entities"]:
                if e["text"].lower() == t["text"].lower():
                    entity_label = e["label"]
                    break
        w = WordNode(
            lemma=t["lemma"],
            pos=t["pos"],
            language="fr",
            is_entity=is_entity,
            entity_label=entity_label,
        )
        gm.upsert_word(w)
        word_keys[t["lemma"]] = w._key

    # Relations de dépendance syntaxique
    for rel in spacy_result.get("relations", []):
        if rel["lemma"] in word_keys and rel["head_lemma"] in word_keys:
            gm.create_dependency(rel["lemma"], rel["head_lemma"], rel["dep"])

    # Co-occurrences
    for (a, b), count in co_occurrences:
        gm.create_co_occurrence(a, b, count)

    # --- Entités typées ---
    entity_map: dict[str, tuple[str, str]] = {}  # text_lower → (type, key)

    for ent in spacy_result["entities"]:
        entity = entity_from_label(ent["label"], ent["text"])
        if entity is None:
            continue
        gm.upsert_entity(entity)
        gm.create_appears_in(entity.collection, entity._key, doc_node._key)
        entity_map[ent["text"].lower()] = (entity.collection, entity._key)
        # Link entity word in Word collection to the typed entity
        word_key = sanitize_key(ent["text"])
        if word_key in word_keys:
            gm.create_entity_edge(
                "Word", word_key,
                entity.collection, entity._key,
                gm.cfg.edge_is_similar,
                "IS_ENTITY",
            )

    # Inférer relations entre entités via dépendances syntaxiques
    for rel in spacy_result.get("relations", []):
        word_lower = rel["lemma"]
        head_lower = rel["head_lemma"]
        w_info = entity_map.get(word_lower)
        h_info = entity_map.get(head_lower)
        if w_info and h_info:
            w_type, w_key = w_info
            h_type, h_key = h_info
            # WORKS_FOR: Person → Organization
            if w_type == "Person" and h_type == "Organization":
                gm.create_works_for(w_key, h_key)
            elif w_type == "Organization" and h_type == "Person":
                gm.create_works_for(h_key, w_key)
            # LOCATED_IN: anything → Location
            if h_type == "Location":
                gm.create_located_in(w_type, w_key, h_key)
            elif w_type == "Location":
                gm.create_located_in(h_type, h_key, w_key)
            # RELATED_TO: entities in same dependency
            if w_type != h_type:
                gm.create_related_to(w_type, w_key, h_type, h_key, weight=0.5)

    # Co-occurrence d'entités dans les mêmes chunks → RELATED_TO
    if entity_map:
        seen_pairs = set()
        for c in chunks:
            chunk_entities = []
            for e_text, (e_type, e_key) in entity_map.items():
                if e_text in c["text"].lower():
                    chunk_entities.append((e_type, e_key))
            for i in range(len(chunk_entities)):
                for j in range(i + 1, len(chunk_entities)):
                    t1, k1 = chunk_entities[i]
                    t2, k2 = chunk_entities[j]
                    pair = (k1, k2) if k1 < k2 else (k2, k1)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        gm.create_related_to(t1, k1, t2, k2, weight=1.0)

    # Sentences
    for c in chunks:
        sn = SentenceNode(
            text=c["text"],
            timestamp=c.get("start_time", 0),
            session_id=session_id,
            chunk_index=c["chunk_index"],
        )
        gm.insert_sentence(sn)
        for t in spacy_result["tokens"]:
            if t["lemma"] in word_keys:
                gm.create_sentence_word_link(sn._key, t["lemma"])

    # Topics
    if topics:
        for i, t in enumerate(topics):
            label = t["sentence"][:50]
            tn = TopicNode(label=label, weight=1.0 / (i + 1))
            gm.upsert_topic(tn)
            gm.create_sentence_topic_link(sn._key, tn._key)

    console.print("[green]Graphe construit avec succès ![/]")
    console.print("[dim]Accès: http://localhost:8529 (root / whispernlp)[/]")
    gm.close()


def cmd_sessions(args: argparse.Namespace):
    """Lister les sessions."""
    store = SQLiteStore()
    store.connect()
    sessions = store.get_all_sessions()

    if not sessions:
        console.print("[yellow]Aucune session trouvée.[/]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Durée", style="yellow")
    table.add_column("Langue", style="green")
    table.add_column("Modèle", style="blue")
    table.add_column("Date", style="dim")

    for s in sessions:
        dur = f"{s['duration']:.1f}s" if s['duration'] else "-"
        table.add_row(s["id"], s["source"][:30], dur, s["language"] or "-",
                       s["model"] or "-", s["created_at"][:19])
    console.print(table)


def cmd_show(args: argparse.Namespace):
    """Afficher le détail d'une session."""
    store = SQLiteStore()
    store.connect()

    session = store.get_session(args.session_id)
    if not session:
        console.print(f"[red]Session introuvable: {args.session_id}[/]")
        return

    chunks = store.get_chunks(args.session_id)

    console.print(Panel(f"[bold]Session: {session['id']}[/]"))
    console.print(f"  Source: {session['source']}")
    console.print(f"  Durée: {session['duration']:.1f}s" if session['duration'] else "")
    console.print(f"  Langue: {session['language']}")
    console.print(f"  Modèle: {session['model']}")
    console.print(f"  Date: {session['created_at']}")

    for c in chunks:
        console.print(f"  [{c['chunk_index']}] [dim]{c['start_time']:.1f}s-{c['end_time']:.1f}s[/] {c['text']}")

    # Show analysis if available
    for analysis in store.get_analysis(args.session_id):
        analysis_type = analysis["analysis_type"]
        if analysis_type == "frequencies":
            console.print(f"\n[bold cyan]Fréquences:[/]")
            for word, count in analysis["result"]["frequencies"][:10]:
                console.print(f"  {word}: {count}")
        elif analysis_type == "burst_topics":
            console.print(f"\n[bold red]Hot topics:[/]")
            for bt in analysis["result"]["burst_topics"][:5]:
                console.print(f"  {bt['word']} (score: {bt['burst_score']:.2f})")
        elif analysis_type == "tfidf":
            console.print(f"\n[bold]TF-IDF top:[/]")
            for item in analysis["result"]["tfidf"][:5]:
                console.print(f"  {item['term']}: {item['score']:.3f}")


def cmd_graph_query(args: argparse.Namespace):
    """Requêter le graphe ArangoDB."""
    from graph.arango_client import GraphManager

    gm = GraphManager()
    if not gm.connect():
        console.print("[red]Impossible de connecter ArangoDB.[/]")
        return

    results = gm.query(args.query)
    table = Table(title="Résultats de la requête")
    if results:
        for key in results[0].keys():
            table.add_column(key, style="cyan")
        for row in results:
            table.add_row(*[str(v)[:40] for v in row.values()])
        console.print(table)
    else:
        console.print("[yellow]Aucun résultat.[/]")

    gm.close()


def cmd_graph_top(args: argparse.Namespace):
    """Afficher les top mots du graphe."""
    from graph.arango_client import GraphManager

    gm = GraphManager()
    if not gm.connect():
        console.print("[red]Impossible de connecter ArangoDB.[/]")
        return

    words = gm.get_top_words(limit=args.limit)
    table = Table(title="Top mots (graphe)")
    table.add_column("Mot", style="cyan")
    table.add_column("POS", style="green")
    table.add_column("Fréquence", style="yellow")
    table.add_column("Entité", style="blue")
    for w in words:
        table.add_row(w["word"], w["pos"], str(w["frequency"]), "✓" if w["is_entity"] else "")
    console.print(table)

    gm.close()


def cmd_entities(args: argparse.Namespace):
    """Lister les entités du graphe."""
    from graph.arango_client import GraphManager

    gm = GraphManager()
    if not gm.connect():
        console.print("[red]Impossible de connecter ArangoDB.[/]")
        return

    entities = gm.get_entities(entity_type=args.type, limit=args.limit)

    if not entities:
        console.print("[yellow]Aucune entité trouvée.[/]")
        return

    table = Table(title=f"Entités ({args.type or 'tous'})")
    table.add_column("Type", style="cyan")
    table.add_column("Nom", style="white")
    table.add_column("Mentions", style="yellow")
    table.add_column("Détail", style="dim")
    for e in entities:
        detail = e.get("title") or e.get("loc_type") or e.get("domain") or ""
        table.add_row(e.get("_type", ""), e.get("name", "")[:40],
                       str(e.get("mentions", 0)), detail[:30])
    console.print(table)
    gm.close()


def cmd_entity(args: argparse.Namespace):
    """Afficher le détail d'une entité et son réseau."""
    from graph.arango_client import GraphManager

    gm = GraphManager()
    if not gm.connect():
        console.print("[red]Impossible de connecter ArangoDB.[/]")
        return

    # Search entity by name across all entity collections
    entities = gm.get_entities(limit=200)
    matches = [e for e in entities if args.name.lower() in e.get("name", "").lower()]

    if not matches:
        console.print(f"[yellow]Aucune entité trouvée pour: {args.name}[/]")
        return

    for ent in matches:
        e_type = ent.get("_type", "")
        e_name = ent.get("name", "")
        e_key = ent.get("_key", "")
        console.print(Panel(f"[bold]{e_type}: {e_name}[/]"))

        # Documents liés
        docs = gm.get_entity_documents(e_type, e_key)
        if docs:
            console.print("  [cyan]Documents:[/]")
            for d in docs:
                console.print(f"    • {d.get('title', d['id'])}")

        # Réseau de connexions
        network = gm.get_entity_network(e_type, e_key, depth=args.depth)
        if network:
            console.print(f"  [cyan]Connexions (profondeur {args.depth}):[/]")
            for n in network[:15]:
                name = n.get("name", n.get("entity", ""))
                rel = n.get("relation", "")
                console.print(f"    → {name} [dim]({rel})[/]")
        console.print("")

    gm.close()


def cmd_export(args: argparse.Namespace):
    """Exporter une session en JSON."""
    store = SQLiteStore()
    store.connect()

    session = store.get_session(args.session_id)
    if not session:
        console.print(f"[red]Session introuvable: {args.session_id}[/]")
        return

    chunks = store.get_chunks(args.session_id)

    frequencies = []
    co_occurrences = []
    tfidf = []
    burst_topics = []

    for analysis in store.get_analysis(args.session_id):
        atype = analysis["analysis_type"]
        if atype == "frequencies":
            frequencies = analysis["result"].get("frequencies", [])
        elif atype == "co_occurrences":
            co_occurrences = analysis["result"].get("co_occurrences", [])
        elif atype == "tfidf":
            tfidf = analysis["result"].get("tfidf", [])
        elif atype == "burst_topics":
            burst_topics = analysis["result"].get("burst_topics", [])

    payload = build_export_payload(
        session=session,
        chunks=chunks,
        frequencies=frequencies,
        co_occurrences=co_occurrences,
        tfidf=tfidf,
        burst_topics=burst_topics,
    )

    output = args.output or f"export_{args.session_id}.json"
    path = export_analysis(payload, output)
    console.print(f"[green]Exporté vers: {path}[/]")


def cmd_pipeline(args: argparse.Namespace):
    """Pipeline complet: transcription + NLP + graphe."""
    from transcription.transcriber import Transcriber

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]Fichier introuvable: {file_path}[/]")
        sys.exit(1)

    store = SQLiteStore()
    store.connect()

    lang = args.language or "fr"
    transcriber = Transcriber(model_name=args.model)

    console.print(f"[bold yellow]=== Pipeline complet ===[/]")
    console.print(f"[yellow]Fichier:[/] {file_path}")
    console.print(f"[yellow]Modèle:[/] {transcriber.model.model_name}")
    console.print(f"[yellow]Construire le graphe:[/] {'oui' if args.build_graph else 'non'}")

    audio = load_audio(str(file_path))
    duration = len(audio) / config.audio.sample_rate

    session_id = store.create_session(
        source=str(file_path),
        language=lang,
        model=transcriber.model.model_name,
        duration=duration,
    )

    chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcription...", total=None)

        def on_progress(current, total, text):
            progress.update(task, description=f"[cyan]Chunk {current}/{total}: {text[:50]}...")
            progress.update(task, total=total, completed=current)

        chunks = transcriber.transcribe_chunks(audio, language=lang, progress_callback=on_progress)

    store.insert_chunks_batch(session_id, chunks)
    _run_nlp_pipeline(store, session_id, chunks, lang, args.build_graph)

    # Export automatique
    output = args.output or f"pipeline_{session_id}.json"
    frequencies = []
    co_occurrences = []
    tfidf = []
    burst_topics = []
    for analysis in store.get_analysis(session_id):
        atype = analysis["analysis_type"]
        if atype == "frequencies":
            frequencies = analysis["result"].get("frequencies", [])
        elif atype == "co_occurrences":
            co_occurrences = analysis["result"].get("co_occurrences", [])
        elif atype == "tfidf":
            tfidf = analysis["result"].get("tfidf", [])
        elif atype == "burst_topics":
            burst_topics = analysis["result"].get("burst_topics", [])

    payload = build_export_payload(
        session={"id": session_id, "source": str(file_path), "duration": duration, "language": lang},
        chunks=chunks,
        frequencies=frequencies,
        co_occurrences=co_occurrences,
        tfidf=tfidf,
        burst_topics=burst_topics,
    )
    export_analysis(payload, output)
    console.print(f"[green]Export JSON: {output}[/]")

    console.print(f"\n[bold green]Pipeline terminé ! Session ID: {session_id}[/]")


INGEST_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".docx"}


def _ingest_file(file_path: Path, store: SQLiteStore, lang: str, build_graph: bool) -> str | None:
    from document.reader import chunk_text, get_metadata, read_file

    try:
        with console.status(f"[bold green]Lecture {file_path.name}..."):
            text = read_file(file_path)
        meta = get_metadata(file_path, text)
        chunks = chunk_text(text)

        session_id = store.create_session(
            source=str(file_path),
            language=lang,
            model=f"document/{meta['file_type']}",
        )
        store.insert_chunks_batch(session_id, chunks)
        store.insert_document(
            session_id=session_id,
            filename=str(file_path),
            title=meta["title"],
            author=meta["author"],
            pages=meta["pages"],
            file_type=meta["file_type"],
            word_count=meta["word_count"],
        )

        _run_nlp_pipeline(
            store, session_id, chunks, lang,
            build_graph=build_graph,
            doc_metadata=meta | {"filename": str(file_path)},
        )
        return session_id
    except Exception as e:
        console.print(f"  [red]✗ {file_path.name}: {e}[/]")
        return None


def cmd_ingest(args: argparse.Namespace):
    """Ingérer un document ou un dossier de documents dans la base de mémoire."""
    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]Introuvable: {path}[/]")
        sys.exit(1)

    store = SQLiteStore()
    store.connect()
    lang = args.language or "fr"

    if path.is_dir():
        supported: list[Path] = []
        for p in path.rglob("*") if args.recursive else path.glob("*"):
            if p.suffix.lower() in INGEST_EXTENSIONS and p.is_file():
                supported.append(p)
        supported.sort()

        if not supported:
            console.print(f"[yellow]Aucun document supporté trouvé dans {path}[/]")
            console.print(f"[dim]Formats: {', '.join(sorted(INGEST_EXTENSIONS))}[/]")
            sys.exit(0)

        console.print(Panel(f"[bold yellow]Dossier: {path}[/] ({len(supported)} fichiers)"))
        console.print(f"[dim]Récursif: {'oui' if args.recursive else 'non'} | Langue: {lang}[/]\n")

        results: list[tuple[str, str | None]] = []
        for i, fp in enumerate(supported, 1):
            short = fp.relative_to(path) if args.recursive else fp.name
            console.print(f"[{i}/{len(supported)}] {short}")
            sid = _ingest_file(fp, store, lang, args.build_graph)
            results.append((short, sid))

        ok = sum(1 for _, s in results if s)
        fail = len(results) - ok

        console.print(f"\n[bold]Résumé dossier:[/] {ok} ✓ / {fail} ✗")
        if fail:
            for name, sid in results:
                if sid is None:
                    console.print(f"  [red]✗ {name}[/]")
    else:
        if path.suffix.lower() not in INGEST_EXTENSIONS:
            console.print(f"[red]Format non supporté: {path.suffix}[/]")
            console.print(f"[dim]Supporté: {', '.join(sorted(INGEST_EXTENSIONS))}[/]")
            sys.exit(1)

        console.print(Panel(f"[bold yellow]Ingestion: {path.name}[/]"))
        sid = _ingest_file(path, store, lang, args.build_graph)
        if sid:
            console.print(f"\n[bold green]✓ Document ingéré ! Session ID: {sid}[/]")


def main():
    parser = argparse.ArgumentParser(
        description="whisper-nlp-graph — Transcription Whisper + NLP + Graphe de connaissances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Enregistrer 60s depuis le micro
  python main.py record --duration 60 --lang fr --build-graph

  # Transcrire un fichier audio
  python main.py transcribe audio.mp3 --lang fr

  # Pipeline complet
  python main.py pipeline audio.mp3 --lang fr --build-graph

  # Ingérer un document dans la base mémoire
  python main.py ingest document.pdf --build-graph
  python main.py ingest notes.md --lang fr --build-graph
  python main.py ingest rapport.docx --build-graph

  # Ingérer tout un dossier de documents
  python main.py ingest ./documents/ --build-graph
  python main.py ingest ./docs/ --recursive --build-graph

  # Lister les entités du graphe
  python main.py entities
  python main.py entities --type Person
  python main.py entities --type Organization

  # Voir le réseau d'une entité
  python main.py entity "Dr. Martin"
  python main.py entity "Pasteur" --depth 3

  # Requêter le graphe ArangoDB
  python main.py graph query "FOR w IN Word SORT w.frequency DESC LIMIT 10 RETURN w"

  # Afficher les top mots du graphe
  python main.py graph top --limit 20

  # Lister les sessions
  python main.py sessions

  # Afficher une session
  python main.py show <session_id>

  # Exporter une session en JSON
  python main.py export <session_id> --output export.json
        """,
    )
    parser.add_argument("--model", default="turbo", choices=["turbo", "large-v3"],
                        help="Modèle Whisper (défaut: turbo)")

    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # record
    record_parser = subparsers.add_parser("record", help="Enregistrer depuis le micro")
    record_parser.add_argument("--duration", type=float, default=30.0, help="Durée d'enregistrement en secondes")
    record_parser.add_argument("--language", "-l", default=None, help="Langue (fr, en, etc.)")
    record_parser.add_argument("--build-graph", action="store_true", help="Construire le graphe après analyse")
    record_parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder")

    # transcribe
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcrire un fichier audio")
    transcribe_parser.add_argument("file", help="Chemin du fichier audio (.wav, .mp3)")
    transcribe_parser.add_argument("--language", "-l", default=None, help="Langue")
    transcribe_parser.add_argument("--build-graph", action="store_true", help="Construire le graphe")

    # pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline complet: transcription + NLP + graphe")
    pipeline_parser.add_argument("file", help="Chemin du fichier audio")
    pipeline_parser.add_argument("--language", "-l", default=None, help="Langue")
    pipeline_parser.add_argument("--build-graph", action="store_true", help="Construire le graphe")
    pipeline_parser.add_argument("--output", "-o", default=None, help="Fichier de sortie JSON")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingérer un document (PDF/TXT/DOCX/MD) dans la base mémoire")
    ingest_parser.add_argument("file", help="Chemin du fichier ou dossier")
    ingest_parser.add_argument("--language", "-l", default=None, help="Langue (défaut: fr)")
    ingest_parser.add_argument("--build-graph", action="store_true", help="Construire le graphe")
    ingest_parser.add_argument("--recursive", "-r", action="store_true", help="Parcourir les sous-dossiers")

    # sessions
    subparsers.add_parser("sessions", help="Lister les sessions")

    # show
    show_parser = subparsers.add_parser("show", help="Afficher une session")
    show_parser.add_argument("session_id", help="ID de la session")

    # export
    export_parser = subparsers.add_parser("export", help="Exporter une session en JSON")
    export_parser.add_argument("session_id", help="ID de la session")
    export_parser.add_argument("--output", "-o", default=None, help="Fichier de sortie")

    # graph
    graph_parser = subparsers.add_parser("graph", help="Commandes liées au graphe")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")

    graph_query = graph_sub.add_parser("query", help="Requête AQL")
    graph_query.add_argument("query", help="Requête AQL")

    graph_top = graph_sub.add_parser("top", help="Top mots du graphe")
    graph_top.add_argument("--limit", type=int, default=20, help="Nombre de mots")

    # entities
    entities_parser = subparsers.add_parser("entities", help="Lister les entités du graphe")
    entities_parser.add_argument("--type", default=None, choices=["Person", "Organization", "Location", "Event"],
                                 help="Type d'entité (défaut: tous)")
    entities_parser.add_argument("--limit", type=int, default=50, help="Nombre max d'entités")

    # entity
    entity_parser = subparsers.add_parser("entity", help="Afficher le détail d'une entité")
    entity_parser.add_argument("name", help="Nom de l'entité (recherche partielle)")
    entity_parser.add_argument("--depth", type=int, default=2, help="Profondeur du réseau (défaut: 2)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "record": cmd_record,
        "transcribe": cmd_transcribe,
        "pipeline": cmd_pipeline,
        "ingest": cmd_ingest,
        "sessions": cmd_sessions,
        "show": cmd_show,
        "export": cmd_export,
        "entities": cmd_entities,
        "entity": cmd_entity,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    elif args.command == "graph":
        if args.graph_command == "query":
            cmd_graph_query(args)
        elif args.graph_command == "top":
            cmd_graph_top(args)
        else:
            graph_parser.print_help()


if __name__ == "__main__":
    main()
