from __future__ import annotations

import sys

from config import config
from rag.query_engine import query as rag_query, build_context as rag_build_context
from rag.retriever import HybridRetriever


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        _run_stdio()
    else:
        _run_cli()


def _run_cli():
    print(f"Serveur MCP {config.mcp.server_name}")
    print(f"   Embeddings: {config.rag.embed_model}")
    print(f"   Graphe: {'activé' if config.rag.use_graph_enrichment else 'désactivé'}")
    print()
    print("  Commandes disponibles:")
    print("    rechercher <texte> → Recherche sémantique")
    print("    contexte <texte>   → Contexte complet avec graphe")
    print("    entites <nom>      → Chercher des entités")
    print("    entite_detail <type> <key> [profondeur] → Détail d'une entité")
    print("    sessions           → Lister les sessions")
    print("    graph_aql <query>  → Requête AQL")
    print("    exit               → Quitter")
    print()

    while True:
        try:
            line = input("mcp> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line == "exit":
            break
        if line.startswith("rechercher "):
            _handle_search(line[len("rechercher "):])
        elif line.startswith("contexte "):
            _handle_contexte(line[len("contexte "):])
        elif line.startswith("entites "):
            _handle_entites(line[len("entites "):])
        elif line.startswith("entite_detail "):
            _handle_entite_detail(line[len("entite_detail "):])
        elif line == "sessions":
            _handle_sessions()
        elif line.startswith("graph_aql "):
            _handle_graph_aql(line[len("graph_aql "):])
        else:
            print(f"  Commande inconnue: {line}")


def _handle_search(question: str):
    retriever = HybridRetriever()
    chunks = retriever.search_only(question)
    print(f"\n  Résultats pour: {question}\n")
    for c in chunks:
        source = c.metadata.get("source", "?")
        stype = c.metadata.get("source_type", "?")
        print(f"  [{c.score:.3f}] ({source}) [{stype}]")
        print(f"  {c.text[:250]}...\n")


def _handle_contexte(question: str):
    print(f"\n  Contexte pour: {question}")
    result = rag_query(question)
    print(f"\n{result.context}\n")
    if result.sources:
        print("  Sources:")
        for s in result.sources:
            print(f"    [{s['score']}] ({s['source_type']}) {s['source'][:60]}")


def _handle_entites(name: str):
    from graph.arango_client import GraphManager
    gm = GraphManager()
    if not gm.connect():
        print("  [rouge]Impossible de connecter ArangoDB.[/]")
        return
    entities = gm.search_entities(name, limit=20)
    gm.close()
    if not entities:
        print(f"  Aucune entité trouvée pour: {name}")
        return
    print(f"\n  Entités pour: {name}\n")
    for e in entities:
        conf = e.get("confidence", "?")
        print(f"  • {e.get('_type', '?')}: {e.get('name', '?')} (confiance: {conf})")


def _handle_entite_detail(args: str):
    parts = args.split()
    if len(parts) < 2:
        print("  Usage: entite_detail <type> <key> [profondeur]")
        return
    e_type, e_key = parts[0], parts[1]
    depth = int(parts[2]) if len(parts) > 2 else 2

    from graph.arango_client import GraphManager
    gm = GraphManager()
    if not gm.connect():
        print("  [rouge]Impossible de connecter ArangoDB.[/]")
        return
    col = gm.db.collection(e_type)
    ent = col.get(e_key)
    if not ent:
        print(f"  Entité introuvable: {e_type}/{e_key}")
        gm.close()
        return
    docs = gm.get_entity_documents(e_type, e_key)
    network = gm.get_entity_network(e_type, e_key, depth=depth)
    gm.close()

    print(f"\n  {ent.get('_type', e_type)}: {ent.get('name', e_key)}")
    print(f"  Confiance: {ent.get('confidence', '?')}")
    if docs:
        print(f"  Documents ({len(docs)}):")
        for d in docs:
            print(f"    • {d.get('title', d['id'])}")
    if network:
        print(f"  Connexions (profondeur {depth}):")
        for n in network[:15]:
            name = n.get("name", n.get("entity", ""))
            rel = n.get("relation", "")
            print(f"    → {name} ({rel})")


def _handle_sessions():
    from storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    store.connect()
    sessions = store.get_all_sessions()
    if not sessions:
        print("  Aucune session trouvée.")
        return
    print(f"\n  Sessions ({len(sessions)}):")
    for s in sessions:
        print(f"    {s['id']}  {s['source'][:50]}  {s['created_at'][:19]}")
    store.close()


def _handle_graph_aql(aql: str):
    from graph.arango_client import GraphManager
    gm = GraphManager()
    if not gm.connect():
        print("  [rouge]Impossible de connecter ArangoDB.[/]")
        return
    results = gm.query(aql)
    gm.close()
    if not results:
        print("  Aucun résultat.")
        return
    print(f"\n  Résultats ({len(results)}):")
    for row in results[:20]:
        print(f"    {row}")


# --- Mode stdio MCP (pour hôtes comme Claude Desktop) ---

def main_stdio():
    import asyncio
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server

    server = Server(config.mcp.server_name, timeout=300, max_timeout=600)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="rechercher",
                description="Recherche sémantique dans les transcriptions et documents. "
                "Retourne les passages pertinents sans génération LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "requete": {
                            "type": "string",
                            "description": "La requête de recherche",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Nombre de résultats (défaut: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["requete"],
                },
            ),
            Tool(
                name="contexte",
                description="Retourne le contexte complet (chunks + graphe de connaissances) "
                "pour une question. Sans génération LLM — le client utilise ce contexte "
                "pour répondre avec son propre LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "La question à contextuer",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Nombre de chunks (défaut: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="entites",
                description="Cherche des entités (Personnes, Organisations, Lieux, Événements) "
                "par nom dans le graphe de connaissances.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nom": {
                            "type": "string",
                            "description": "Nom ou partie du nom à chercher",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Nombre max de résultats (défaut: 20)",
                            "default": 20,
                        },
                    },
                    "required": ["nom"],
                },
            ),
            Tool(
                name="entite_detail",
                description="Affiche le détail d'une entité : documents liés et réseau de connexions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Type d'entité (Person, Organization, Location, Event)",
                            "enum": ["Person", "Organization", "Location", "Event"],
                        },
                        "key": {
                            "type": "string",
                            "description": "Clé de l'entité (sans accent, sans espace)",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Profondeur du réseau (défaut: 2)",
                            "default": 2,
                        },
                    },
                    "required": ["type", "key"],
                },
            ),
            Tool(
                name="sessions",
                description="Liste les sessions (transcriptions audio, documents ingérés) disponibles.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="graph_aql",
                description="Exécute une requête AQL brute sur le graphe ArangoDB.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Requête AQL",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "rechercher":
            requete = arguments.get("requete", "")
            top_k = arguments.get("top_k", 5)
            if not requete:
                return [TextContent(type="text", text="Requête vide")]
            retriever = HybridRetriever()
            chunks = retriever.search_only(requete, top_k=top_k)
            if not chunks:
                return [TextContent(type="text", text="Aucun résultat trouvé.")]
            lines = [f"**Résultats pour:** _{requete}_\n"]
            for c in chunks:
                source = c.metadata.get("source", "?")
                stype = c.metadata.get("source_type", "?")
                lines.append(f"**[{c.score:.3f}]** ({source}) [{stype}]")
                lines.append(c.text[:500])
                lines.append("")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "contexte":
            question = arguments.get("question", "")
            top_k = arguments.get("top_k", 5)
            if not question:
                return [TextContent(type="text", text="Question vide")]
            result = rag_query(question, top_k=top_k)
            text = result.context
            if result.sources:
                text += "\n\n**Sources:**\n"
                for s in result.sources:
                    text += f"- [{s['score']}] `{s['source']}` ({s['source_type']})\n"
            return [TextContent(type="text", text=text)]

        elif name == "entites":
            nom = arguments.get("nom", "")
            limit = arguments.get("limit", 20)
            if not nom:
                return [TextContent(type="text", text="Nom vide")]
            from graph.arango_client import GraphManager
            gm = GraphManager()
            if not gm.connect():
                return [TextContent(type="text", text="Impossible de connecter ArangoDB.")]
            entities = gm.search_entities(nom, limit=limit)
            gm.close()
            if not entities:
                return [TextContent(type="text", text=f"Aucune entité trouvée pour: {nom}")]
            lines = [f"**Entités pour:** _{nom}_\n"]
            for e in entities:
                conf = e.get("confidence", "?")
                lines.append(f"- `{e.get('_type')}` **{e.get('name')}** (confiance: {conf})")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "entite_detail":
            e_type = arguments.get("type", "")
            e_key = arguments.get("key", "")
            depth = arguments.get("depth", 2)
            if not e_type or not e_key:
                return [TextContent(type="text", text="Type et key requis")]
            from graph.arango_client import GraphManager
            gm = GraphManager()
            if not gm.connect():
                return [TextContent(type="text", text="Impossible de connecter ArangoDB.")]
            col = gm.db.collection(e_type)
            ent = col.get(e_key)
            if not ent:
                gm.close()
                return [TextContent(type="text", text=f"Entité introuvable: {e_type}/{e_key}")]
            name = ent.get("name", e_key)
            conf = ent.get("confidence", "?")
            docs = gm.get_entity_documents(e_type, e_key)
            network = gm.get_entity_network(e_type, e_key, depth=depth)
            gm.close()
            lines = [f"**{e_type}: {name}** (confiance: {conf})\n"]
            if docs:
                lines.append(f"**Documents ({len(docs)}):**")
                for d in docs:
                    lines.append(f"- {d.get('title', d['id'])}")
                lines.append("")
            if network:
                lines.append(f"**Connexions (profondeur {depth}):**")
                for n in network[:15]:
                    nname = n.get("name", n.get("entity", ""))
                    rel = n.get("relation", "")
                    lines.append(f"- {nname} ({rel})")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "sessions":
            from storage.sqlite_store import SQLiteStore
            store = SQLiteStore()
            store.connect()
            sessions = store.get_all_sessions()
            store.close()
            if not sessions:
                return [TextContent(type="text", text="Aucune session trouvée.")]
            lines = [f"**Sessions ({len(sessions)})**\n"]
            for s in sessions:
                dur = f"{s['duration']:.1f}s" if s.get("duration") else "-"
                lines.append(f"- `{s['id']}` {s['source'][:50]} ({dur}) {s['created_at'][:19]}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "graph_aql":
            query = arguments.get("query", "")
            if not query:
                return [TextContent(type="text", text="Requête AQL vide")]
            from graph.arango_client import GraphManager
            gm = GraphManager()
            if not gm.connect():
                return [TextContent(type="text", text="Impossible de connecter ArangoDB.")]
            try:
                results = gm.query(query)
                gm.close()
                if not results:
                    return [TextContent(type="text", text="Aucun résultat.")]
                lines = [f"**Résultats ({len(results)})**\n"]
                for row in results[:20]:
                    lines.append(f"- `{row}`")
                return [TextContent(type="text", text="\n".join(lines))]
            except Exception as e:
                gm.close()
                return [TextContent(type="text", text=f"Erreur AQL: {e}")]

        return [TextContent(type="text", text=f"Outil inconnu: {name}")]

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main_stdio()
