from __future__ import annotations

import sys

from config import config
from rag.query_engine import query as rag_query
from rag.retriever import HybridRetriever


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        _run_stdio()
    else:
        _run_cli()


def _run_cli():
    print(f"🧠 Serveur MCP {config.mcp.server_name}")
    print(f"   LLM: {config.rag.llm_model} | Embeddings: {config.rag.embed_model}")
    print(f"   Graphe: {'activé' if config.rag.use_graph_enrichment else 'désactivé'}")
    print()
    print("  Commandes disponibles:")
    print("    question <texte>   → RAG complet avec LLM")
    print("    rechercher <texte> → Recherche sémantique uniquement")
    print("    sessions           → Lister les sessions disponibles")
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
        if line.startswith("question "):
            q = line[len("question "):]
            _handle_question(q)
        elif line.startswith("rechercher "):
            q = line[len("rechercher "):]
            _handle_search(q)
        elif line == "sessions":
            _handle_sessions()
        else:
            print(f"  Commande inconnue: {line}")


def _handle_question(question: str):
    print(f"\n  Question: {question}")
    result = rag_query(question)
    print(f"\n  Réponse:\n{result.answer}\n")
    if result.sources:
        print("  Sources:")
        for s in result.sources:
            print(f"    [{s['score']}] {s['source'][:60]}")


def _handle_search(question: str):
    retriever = HybridRetriever()
    chunks = retriever.search_only(question)
    print(f"\n  Résultats pour: {question}\n")
    for c in chunks:
        source = c.metadata.get("source", "?")
        print(f"  [{c.score:.3f}] ({source})")
        print(f"  {c.text[:200]}...\n")


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
                name="question",
                description="Pose une question sur les transcriptions et documents ingérés. "
                "Utilise la recherche sémantique + graphe de connaissances pour répondre.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "La question à poser",
                        }
                    },
                    "required": ["question"],
                },
            ),
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
                name="sessions",
                description="Liste les sessions (transcriptions/documents) disponibles.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "question":
            question = arguments.get("question", "")
            if not question:
                return [TextContent(type="text", text="❌ Question vide")]
            result = rag_query(question)
            text = result.answer
            if result.sources:
                text += "\n\n**📚 Sources :**\n"
                for s in result.sources:
                    text += f"- [{s['score']}] `{s['source'][:60]}`\n"
            return [TextContent(type="text", text=text)]

        elif name == "rechercher":
            requete = arguments.get("requete", "")
            top_k = arguments.get("top_k", 5)
            if not requete:
                return [TextContent(type="text", text="❌ Requête vide")]
            retriever = HybridRetriever()
            chunks = retriever.search_only(requete, top_k=top_k)
            if not chunks:
                return [TextContent(type="text", text="Aucun résultat trouvé.")]
            lines = [f"**Résultats pour:** _{requete}_\n"]
            for c in chunks:
                source = c.metadata.get("source", "?")
                lines.append(f"**[{c.score:.3f}]** ({source})")
                lines.append(c.text[:500])
                lines.append("")
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

        return [TextContent(type="text", text=f"❌ Outil inconnu: {name}")]

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main_stdio()
