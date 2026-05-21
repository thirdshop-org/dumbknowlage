from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.sse import SseServerTransport

from config import config


def create_mcp_server() -> Server:
    server = Server(config.mcp.server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="rechercher",
                description="Recherche sémantique dans les transcriptions et documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "requete": {"type": "string", "description": "La requête"},
                        "top_k": {"type": "integer", "description": "Nombre de résultats (défaut: 5)", "default": 5},
                    },
                    "required": ["requete"],
                },
            ),
            Tool(
                name="contexte",
                description="Retourne le contexte complet (chunks + graphe) pour une question.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "La question"},
                        "top_k": {"type": "integer", "description": "Nombre de chunks (défaut: 5)", "default": 5},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="entites",
                description="Cherche des entités par nom dans le graphe.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nom": {"type": "string", "description": "Nom à chercher"},
                        "limit": {"type": "integer", "description": "Max résultats (défaut: 20)", "default": 20},
                    },
                    "required": ["nom"],
                },
            ),
            Tool(
                name="entite_detail",
                description="Affiche le détail d'une entité.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["Person", "Organization", "Location", "Event"]},
                        "key": {"type": "string", "description": "Clé de l'entité"},
                        "depth": {"type": "integer", "description": "Profondeur (défaut: 2)", "default": 2},
                    },
                    "required": ["type", "key"],
                },
            ),
            Tool(
                name="sessions",
                description="Liste les sessions disponibles.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="graph_aql",
                description="Exécute une requête AQL sur le graphe ArangoDB.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Requête AQL"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "rechercher":
            return await _search(arguments)
        elif name == "contexte":
            return await _context(arguments)
        elif name == "entites":
            return await _entities(arguments)
        elif name == "entite_detail":
            return await _entity_detail(arguments)
        elif name == "sessions":
            return await _sessions()
        elif name == "graph_aql":
            return await _graph_aql(arguments)
        return [TextContent(type="text", text=f"Outil inconnu: {name}")]

    return server


async def _search(args: dict) -> list[TextContent]:
    from rag.retriever import HybridRetriever
    requete = args.get("requete", "")
    top_k = args.get("top_k", 5)
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


async def _context(args: dict) -> list[TextContent]:
    from rag.query_engine import query as rag_query
    question = args.get("question", "")
    top_k = args.get("top_k", 5)
    if not question:
        return [TextContent(type="text", text="Question vide")]
    result = rag_query(question, top_k=top_k)
    text = result.context
    if result.sources:
        text += "\n\n**Sources:**\n"
        for s in result.sources:
            text += f"- [{s['score']}] `{s['source']}` ({s['source_type']})\n"
    return [TextContent(type="text", text=text)]


async def _entities(args: dict) -> list[TextContent]:
    from graph.arango_client import GraphManager
    nom = args.get("nom", "")
    limit = args.get("limit", 20)
    if not nom:
        return [TextContent(type="text", text="Nom vide")]
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


async def _entity_detail(args: dict) -> list[TextContent]:
    from graph.arango_client import GraphManager
    e_type = args.get("type", "")
    e_key = args.get("key", "")
    depth = args.get("depth", 2)
    if not e_type or not e_key:
        return [TextContent(type="text", text="Type et key requis")]
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
            doc_name = d.get("filename") or d.get("title") or d["id"]
            lines.append(f"- {doc_name}")
        lines.append("")
    if network:
        lines.append(f"**Connexions (profondeur {depth}):**")
        for n in network[:15]:
            nname = n.get("name", n.get("entity", ""))
            rel = n.get("relation", "")
            lines.append(f"- {nname} ({rel})")
    return [TextContent(type="text", text="\n".join(lines))]


async def _sessions() -> list[TextContent]:
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


async def _graph_aql(args: dict) -> list[TextContent]:
    from graph.arango_client import GraphManager
    query = args.get("query", "")
    if not query:
        return [TextContent(type="text", text="Requête AQL vide")]
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
