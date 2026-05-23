from __future__ import annotations

import json
import threading
import time
from queue import Queue, Empty

import httpx
import pytest


class _SSESession:
    """Manages a single MCP SSE connection for testing.

    Usage:
        with _SSESession("https://example.com") as sse:
            result = sse.send_request("tools/list")
    """

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._ctx: httpx.StreamContextManager | None = None
        self._response: httpx.Response | None = None
        self._event_queue: Queue = Queue()
        self._reader_thread: threading.Thread | None = None
        self._post_endpoint: str | None = None
        self._req_id: int = 0
        self._closed = False

    def __enter__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self._ctx = self._client.stream("GET", "/mcp")
        self._response = self._ctx.__enter__()
        self._reader_thread = threading.Thread(target=self._read_sse, daemon=True)
        self._reader_thread.start()
        self._wait_for_endpoint()
        return self

    def __exit__(self, *args):
        self._closed = True
        if self._ctx:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
        if self._client:
            self._client.close()

    def _read_sse(self):
        try:
            current_event = ""
            for line in self._response.iter_lines():
                if self._closed:
                    break
                line = line.strip()
                if not line:
                    current_event = ""
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if current_event:
                        self._event_queue.put((current_event, data))
                        current_event = ""
        except Exception:
            pass

    def _wait_for_endpoint(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event, data = self._event_queue.get(timeout=0.5)
                if event == "endpoint":
                    self._post_endpoint = data
                    return
            except Empty:
                continue
        raise TimeoutError("MCP endpoint event not received within timeout")

    def send_request(self, method: str, params: dict | None = None) -> dict:
        assert self._post_endpoint is not None, "MCP endpoint not initialized"
        self._req_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params or {},
        }
        resp = self._client.post(self._post_endpoint, json=body)
        resp.raise_for_status()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                event, data = self._event_queue.get(timeout=0.5)
                parsed = json.loads(data)
                if event == "message" and parsed.get("id") == self._req_id:
                    return parsed
            except Empty:
                continue
        raise TimeoutError(f"No response for MCP method: {method}")


class TestMCP:
    def test_mcp_initialize(self, base_url):
        with _SSESession(base_url) as sse:
            result = sse.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            })
            assert "result" in result
            info = result["result"]
            assert "serverInfo" in info
            assert info["serverInfo"]["name"] == "whisper-nlp-graph"

    def test_mcp_list_tools(self, base_url):
        with _SSESession(base_url) as sse:
            sse.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            })
            result = sse.send_request("tools/list")
            assert "result" in result
            tools = result["result"]["tools"]
            tool_names = {t["name"] for t in tools}
            expected = {"rechercher", "contexte", "entites", "entite_detail", "sessions", "graph_aql"}
            missing = expected - tool_names
            assert not missing, f"Missing tools: {missing}"

    def test_mcp_call_sessions(self, base_url):
        with _SSESession(base_url) as sse:
            sse.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            })
            result = sse.send_request("tools/call", {
                "name": "sessions",
                "arguments": {},
            })
            assert "result" in result
            content = result["result"]["content"]
            assert len(content) > 0
            assert content[0]["type"] == "text"

    def test_mcp_call_rechercher(self, base_url):
        with _SSESession(base_url) as sse:
            sse.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            })
            result = sse.send_request("tools/call", {
                "name": "rechercher",
                "arguments": {"requete": "intelligence artificielle", "top_k": 2},
            })
            assert "result" in result
            content = result["result"]["content"]
            assert len(content) > 0
            text = content[0]["text"]
            assert "Résultats" in text or "Aucun" in text
