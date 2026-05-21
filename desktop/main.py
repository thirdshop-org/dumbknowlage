"""notes-graph Desktop App — PySide6 frontend for the deployed server."""

import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("NOTES_GRAPH_SERVER", "https://dumbknowlage.thirdshop.fr")

# Add project root so client/ and other modules are importable
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Patch: force 'from config import config' to resolve to ClientConfig
import client.config as _client_config  # noqa: E402
sys.modules["config"] = _client_config

from desktop.app import run_app  # noqa: E402

if __name__ == "__main__":
    run_app()
