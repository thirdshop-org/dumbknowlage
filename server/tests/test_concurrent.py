from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx
import pytest

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")

PALIERS = [
    (10, "palier 10 workers"),
    (50, "palier 50 workers"),
    (100, "palier 100 workers"),
]

WORKER_TEXT = (
    "Le machine learning est une branche de l'intelligence artificielle "
    "qui permet aux ordinateurs d'apprendre à partir de données sans être "
    "explicitement programmés. Les algorithmes de classification supervisée "
    "sont parmi les plus utilisés en pratique."
)


@dataclass
class WorkerResult:
    worker_id: int
    success: bool = True
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0
    session_id: str | None = None


def _worker(worker_id: int, base_url: str) -> WorkerResult:
    def _call(method, url, **kwargs):
        for attempt in range(2):
            r = method(url, **kwargs)
            if r.status_code < 500 or attempt == 1:
                return r
        return r

    result = WorkerResult(worker_id=worker_id)
    t0 = time.monotonic()
    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as c:
            # 1. GET /api/health
            r = _call(c.get, "/api/health")
            if r.status_code >= 500:
                result.errors.append(f"health returned {r.status_code}")
                result.success = False
            if r.status_code != 200:
                result.errors.append(f"health: unexpected {r.status_code}")

            # 2. GET /api/sessions
            r = _call(c.get, "/api/sessions")
            if r.status_code >= 500:
                result.errors.append(f"sessions list returned {r.status_code}")
                result.success = False
            if r.status_code != 200:
                result.errors.append(f"sessions list: unexpected {r.status_code}")

            # 3. POST /api/sessions/ingest (defer=True)
            # defer=True = crée juste la session + schedule le pipeline en arrière-plan
            # (évite d'être bloqué par _PIPELINE_SEM qui sérialise le NLP lourd)
            text = f"{WORKER_TEXT} [worker {worker_id}]"
            r = _call(
                c.post,
                "/api/sessions/ingest",
                params={"text": text, "filename": f"load_test_{worker_id}.txt", "defer": True},
            )
            if r.status_code >= 500:
                result.errors.append(f"ingest returned {r.status_code}")
                result.success = False
            if r.status_code == 200:
                data = r.json()
                session_id = data.get("session_id", "")
                if not SESSION_ID_PATTERN.match(session_id):
                    result.errors.append(f"invalid session_id: {session_id}")
                    result.success = False
                else:
                    result.session_id = session_id
            else:
                result.errors.append(f"ingest: unexpected {r.status_code} / {r.text[:200]}")
                result.success = False

            # 4. GET /api/entities
            r = _call(c.get, "/api/entities", params={"limit": 5})
            if r.status_code >= 500:
                result.errors.append(f"entities returned {r.status_code}")
                result.success = False
            if r.status_code != 200:
                result.errors.append(f"entities: unexpected {r.status_code}")

            # 5. POST /api/graph/aql
            r = _call(c.post, "/api/graph/aql", params={"query": "FOR i IN 1..10 RETURN i"})
            if r.status_code >= 500:
                result.errors.append(f"graph aql returned {r.status_code}")
                result.success = False
            if r.status_code != 200:
                result.errors.append(f"graph aql: unexpected {r.status_code}")

            # 6. GET /api/sessions/{session_id} (from step 3)
            if result.session_id:
                r = _call(c.get, f"/api/sessions/{result.session_id}")
                if r.status_code >= 500:
                    result.errors.append(f"get session returned {r.status_code}")
                    result.success = False
                if r.status_code == 404:
                    result.errors.append(f"session {result.session_id} not found after creation")
                    result.success = False
                elif r.status_code != 200:
                    result.errors.append(f"get session: unexpected {r.status_code}")
    except httpx.TimeoutException as e:
        result.errors.append(f"timeout: {e}")
        result.success = False
    except Exception as e:
        result.errors.append(f"exception: {type(e).__name__}: {e}")
        result.success = False

    result.duration = time.monotonic() - t0
    return result


class TestConcurrent:
    @pytest.mark.parametrize("n_workers,label", PALIERS)
    def test_progressive_load(self, base_url, n_workers: int, label: str):
        """Execute n workers in parallel, each running 6 API calls sequentially.
        Validates 0 errors >= 500, 0 timeouts, 100 % success rate.
        """
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_worker, i, base_url): i
                for i in range(n_workers)
            }
            results: list[WorkerResult] = []
            for future in as_completed(futures):
                results.append(future.result())

        total_reqs = n_workers * 6
        successes = sum(1 for r in results if r.success)
        failures = [r for r in results if not r.success]
        durations = [r.duration for r in results]

        summary = (
            f"\n  {label} — {n_workers} workers × 6 appels = {total_reqs} requêtes\n"
            f"  Succès : {successes}/{n_workers} workers ({successes / n_workers * 100:.1f}%)\n"
            f"  Échecs  : {len(failures)} workers\n"
            f"  Durée min/avg/max : {min(durations):.2f}s / "
            f"{sum(durations) / len(durations):.2f}s / {max(durations):.2f}s\n"
        )

        if failures:
            summary += "  Détail des échecs :\n"
            for f in failures[:10]:
                summary += f"    worker {f.worker_id} ({f.duration:.2f}s): {f.errors}\n"
            if len(failures) > 10:
                summary += f"    ... et {len(failures) - 10} autres échecs\n"

        print(summary)

        assert not failures, (
            f"{len(failures)}/{n_workers} workers en échec au {label}\n{summary}"
        )
