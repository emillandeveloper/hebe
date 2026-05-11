"""
Smoke test for memory_store + embeddings + db_sqlite integration.

Uses a FakeEmbedder (deterministic hash-based vectors, NOT semantic) so the
test runs without sentence-transformers installed. The test validates the
plumbing: schema creation, insert, similarity ranking math, filters, recency,
touch (access_count update).

Run from the directory containing test_pkg/:
    python smoke_memory.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np


def main() -> None:
    # Point HEBE_DB_PATH at a temp file BEFORE importing db_sqlite
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_hebe.db")
    os.environ["HEBE_DB_PATH"] = db_path
    print(f"[smoke] db at {db_path}")

    # Auto-localiza el paquete `app`. Funciona tanto si pones este script en
    # backend/app/cognitive/memory/smoke_memory.py como en backend/scripts/, etc.
    # Sube directorios hasta encontrar un `app/__init__.py` hermano.
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    backend_root = None
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "app", "__init__.py")):
            backend_root = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if backend_root is None:
        # Fallback al layout de test que uso yo para verificar
        backend_root = os.path.join(here, "test_pkg")
    sys.path.insert(0, backend_root)
    print(f"[smoke] backend root: {backend_root}")

    from app.cognitive.memory import embeddings as emb_mod
    from app.services.db_sqlite import init_db
    from app.cognitive.memory.memory_store import (
        init_memory_chunks_schema,
        add_chunk,
        search_chunks,
        get_recent_chunks,
        count_chunks,
        deactivate_chunk,
    )

    # FakeEmbedder: hash a la posición → vector unitario aleatorio determinista.
    # No semántico, pero suficiente para testear plumbing y ordenamiento.
    class FakeEmbedder:
        model_name = "fake-hash-v1"
        dim = 64

        def embed(self, text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            return v

        def embed_batch(self, texts):
            return np.stack([self.embed(t) for t in texts])

    emb_mod.set_default_embedder(FakeEmbedder())

    # 1) Schema
    init_db()
    init_memory_chunks_schema()
    assert count_chunks() == 0, "tabla nueva debería estar vacía"
    print("[smoke] schema OK, 0 chunks")

    # 2) Insert chunks de tipos variados
    id1 = add_chunk(
        "Pedro juega Hollow Knight habitualmente y le caen mal los souls",
        kind="viewer_fact",
        subject="pedro",
        importance=0.7,
    )
    id2 = add_chunk(
        "Stream del 5/5: jugamos Persona 5 Royal 4 horas, vino Pedro a hablar de souls",
        kind="stream_summary",
        source_session="stream_2026_05_05",
        importance=0.6,
    )
    id3 = add_chunk(
        "Leo prefiere respuestas cortas y directas, sin '¡qué genial!'",
        kind="leo_fact",
        subject="leo",
        importance=0.9,
    )
    id4 = add_chunk(
        "Conversación JARVIS 6/5: Leo y Hebe rediseñaron la cleaner",
        kind="jarvis_summary",
        source_session="jarvis_2026_05_06",
    )
    print(f"[smoke] inserted ids: {id1}, {id2}, {id3}, {id4}")
    assert count_chunks() == 4

    # NOTA: con FakeEmbedder los scores son ruido alrededor de 0, así que
    # pasamos min_similarity=-1.0 para que el filtro de score no descarte nada.
    # En producción el default 0.0 es razonable (filtra correlación negativa).

    # 3) Search con filtro por kind — la query no es semántica con FakeEmbedder,
    #    pero el filtro SQL sí debe limitar el universo de candidatos.
    results = search_chunks("hollow knight", kind="viewer_fact", top_k=3, min_similarity=-1.0)
    print(f"[smoke] viewer_fact results: {len(results)} (esperado: 1)")
    assert len(results) == 1
    assert results[0]["id"] == id1
    assert results[0]["subject"] == "pedro"
    assert "score" in results[0]

    # 4) Search por subject
    results = search_chunks("preferencias", subject="leo", top_k=5, min_similarity=-1.0)
    print(f"[smoke] subject=leo results: {len(results)} (esperado: 1)")
    assert len(results) == 1
    assert results[0]["id"] == id3

    # 5) Search sin filtro: deben volver hasta top_k de los 4
    results = search_chunks("cualquier cosa", top_k=10, min_similarity=-1.0)
    print(f"[smoke] sin filtro: {len(results)} (esperado: 4)")
    assert len(results) == 4

    # 6) Recency
    recent = get_recent_chunks(kind="stream_summary", limit=5)
    assert len(recent) == 1
    assert recent[0]["id"] == id2
    print("[smoke] recency OK")

    # 7) Touch: chunk1 ya fue tocado en step 3 (1) y step 5 (2). Dos búsquedas
    # más → access_count debería ser 4.
    _ = search_chunks("hollow knight", kind="viewer_fact", top_k=1, min_similarity=-1.0)
    after = search_chunks("hollow knight", kind="viewer_fact", top_k=1, min_similarity=-1.0)
    print(f"[smoke] access_count tras 4 hits acumulados: {after[0]['access_count']} (esperado: 4)")
    assert after[0]["access_count"] == 4
    assert after[0]["last_accessed_at"] is not None

    # 8) Soft-delete
    deactivate_chunk(id1)
    after = search_chunks("hollow knight", kind="viewer_fact", top_k=5, min_similarity=-1.0)
    assert len(after) == 0
    assert count_chunks(active_only=True) == 3
    assert count_chunks(active_only=False) == 4
    print("[smoke] soft-delete OK")

    # 9) Filtro por embedding_model: si el embedder cambia de model_name,
    #    los chunks viejos no deben aparecer en búsquedas (necesitarían re-embed).
    class OtherModelEmbedder(FakeEmbedder):
        model_name = "fake-hash-v2"

    emb_mod.set_default_embedder(OtherModelEmbedder())
    results = search_chunks("cualquier cosa", top_k=10, min_similarity=-1.0)
    print(f"[smoke] tras cambio de modelo: {len(results)} (esperado: 0, hay que re-embed)")
    assert len(results) == 0

    print("\n[smoke] TODO OK ✓")


if __name__ == "__main__":
    main()
