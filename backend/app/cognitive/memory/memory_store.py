"""
RAG memory store for Hebe.

Dos capas con patrones de acceso distintos:

  memory_facts        (existente, en db_sqlite.py)
      Hechos estructurados: viewer_fact, leo_fact, citas, preferencias.
      Lookup por `subject` y `kind`. Sin embeddings.
      Funciones existentes: add_memory_fact, search_memory_facts, etc.

  memory_chunks       (nueva, definida aquí)
      Memorias en texto libre con embeddings: stream summaries, jarvis
      summaries, momentos. Lookup por similitud (con filtro opcional por
      kind/subject).
      Funciones: add_chunk, search_chunks, get_recent_chunks.

Las dos capas pueden coexistir y consultarse en el mismo paso de retrieval
desde response_synthesizer.

El nombre del modelo de embedding se guarda por chunk. Si algún día se cambia
de modelo, los chunks viejos siguen siendo buscables solo si reembedeas
(scripts/reembed_memory.py — pendiente).
"""

from __future__ import annotations

from typing import Optional, Any
import sqlite3

import numpy as np

from app.services.db_sqlite import (
    get_db_connection,
    utc_now_iso,
    dumps_json,
    loads_json,
)
from app.cognitive.memory.embeddings import (
    Embedder,
    get_default_embedder,
    vector_to_blob,
    blob_to_vector,
)


# =============================================================================
# Schema
# =============================================================================

def init_memory_chunks_schema(conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Crea la tabla memory_chunks + índices si no existen.
    Idempotente. Llamar desde db_sqlite.init_db().
    """
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,                 -- stream_summary | jarvis_summary | viewer_moment | leo_moment | misc
            subject TEXT,                       -- viewer_id, "leo", stream_id, NULL
            source_session TEXT,                -- identificador de stream o conversación
            embedding BLOB NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            created_at TEXT NOT NULL,
            tags TEXT,                          -- JSON opcional
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kind_active ON memory_chunks(kind, active)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_subject ON memory_chunks(subject)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON memory_chunks(source_session)")

    if own_conn:
        conn.commit()
        conn.close()


# =============================================================================
# Write API
# =============================================================================

def add_chunk(
    text: str,
    kind: str,
    *,
    subject: Optional[str] = None,
    source_session: Optional[str] = None,
    importance: float = 0.5,
    tags: Optional[dict] = None,
    embedder: Optional[Embedder] = None,
) -> int:
    """Embedde `text` y persiste el chunk. Devuelve el id nuevo."""
    if not text or not text.strip():
        raise ValueError("add_chunk: text vacío")

    embedder = embedder or get_default_embedder()
    vec = embedder.embed(text)
    blob = vector_to_blob(vec)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memory_chunks
            (text, kind, subject, source_session, embedding, embedding_model,
             embedding_dim, importance, access_count, last_accessed_at,
             created_at, tags, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, 1)
        """,
        (
            text,
            kind,
            subject,
            source_session,
            blob,
            embedder.model_name,
            embedder.dim,
            float(importance),
            utc_now_iso(),
            dumps_json(tags),
        ),
    )
    chunk_id = cur.lastrowid
    conn.commit()
    conn.close()
    return chunk_id


def add_chunk_if_new(
    text: str,
    kind: str,
    *,
    subject: Optional[str] = None,
    source_session: Optional[str] = None,
    importance: float = 0.5,
    tags: Optional[dict] = None,
    min_similarity: float = 0.92,
    embedder: Optional[Embedder] = None,
) -> tuple[int | None, bool]:
    """
    Persist a chunk unless an active near-duplicate already exists.

    Returns (chunk_id, created). If a duplicate is found, chunk_id is the
    existing row id and created is False.
    """
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return None, False

    try:
        candidates = search_chunks(
            text,
            kind=kind,
            subject=subject,
            top_k=3,
            min_similarity=min_similarity,
            active_only=True,
            touch=False,
            embedder=embedder,
        )
        for item in candidates:
            existing_text = " ".join(str(item.get("text") or "").strip().lower().split())
            if existing_text == normalized or float(item.get("score") or 0.0) >= min_similarity:
                return int(item["id"]), False
    except Exception as exc:
        print(f"[HEBE][MEMORY] chunk dedup failed, inserting anyway: {exc!r}", flush=True)

    return (
        add_chunk(
            text=text,
            kind=kind,
            subject=subject,
            source_session=source_session,
            importance=importance,
            tags=tags,
            embedder=embedder,
        ),
        True,
    )


# =============================================================================
# Read API
# =============================================================================

def search_chunks(
    query: str,
    *,
    kind: Optional[str] = None,
    subject: Optional[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.0,
    active_only: bool = True,
    touch: bool = True,
    embedder: Optional[Embedder] = None,
) -> list[dict]:
    """
    Devuelve los top_k chunks más similares a `query`, opcionalmente filtrados
    por kind/subject.

    Cada resultado lleva los campos de memory_chunks + 'score' (cosine similarity
    en [-1, 1], pero en la práctica >=0 con embeddings normalizados).

    Si touch=True, incrementa access_count y actualiza last_accessed_at de los
    chunks que efectivamente se devuelven (no de todos los candidatos).

    NOTA: solo busca dentro de chunks embedded con el mismo modelo que el
    embedder activo. Chunks con un modelo distinto se ignoran (necesitan re-embed).
    """
    if not query or not query.strip():
        return []

    embedder = embedder or get_default_embedder()
    qvec = embedder.embed(query)

    conn = get_db_connection()
    cur = conn.cursor()

    where = ["embedding_model = ?"]
    params: list[Any] = [embedder.model_name]
    if active_only:
        where.append("active = 1")
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if subject:
        where.append("subject = ?")
        params.append(subject)

    sql = (
        "SELECT id, text, kind, subject, source_session, embedding, embedding_dim, "
        "       importance, access_count, last_accessed_at, created_at, tags "
        "FROM memory_chunks WHERE " + " AND ".join(where)
    )
    cur.execute(sql, params)
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return []

    # Embeddings normalizados → cosine = dot product. Para miles de filas
    # esto es despreciable. Si crece a >100k, cambiar a sqlite-vec.
    scored: list[tuple[float, dict]] = []
    for row in rows:
        vec = blob_to_vector(row["embedding"], row["embedding_dim"])
        score = float(np.dot(qvec, vec))
        if score < min_similarity:
            continue
        item = {
            "id": row["id"],
            "text": row["text"],
            "kind": row["kind"],
            "subject": row["subject"],
            "source_session": row["source_session"],
            "importance": row["importance"],
            "access_count": row["access_count"],
            "last_accessed_at": row["last_accessed_at"],
            "created_at": row["created_at"],
            "tags": loads_json(row["tags"]),
            "score": score,
        }
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for _, item in scored[:top_k]]

    if touch and top:
        now = utc_now_iso()
        ids = [item["id"] for item in top]
        placeholders = ",".join("?" for _ in ids)
        cur.execute(
            f"UPDATE memory_chunks "
            f"SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE id IN ({placeholders})",
            [now, *ids],
        )
        conn.commit()
        for item in top:
            item["access_count"] += 1
            item["last_accessed_at"] = now

    conn.close()
    return top


def get_recent_chunks(
    *,
    kind: Optional[str] = None,
    subject: Optional[str] = None,
    limit: int = 10,
    active_only: bool = True,
) -> list[dict]:
    """
    Recall por recencia, sin embeddings. Útil para "los N últimos
    stream_summary" o "los últimos resúmenes JARVIS de la semana".
    """
    conn = get_db_connection()
    cur = conn.cursor()

    where = []
    params: list[Any] = []
    if active_only:
        where.append("active = 1")
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if subject:
        where.append("subject = ?")
        params.append(subject)

    sql = (
        "SELECT id, text, kind, subject, source_session, importance, "
        "       access_count, last_accessed_at, created_at, tags "
        "FROM memory_chunks"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "text": row["text"],
                "kind": row["kind"],
                "subject": row["subject"],
                "source_session": row["source_session"],
                "importance": row["importance"],
                "access_count": row["access_count"],
                "last_accessed_at": row["last_accessed_at"],
                "created_at": row["created_at"],
                "tags": loads_json(row["tags"]),
            }
        )
    return out


# =============================================================================
# Lifecycle
# =============================================================================

def deactivate_chunk(chunk_id: int) -> None:
    """Soft-delete (active=0). El chunk deja de aparecer en búsquedas."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE memory_chunks SET active = 0 WHERE id = ?", (chunk_id,))
    conn.commit()
    conn.close()


def update_chunk_importance(chunk_id: int, importance: float) -> None:
    """Ajusta importance manualmente. La consolidación periódica lo hará sola
    en Fase 3, pero mientras tanto puede ser útil para curar a mano."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE memory_chunks SET importance = ? WHERE id = ?",
        (float(importance), chunk_id),
    )
    conn.commit()
    conn.close()


def count_chunks(kind: Optional[str] = None, active_only: bool = True) -> int:
    """Útil para health checks / debug."""
    conn = get_db_connection()
    cur = conn.cursor()
    where = []
    params: list[Any] = []
    if active_only:
        where.append("active = 1")
    if kind:
        where.append("kind = ?")
        params.append(kind)
    sql = "SELECT COUNT(*) AS c FROM memory_chunks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur.execute(sql, params)
    n = cur.fetchone()["c"]
    conn.close()
    return int(n)
