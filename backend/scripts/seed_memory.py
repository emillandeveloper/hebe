"""
Seed inicial de memoria para Hebe. Lanzar manualmente desde backend/:

    python scripts/seed_memory.py

Crea unos cuantos chunks de partida para que Hebe tenga algo que recordar
de entrada, sin esperar a que se acumule histórico de streams.

NO es idempotente: si lo lanzas dos veces, obtendrás duplicados.
Es responsabilidad de Leo no lanzarlo más de una vez. Si necesitas resetear,
usa sqlite3 hebe.db "DELETE FROM memory_chunks;" antes de relanzar.
"""

from __future__ import annotations

import os
import sys

# Añadir backend/ al sys.path para que los imports de app.* funcionen.
_here = os.path.dirname(os.path.abspath(__file__))
_backend = os.path.dirname(_here)
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from app.services.db_sqlite import init_db
from app.cognitive.memory.memory_store import (
    add_chunk,
    count_chunks,
    init_memory_chunks_schema,
)


# =====================================================================
# Lista de chunks semilla. Edita a tu gusto antes de lanzar el script.
# Formato: (texto, kind, subject, importance)
#
# kind útiles en Fase 1:
#   "leo_fact"        — preferencias, datos de Leo (subject="leo")
#   "stream_summary"  — resumen de un stream pasado (subject=None o stream_id)
#   "jarvis_summary"  — resumen de conversación JARVIS (subject=None)
#   "viewer_fact"     — dato sobre un viewer (subject=user_login del viewer)
# =====================================================================

SEED: list[tuple[str, str, str | None, float]] = [
    # leo_facts — preferencias que Hebe debe recordar siempre
    (
        "Leo prefiere respuestas cortas y directas, sin '¡qué genial!' ni '¿en qué puedo ayudarte?'",
        "leo_fact",
        "leo",
        0.9,
    ),
    (
        "Leo está jugando su primera partida de Persona 5 Royal",
        "leo_fact",
        "leo",
        0.7,
    ),
    (
        "Leo es VTuber, su modelo se llama Hebe Nifelheim",
        "leo_fact",
        "leo",
        0.8,
    ),
    # jarvis_summary — contexto de conversaciones recientes
    (
        "Sesión JARVIS reciente: Leo y Hebe rediseñaron el cleaner de respuestas "
        "y montaron la capa de memoria RAG (Fase 1)",
        "jarvis_summary",
        None,
        0.6,
    ),
]


def main() -> None:
    print("[seed] Inicializando DB…")
    init_db()
    init_memory_chunks_schema()

    before = count_chunks()
    print(f"[seed] Memoria antes: {before} chunks")

    for text, kind, subject, importance in SEED:
        chunk_id = add_chunk(
            text,
            kind=kind,
            subject=subject,
            importance=importance,
        )
        print(f"  + id={chunk_id}  kind={kind}  subject={subject!r}")

    after = count_chunks()
    print(f"[seed] Memoria después: {after} chunks (+{after - before})")
    print("[seed] Listo.")


if __name__ == "__main__":
    main()
