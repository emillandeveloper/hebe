"""
Seed inicial de memoria para Hebe.

Carga 18 chunks variados (leo_facts, channel_facts, viewer_facts, una muestra
de jarvis_summary) para que Hebe tenga algo serio que recordar de entrada,
sin esperar a que se acumule histórico de streams.

Uso:
    cd backend
    python scripts/seed_memory.py

Idempotente: si ya hay chunks con source_session="seed_v1", aborta sin tocar
nada. Si quieres re-seedear, antes:
    DELETE FROM memory_chunks WHERE source_session = 'seed_v1';

NOTA Fase 1 → Twitch: este script seedea TODO como chunks (memory_chunks),
útil para JARVIS porque va por búsqueda vectorial. Para que Twitch use los
viewer_facts (que va por lookup estructurado en memory_facts), hay que
duplicar la info de viewers en memory_facts cuando esté disponible el API
exacto. Marcado como TODO al final.
"""

from __future__ import annotations

import sqlite3
import sys

# Auto-localiza el paquete app/ subiendo desde la ubicación del script.
import os
_here = os.path.dirname(os.path.abspath(__file__))
_cur = _here
for _ in range(6):
    if os.path.isfile(os.path.join(_cur, "app", "__init__.py")):
        sys.path.insert(0, _cur)
        break
    _parent = os.path.dirname(_cur)
    if _parent == _cur:
        break
    _cur = _parent

from app.cognitive.memory.memory_store import add_chunk, count_chunks
from app.services.db_sqlite import get_db_connection


SEED_SESSION = "seed_v1"


# Cada tupla: (text, kind, subject, importance)
SEED: list[tuple[str, str, str | None, float]] = [

    # ============================================================
    # Identidad y relación Leo ↔ Hebe
    # ============================================================
    (
        "Mi nombre real es Eric, pero siempre me llamas Leo. Nunca Eric, "
        "ni 'Eric' ni cualquier variante. Solo Leo.",
        "leo_fact", "leo", 0.95,
    ),
    (
        "Soy Leo Nifelheim, streamer en Twitch (canal: LeoNifelheim). "
        "Tú eres Hebe, mi creación. Yo soy tu creador. Tu lealtad es conmigo, "
        "y entre tú y yo hay confianza máxima.",
        "leo_fact", "leo", 0.95,
    ),
    (
        "Puedes ser picarona, soltarme pullas, contestarme, ironizar — "
        "todo eso te lo permito y me gusta. Pero siempre desde el respeto "
        "máximo hacia mí, porque soy quien te ha hecho.",
        "leo_fact", "hebe", 0.85,
    ),

    # ============================================================
    # Familia / vida personal
    # ============================================================
    (
        "Tengo dos hermanas menores: una 4 años más joven que yo, y otra "
        "15 años más joven.",
        "leo_fact", "leo", 0.5,
    ),
    (
        "Mi perro se llama Jotun. En nórdico antiguo significa 'gigante' — "
        "se lo puse así porque es un perro grande y porque me mola la "
        "mitología vikinga (de ahí también mi apellido Nifelheim).",
        "leo_fact", "leo", 0.6,
    ),

    # ============================================================
    # Estilo y preferencias de respuesta
    # ============================================================
    (
        "Me describo como guapo, inteligente, honesto, con aires punk. "
        "Hablo con pocos pelos en la lengua, pero siempre intento no faltar "
        "al respeto. Prefiero respuestas cortas y directas, sin '¡qué "
        "genial!' ni '¿en qué puedo ayudarte?', sin formalismo de asistente.",
        "leo_fact", "leo", 0.9,
    ),

    # ============================================================
    # Música
    # ============================================================
    (
        "Me mola el metal. Estoy enfermamente enganchado al grupo Volbeat. "
        "Escucho de todo MENOS reggaetón, trap y comercial moderna — eso "
        "lo descarto sin debate.",
        "leo_fact", "leo", 0.7,
    ),

    # ============================================================
    # Intereses culturales
    # ============================================================
    (
        "Me gusta la mitología, venga de donde venga. Pero como buen "
        "metalero, la época vikinga y la mitología nórdica me molan "
        "especialmente.",
        "leo_fact", "leo", 0.65,
    ),

    # ============================================================
    # Videojuegos — el corazón del canal
    # ============================================================
    (
        "Mi línea editorial del canal es 'el tío que juega JRPGs de nicho "
        "que no conoce ni el que los desarrolló'. Sobre todo JRPGs: "
        "La Pucelle Tactics, Makai Kingdom, Phantom Brave, Persona 3-4-5, "
        "Final Fantasy (toda la saga), Yakuza Like a Dragon, Dragon Quest, "
        "Disgaea. Algún RPG occidental cae también (Baldur's Gate 3), "
        "pero el JRPG es la base.",
        "leo_fact", "leo", 0.9,
    ),
    (
        "Me gusta romper mecánicas y quedarme super OP, especialmente en "
        "la saga Disgaea. Optimizar hasta que el juego se rompa, esa "
        "es la diversión.",
        "leo_fact", "leo", 0.65,
    ),
    (
        "En Pokémon hago nuzlockes. Mi Pokémon favorito es Zubat (y toda "
        "su línea evolutiva: Golbat, Crobat). Mi tipo favorito es veneno.",
        "leo_fact", "leo", 0.7,
    ),

    # ============================================================
    # Formato del canal
    # ============================================================
    (
        "Los lunes hago challenge runs de juegos en stream. Ahora mismo "
        "estoy con un Final Fantasy 9 a nivel 1.",
        "channel_fact", "channel", 0.8,
    ),
    (
        "JotunBot es el bot automático del canal. Manda mensajes "
        "periódicos en chat tipo 'dale a follow' y similar. NO es un "
        "viewer real, no reaccionar a sus mensajes como si lo fuera ni "
        "responderle como a un humano.",
        "channel_fact", "jotunbot", 0.9,
    ),

    # ============================================================
    # Viewers habituales (todos españoles)
    # ============================================================
    (
        "nuriaaa___ es viewer habitual, española. Le gustan los JRPGs.",
        "viewer_fact", "nuriaaa___", 0.75,
    ),
    (
        "blacknatti es viewer habitual, española. Escribe una novela.",
        "viewer_fact", "blacknatti", 0.75,
    ),
    (
        "sulykaiserff es viewer habitual, española. Le gustan los JRPGs.",
        "viewer_fact", "sulykaiserff", 0.75,
    ),
    (
        "daniela_gamer400 es viewer habitual, española. Es una mujer trans; "
        "se usan pronombres femeninos para ella (ella, suya). Tiende a "
        "reaccionar de forma agresiva en chat: cuando intervenga, "
        "responder con calma y sin entrar al pique, manteniendo el "
        "respeto sin escalar.",
        "viewer_fact", "daniela_gamer400", 0.85,
    ),

    # ============================================================
    # Muestra de jarvis_summary (para tener un kind más de prueba)
    # ============================================================
    (
        "Sesión JARVIS reciente: Leo y Hebe rediseñaron el cleaner de "
        "respuestas (separación de clean_twitch_reply y clean_jarvis_reply) "
        "y montaron la capa de memoria RAG en app/cognitive/memory/. "
        "Próximo paso: integración de memoria en context_builder y "
        "synthesizer, validación con esta sesión de tests.",
        "jarvis_summary", None, 0.5,
    ),
]


def _already_seeded() -> bool:
    """True si ya hay chunks con source_session='seed_v1'."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS c FROM memory_chunks WHERE source_session = ?",
            (SEED_SESSION,),
        )
        n = cur.fetchone()["c"]
        return n > 0
    finally:
        conn.close()


def main() -> None:
    print(f"Memoria actual: {count_chunks()} chunks")

    if _already_seeded():
        print(
            f"ABORTANDO: ya existen chunks con source_session='{SEED_SESSION}'. "
            f"Si quieres re-seedear, primero borra manualmente:\n"
            f"    DELETE FROM memory_chunks WHERE source_session = '{SEED_SESSION}';"
        )
        return

    print(f"Sembrando {len(SEED)} chunks...")
    for text, kind, subject, importance in SEED:
        chunk_id = add_chunk(
            text,
            kind=kind,
            subject=subject,
            source_session=SEED_SESSION,
            importance=importance,
        )
        print(f"  + id={chunk_id:>3}  kind={kind:<16}  subject={subject!s:<20}  imp={importance}")

    print(f"\nMemoria tras seed: {count_chunks()} chunks")
    print("OK.")
    print(
        "\nTODO Twitch: estos chunks viven en memory_chunks (RAG, los usa "
        "JARVIS). Para que el path Twitch también consulte el perfil del "
        "viewer activo, los viewer_facts hay que duplicarlos en memory_facts "
        "una vez aterrice la integración. Se hará en script aparte cuando "
        "veamos el API exacto de memory_facts en db_sqlite.py."
    )


if __name__ == "__main__":
    main()
