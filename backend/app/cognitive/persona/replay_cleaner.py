from __future__ import annotations

import re


STREAM_MAX_CHARS = 240


BAD_WRAPPER_PATTERNS = [
    r"^aquí (estoy )?respondiendo a\b",
    r"^estoy respondiendo a\b",
    r"^respuesta para\b",
    r"^mensaje para\b",
    r"^como hebe\b",
    r"^hebe respondería\b",
]


TURN_MARKERS = [
    "\n[chatter]:",
    "\n[chatter]",
    "\n[tú]:",
    "\n[tu]:",
    "\nLeo:",
    "\nleo:",
    "\nLeoNifelheim:",
    "\nleonifelheim:",
    "\nHebe:",
    "\nHEBE:",
    "\nUsuario:",
    "\nViewer:",
    "\nChat:",
]


PREFIXES = [
    "Hebe:",
    "HEBE:",
    "[tú]:",
    "[tu]:",
    "tú:",
    "tu:",
    "Respuesta:",
    "Respuesta de Hebe:",
    "Mensaje:",
    "Mensaje final:",
]


INLINE_TURN_MARKERS = [
    "[chatter]:",
    "[tú]:",
    "[tu]:",
    "Hebe:",
    "HEBE:",
    "Leo:",
    "LeoNifelheim:",
    "Usuario:",
    "Viewer:",
    "Chat:",
]


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\"'“”")
    return text


def clean_stream_reply(
    text: str | None,
    *,
    source_message: str | None = None,
) -> str:
    """
    Limpia respuestas generadas para Twitch.

    Objetivo:
    - evitar diálogos inventados
    - quitar prefijos tipo "Hebe:"
    - cortar turnos extra
    - descartar basura tipo "estoy respondiendo a..."
    - evitar copia literal del mensaje del chatter
    - garantizar una sola línea
    """
    if not text:
        return ""

    cleaned = str(text).strip().replace("\r", "")

    if not cleaned:
        return ""

    cleaned = cleaned.strip(" \t\n\"'“”")

    # Si intenta generar otro turno, cortamos antes.
    for marker in TURN_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    # Nos quedamos con la primera línea útil.
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return ""

    cleaned = lines[0].strip()

    # Quitar prefijos repetidos si aparecen.
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                changed = True

    cleaned = cleaned.strip(" \t\n\"'“”")

    if not cleaned:
        return ""

    # Descartar wrappers horribles del modelo.
    lowered = cleaned.lower()
    for pattern in BAD_WRAPPER_PATTERNS:
        if re.search(pattern, lowered):
            return ""

    # Evitar pseudo-diálogo en una sola línea.
    for marker in INLINE_TURN_MARKERS:
        if marker in cleaned:
            before = cleaned.split(marker, 1)[0].strip()
            if before:
                cleaned = before
            else:
                return ""

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    # Si ha copiado literalmente al chatter, mejor silencio.
    if source_message:
        if _normalize_for_compare(cleaned) == _normalize_for_compare(source_message):
            return ""

    if len(cleaned) > STREAM_MAX_CHARS:
        shortened = cleaned[:STREAM_MAX_CHARS].rsplit(" ", 1)[0].strip()
        cleaned = shortened or cleaned[:STREAM_MAX_CHARS].strip()

    return cleaned