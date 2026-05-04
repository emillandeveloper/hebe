"""
Limpieza de respuestas generadas para el chat de Twitch.

Renombrado desde replay_cleaner.py (typo histórico de la deuda 7.3 del bible).

Tres responsabilidades:
1. clean_stream_reply: limpieza estructural (turnos extra, prefijos,
   wrappers ridículos, copia literal del input, longitud). Lo que ya hacía.
2. detect_helper_pattern: detección de "fugas IA" del RLHF de qwen 2.5
   (¿en qué puedo ayudarte?, lo siento pero no puedo, etc).
3. STREAM_MAX_CHARS y demás constantes de presentación.
4. Defensa contra roleplay roto en modelos menos censurados.

NO conoce el modelo, no conoce el formato del prompt, no llama a Ollama.
Recibe el texto crudo del modelo y devuelve el texto que va al chat (o ""),
y por separado expone una API para que el caller decida si hacer retry.
"""

from __future__ import annotations

import re
from typing import Optional


# =============================================================================
# Limpieza estructural (lo que ya hacía replay_cleaner)
# =============================================================================

STREAM_MAX_CHARS = 240


# Modelos más libres/roleplay tienden a escupir etiquetas de diálogo:
# "[leo]:", "[katya]:", "nomad:", etc. Hebe nunca debe publicar eso.
SPEAKER_PREFIX_RE = re.compile(
    r"^\s*\[?[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_\- ]{1,32}\]?\s*:\s*"
)

INLINE_SPEAKER_RE = re.compile(
    r"\s+\[?[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_\- ]{1,32}\]?\s*:"
)

BAD_ROLEPLAY_CLEAN_PATTERNS = [
    # "soy tu amiga chatter leo" / "chatter Leo" no es voz de Hebe:
    # es el modelo confundiendo la etiqueta del prompt con una persona.
    r"\bchatter\s+leo\b",

    # Fuga meta típica cuando el modelo cree que está escribiendo un ejemplo.
    r"\b(me has hecho la tarea|crear un ejemplo de conversaci[oó]n|di[aá]logo real)\b",

    # Traducciones o aclaraciones entre paréntesis: "maldita sea! (mala suerte)".
    r"\([a-záéíóúüñ ]{3,40}\)\s*$",

    # Modo coach/terapéutico, impropio de Hebe en Twitch.
    r"\b(no te preocupes por los problemas|solo respira|piensa en lo que quieres de verdad)\b",

    # Tono irrespetuoso con Leo que ya salió en pruebas.
    r"\b(c[aá]llate,?\s+chaval|chaval)\b",
]


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

    # Si sigue empezando por una etiqueta genérica tipo "[leo]:",
    # "[katya]:" o "nomad:", es roleplay roto. Mejor silencio que
    # publicar una línea fuera de personaje.
    if SPEAKER_PREFIX_RE.match(cleaned):
        return ""

    # Descartar wrappers horribles del modelo.
    lowered = cleaned.lower()
    for pattern in BAD_WRAPPER_PATTERNS:
        if re.search(pattern, lowered):
            return ""

    # Descartar fugas de roleplay/meta que no se arreglan limpiando.
    for pattern in BAD_ROLEPLAY_CLEAN_PATTERNS:
        if re.search(pattern, lowered):
            return ""

    # Evitar pseudo-diálogo en una sola línea.
    # OJO: no cortar vocativos normales como "vale, Leo: ..."
    for marker in INLINE_TURN_MARKERS:
        if marker in cleaned:
            idx = cleaned.find(marker)
            before = cleaned[:idx].strip()

            # Solo cortamos si el marcador parece realmente otro turno,
            # no una frase normal con un nombre.
            if marker.startswith("[") or idx == 0:
                if before:
                    cleaned = before
                else:
                    return ""

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Cortar si intenta meter otro hablante en mitad de la frase.
    # Ejemplo: "vale, Leo. [katya]: no..." -> "vale, Leo."
    match = INLINE_SPEAKER_RE.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()].strip()

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


# =============================================================================
# Detección de patrones helper (NUEVO 28/04)
# =============================================================================
#
# Patrones de "fuga IA" verificados como reincidentes en el log del 27/04.
# Cada entrada es (nombre_patrón, regex_compilado). El nombre se usa en logs
# y métricas para saber QUÉ está enganchando.
#
# Criterios de inclusión (los tres deben cumplirse):
#   (a) Hebe NUNCA lo diría sin importar el contexto.
#   (b) Aparece 3+ veces en logs reales del canal.
#   (c) No se solapa con uso legítimo de la voz de Hebe en el corpus.
#
# Si una respuesta engancha cualquiera de estos patrones, el caller
# (response_synthesizer) decide qué hacer: típicamente retry con seed
# distinto, máximo 1 reintento, y publicar igualmente si retry vuelve a
# fallar. Mejor algo imperfecto que silencio en chat.
#
# Esta lista es CERRADA. Si necesitas añadir, primero verifica los tres
# criterios. Si la lista crece más de 12-15 entradas, es señal de que ya
# toca pasar al clasificador LLM (pieza 5 del bible) y jubilar este filtro.

_HELPER_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    # "¿en qué puedo ayudarte/asistirte?" - 9+ apariciones literales en log.
    ("ask_help",
     re.compile(r"\b(en qu[eé]) puedo (ayudarte|asistirte)", re.I)),

    # "hay algo / algún tema [más / específico / en específico / en concreto / ...]"
    # 12+ apariciones, con muchas variantes de "específico" y "concreto".
    ("anything_else",
     re.compile(
         r"hay (algo|alg[uú]n tema) "
         r"(m[aá]s|espec[ií]fico|en espec[ií]fico|en concreto|"
         r"en lo que|sobre el que|en el que|del que)",
         re.I,
     )),

    # "lo siento, pero no puedo / no voy a [replicar/responder/hacer]"
    # El disclaimer ChatGPT puro. 5+ apariciones ante insultos legítimos.
    ("sorry_cant",
     re.compile(
         r"lo siento[,.]?\s*pero no (puedo|voy a) (replicar|responder|hacer)",
         re.I,
     )),

    # Auto-identificación como IA o asistente.
    ("as_ai",
     re.compile(r"\b(como (una )?ia|soy una asistente|estoy aqu[ií] para)\b", re.I)),

    # "no dudes en [preguntar/compartir/...]"
    ("dont_hesitate",
     re.compile(
         r"no dudes en (preguntar|compartir|consultar|decirme|hac[eé]rmelo)",
         re.I,
     )),

    # "qué interesante" / "qué buena pregunta" / "buena pregunta para [reflexionar/pensar]"
    # Excluye sarcasmo seco tipo "buena pregunta, idiota": exige "qué" delante
    # o "para reflexionar/pensar/considerar/debatir" detrás.
    ("interesting_q",
     re.compile(
         r"qu[eé] (interesante|buena pregunta)"
         r"|buena pregunta para (reflexionar|pensar|considerar|debatir)",
         re.I,
     )),

    # "espero que te sirva / haberte ayudado"
    ("hope_helps",
     re.compile(r"espero (que (te (sirva|ayude))|haberte ayudado)", re.I)),

    # "mantengamos / mantén / mantente la calma" - tic de moderación robotizada.
    ("keep_calm",
     re.compile(r"(mantengamos|mant[eé]n|mantente|mantendr[eé]) la calma", re.I)),

    # Modelos menos censurados pueden caer en "roleplay roto":
    # etiquetas de hablante, otros personajes, o metacomentarios de ejemplo.
    ("speaker_label",
     re.compile(r"^\s*\[[^\]]{1,40}\]\s*:", re.I)),

    ("chatter_leo_literal",
     re.compile(r"\bchatter\s+leo\b", re.I)),

    ("invented_dialogue",
     re.compile(r"\[[^\]]{1,40}\]\s*:", re.I)),

    ("translation_parenthesis",
     re.compile(r"\([a-záéíóúüñ ]{3,40}\)\s*$", re.I)),

    ("fake_assistant_task",
     re.compile(
         r"(me has hecho la tarea|crear un ejemplo de conversaci[oó]n|di[aá]logo real)",
         re.I,
     )),

    ("therapy_mode",
     re.compile(
         r"(no te preocupes por los problemas|solo respira|piensa en lo que quieres de verdad)",
         re.I,
     )),

    ("leo_disrespect_chaval",
     re.compile(r"\b(c[aá]llate,?\s+chaval|chaval)\b", re.I)),
]


def detect_helper_pattern(text: str) -> Optional[str]:
    """
    Devuelve el nombre del primer patrón helper que engancha; None si
    la respuesta está limpia.

    NO transforma la respuesta. La decisión de retry o publicar la toma
    el caller (response_synthesizer).
    """
    if not text:
        return None
    for name, pattern in _HELPER_PATTERNS:
        if pattern.search(text):
            return name
    return None