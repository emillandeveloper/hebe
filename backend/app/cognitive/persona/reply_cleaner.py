"""
Limpieza de respuestas generadas por el modelo.

Dos funciones públicas según contexto:
  clean_twitch_reply(text, source_message=None)
      Para respuestas que van al chat de Twitch.
      Trunca a 240 chars respetando palabra, protege contra copia literal.

  clean_jarvis_reply(text)
      Para respuestas que van a JARVIS (UI/voz con Leo).
      Solo limpia prefijos y normaliza espacio. Sin límite de longitud.

Ambas comparten _strip_basic() para quitar prefijos de nombre, comillas
y espacios sobrantes.

detect_helper_pattern(text) → str | None  (intacta, solo para Twitch/retry)

Las defensas contra modelos 3B (TURN_MARKERS, INLINE_TURN_MARKERS,
BAD_WRAPPER_PATTERNS, split de primera línea) se han eliminado porque
GPT-5-mini no genera esa basura y estaban mutilando respuestas legítimas.
Ejemplo real mutilado: "perfecto, Leo. buen trabajo: menos humo, más eficacia."
→ publicado solo "perfecto, Leo." por el split en "Leo:".
"""

from __future__ import annotations

import re
from typing import Optional


# =============================================================================
# Constantes
# =============================================================================

STREAM_MAX_CHARS = 240

_PREFIXES = [
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


# =============================================================================
# Núcleo compartido
# =============================================================================

def _strip_basic(text: str) -> str:
    """
    Limpieza mínima compartida por clean_twitch_reply y clean_jarvis_reply.

    Hace:
    - strip + elimina \\r
    - quita comillas decorativas de inicio/fin
    - quita prefijos de hablante ("Hebe:", "[tú]:", etc.)
    - colapsa espacios múltiples a uno

    NO hace:
    - split por líneas
    - truncado de longitud
    - detección de patrones
    """
    if not text:
        return ""

    cleaned = str(text).strip().replace("\r", "")
    if not cleaned:
        return ""

    cleaned = cleaned.strip(" \t\n\"‘’“”")

    # Quitar prefijos de hablante repetidos (puede haber varios apilados)
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                changed = True

    cleaned = cleaned.strip(" \t\n\"‘’“”")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\"‘’“”")
    return text


# =============================================================================
# API pública
# =============================================================================

def clean_twitch_reply(
    text: str | None,
    *,
    source_message: str | None = None,
) -> str:
    """
    Limpieza para respuestas que van al chat de Twitch.

    - Aplica _strip_basic (prefijos, comillas, espacios).
    - Une líneas en un solo espacio (GPT-5-mini puede dar respuesta
      en varias frases separadas por newline; unirlas es correcto).
    - Si la respuesta es copia literal del source_message, devuelve "".
    - Trunca a STREAM_MAX_CHARS=240 respetando palabra completa.
    """
    cleaned = _strip_basic(text)
    if not cleaned:
        return ""

    # Unir líneas → una sola línea para el chat de Twitch
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    # Anti-copia literal
    if source_message:
        if _normalize_for_compare(cleaned) == _normalize_for_compare(source_message):
            return ""

    # Truncado respetando palabra
    if len(cleaned) > STREAM_MAX_CHARS:
        shortened = cleaned[:STREAM_MAX_CHARS].rsplit(" ", 1)[0].strip()
        cleaned = shortened or cleaned[:STREAM_MAX_CHARS].strip()

    return cleaned


def clean_jarvis_reply(text: str | None) -> str:
    """
    Limpieza para respuestas que van a JARVIS personal (UI/voz con Leo).

    - Aplica _strip_basic (prefijos, comillas, espacios).
    - Preserva párrafos/líneas múltiples (Leo puede pedir respuestas largas).
    - Sin truncado de longitud.
    - Sin anti-copia (Leo puede repetir lo que él mismo ha dicho).
    """
    cleaned = _strip_basic(text)
    if not cleaned:
        return ""

    # Preservamos la estructura de líneas, solo normalizamos espacios intralínea
    lines = cleaned.split("\n")
    normalized = "\n".join(re.sub(r" +", " ", ln).strip() for ln in lines)
    normalized = normalized.strip()

    return normalized


# DEPRECATED: usar clean_twitch_reply o clean_jarvis_reply según contexto.
# Mantenida para retrocompatibilidad mientras se actualizan call sites.
def clean_stream_reply(
    text: str | None,
    *,
    source_message: str | None = None,
) -> str:
    return clean_twitch_reply(text, source_message=source_message)


# =============================================================================
# Detección de patrones helper
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
# Esta lista es CERRADA. Si la lista crece más de 12-15 entradas, es señal
# de que ya toca pasar al clasificador LLM y jubilar este filtro.

_HELPER_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("ask_help",
     re.compile(r"\b(en qu[eé]) puedo (ayudarte|asistirte)", re.I)),

    ("anything_else",
     re.compile(
         r"hay (algo|alg[uú]n tema) "
         r"(m[aá]s|espec[ií]fico|en espec[ií]fico|en concreto|"
         r"en lo que|sobre el que|en el que|del que)",
         re.I,
     )),

    ("sorry_cant",
     re.compile(
         r"lo siento[,.]?\s*pero no (puedo|voy a) (replicar|responder|hacer)",
         re.I,
     )),

    ("as_ai",
     re.compile(r"\b(como (una )?ia|soy una asistente|estoy aqu[ií] para)\b", re.I)),

    ("dont_hesitate",
     re.compile(
         r"no dudes en (preguntar|compartir|consultar|decirme|hac[eé]rmelo)",
         re.I,
     )),

    ("interesting_q",
     re.compile(
         r"qu[eé] (interesante|buena pregunta)"
         r"|buena pregunta para (reflexionar|pensar|considerar|debatir)",
         re.I,
     )),

    ("hope_helps",
     re.compile(r"espero (que (te (sirva|ayude))|haberte ayudado)", re.I)),

    ("keep_calm",
     re.compile(r"(mantengamos|mant[eé]n|mantente|mantendr[eé]) la calma", re.I)),

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
