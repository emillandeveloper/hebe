"""
Normalización de display_names de Twitch para meterlos en el prompt.

Twitch entrega cosas tipo 'nuriiia___', 'daniela_gamer400', '@HebeNifelheim'.
Pasarlas crudas al few-shot mete ruido en la respuesta de Hebe (el modelo
copia los guiones bajos y los dígitos en su salida) y rompe la inmersión.

Este módulo es agnóstico al cognitive_flow. Solo limpia strings.

NOTA: la detección de broadcaster ya la hace ResponseSynthesizer._is_broadcaster
de forma más rica (mira varios campos del payload). Aquí no la dupliques.
"""

from __future__ import annotations

import re

# Mapeo explícito de regulares del canal. Ampliar a mano cuando se observen
# nuevos viewers frecuentes en los logs. La clave es display_name.lower().
_KNOWN_ALIASES: dict[str, str] = {
    "leonifelheim":     "Leo",
    "nuriiia___":       "Nuria",
    "daniela_gamer400": "Daniela",
    "cibernoman":       "Ciber",
    "blacknatti120":    "Natti",
    "sulykaiserff":     "Suly",
    "jotunbot":         "JotunBot",
    "hebenifelheim":    "Hebe",
}


def normalize_chatter_name(display_name: str | None) -> str:
    """
    'nuriiia___' -> 'Nuria', 'daniela_gamer400' -> 'Daniela'.

    Para los desconocidos: quita '@' inicial, recorta sufijos numéricos y
    underscores, parte por separadores no alfabéticos y devuelve el primer
    token capitalizado.

    Devuelve 'alguien' si la entrada está vacía o solo tiene caracteres
    raros. NUNCA lanza excepción (lo llamamos en el hot path).
    """
    if not display_name:
        return "alguien"

    raw = display_name.lstrip("@").strip()
    if not raw:
        return "alguien"

    key = raw.lower()
    if key in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[key]

    # Genérico: quitar trailing dígitos y underscores ('user_123' -> 'user').
    trimmed = re.sub(r"[_\d]+$", "", key)
    if not trimmed:
        # Nombre que era todo dígitos/underscores. Dejamos algo legible.
        return raw[:12] or "alguien"

    # Partir por separadores no alfanuméricos y coger el primer token útil.
    tokens = re.split(r"[_\W]+", trimmed)
    first = next((t for t in tokens if t), trimmed)
    return first.capitalize() if first else "alguien"