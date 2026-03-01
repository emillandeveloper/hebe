from __future__ import annotations

from typing import Callable, Optional
import requests


EmitFn = Callable[[str, dict], None]


class WikiES:
    """
    Wikipedia ES summary via REST API.
    Returns first paragraph (same behavior you had in engine).
    """

    def __init__(self, emit: Optional[EmitFn] = None):
        self.emit = emit

    def summary_first_paragraph(self, query: str) -> str:
        try:
            q = (query or "").strip()
            if not q:
                return "No me diste un tema para buscar."

            url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}"
            r = requests.get(url, timeout=10)

            if self.emit:
                self.emit("wiki.request", {"url": url, "status": r.status_code})

            if r.status_code != 200:
                return f"Error en Wikipedia: Código {r.status_code}"

            data = r.json() or {}
            extract = data.get("extract", "")
            if not extract:
                return "No encontré información relevante en Wikipedia."

            first = extract.split(". ")[0] + "." if ". " in extract else extract
            return first

        except Exception as e:
            return f"Error al buscar en Wikipedia: {e}"