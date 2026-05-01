"""
Métricas acumuladas de un stream para revisión post-mortem.

Pensado para uso ligero: instanciar uno por sesión de stream en el
ResponseSynthesizer, llamar a `record(...)` en cada respuesta y volcar
con `log_summary()` cuando termine el stream o cada N mensajes.

NO persiste a disco. Si la sesión se reinicia, los datos se pierden.
Para análisis histórico, se basa en grep sobre el log de stdout.

Usa print(..., flush=True) para mantener consistencia con el resto
del proyecto (que NO usa logging Python).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class StreamReplyStats:
    """
    Acumulador en memoria de cómo van las respuestas de Hebe en stream.

    Campos:
    - total: respuestas generadas (incluye las que se publicaron tras retry).
    - retried: cuántas necesitaron al menos un retry por detección de helper.
    - salvaged: de las retried, cuántas terminaron limpias tras el retry.
    - published_with_helper: retried que NO se salvaron y se publicaron igual.
    - pattern_count: histograma de qué patrones helper engancharon.
    - by_chatter: cuántas respuestas dirigidas a cada chatter (nombre limpio).
    """

    total: int = 0
    retried: int = 0
    salvaged: int = 0
    published_with_helper: int = 0
    pattern_count: Counter = field(default_factory=Counter)
    by_chatter: Counter = field(default_factory=Counter)

    def record(
        self,
        *,
        chatter: str,
        helper_hits: list[str],
        retried: bool,
        salvaged: bool,
    ) -> None:
        """
        Registra una respuesta. Llamar una vez por respuesta generada,
        haya tenido retries o no.

        helper_hits: lista de nombres de patrón detectados (uno por intento).
        retried: True si hubo al menos un retry.
        salvaged: True si retried Y la respuesta final salió limpia.
        """
        self.total += 1
        self.by_chatter[chatter] += 1
        for pat in helper_hits:
            self.pattern_count[pat] += 1
        if retried:
            self.retried += 1
            if salvaged:
                self.salvaged += 1
            else:
                self.published_with_helper += 1

    def log_summary(self) -> None:
        """
        Vuelca el estado actual a stdout con tag [HEBE][STREAM_SUMMARY].

        Pensado para grep posterior sobre los logs:
        $ grep "STREAM_SUMMARY" hebe_stream.log
        """
        retried_pct = (self.retried / self.total * 100) if self.total else 0.0
        salvaged_pct = (self.salvaged / self.retried * 100) if self.retried else 0.0
        print(
            f"[HEBE][STREAM_SUMMARY] "
            f"total={self.total} "
            f"retried={self.retried} ({retried_pct:.1f}%) "
            f"salvaged={self.salvaged} ({salvaged_pct:.1f}% del retried) "
            f"published_with_helper={self.published_with_helper} "
            f"patterns={dict(self.pattern_count.most_common())} "
            f"top_chatters={dict(self.by_chatter.most_common(8))}",
            flush=True,
        )

    def reset(self) -> None:
        """Volver a cero (útil al iniciar nuevo stream sin reiniciar backend)."""
        self.total = 0
        self.retried = 0
        self.salvaged = 0
        self.published_with_helper = 0
        self.pattern_count.clear()
        self.by_chatter.clear()