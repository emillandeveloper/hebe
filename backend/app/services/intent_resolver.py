# app/services/intent_resolver.py
from __future__ import annotations
from pathlib import Path
import joblib
from dataclasses import dataclass, field
from typing import Any, Optional
import json

from app.services.nlu_catalog import INTENTS, INTENT_KEYWORDS, SLOT_EXTRACTORS

@dataclass
class IntentFrame:
    type: str                 # "action" | "chat" | "clarify"
    intent: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    raw_text: str = ""
    source: str = "voice"     # "voice" | "ui"
    model: str = "gate"       # "gate" | "llm"

@dataclass
class NLUContext:
    last_intent: str | None = None
    last_slots: dict[str, Any] = field(default_factory=dict)

class HybridIntentResolver:
    """
    Resolver híbrido:
      1) Gate local (heurístico bootstrap; luego lo sustituyes por sklearn/fastText)
      2) Slot extraction local
      3) Fallback LLM JSON si el gate está inseguro o faltan slots
    """
    def __init__(self, llm=None, gate_threshold: float = 0.62, model_path: str = "models/intent_gate.joblib",):
        self.llm = llm
        self.gate_threshold = gate_threshold
        self.model_path = model_path

        self._clf = None
        self._load_model()

    def _normalize(self, s: str) -> str:
        import re
        import unicodedata

        s = (s or "").lower().strip()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")

        # elimina wakewords y muletillas
        fillers = ["hebe", "oye", "por favor", "eh", "vale"]
        for f in fillers:
            s = s.replace(f, "")

        # solo letras/numeros/espacios
        s = re.sub(r"[^a-z0-9\s]", " ", s)

        # colapsar espacios
        s = re.sub(r"\s+", " ", s).strip()

        return s
    
    def _resolve_model_path(self) -> Path:
        backend_root = Path(__file__).resolve().parents[2]
        return backend_root / self.model_path

    def _load_model(self):
        p = self._resolve_model_path()
        if p.exists():
            self._clf = joblib.load(p)
            print(f"[NLU] Loaded gate model: {p}")
        else:
            self._clf = None
            print(f"[NLU] Gate model not found: {p}")

    def resolve(self, text: str, ctx: NLUContext, source: str = "voice") -> IntentFrame:
        raw = (text or "").strip()
        t = self._normalize(raw)

        # 1) Gate local bootstrap: score por matches de keywords
        best_intent, best_score = self._gate(t)

        # Si no hay nada claro: chat
        if best_intent is None:
            return IntentFrame(type="chat", intent="chat", confidence=0.40, raw_text=raw, source=source, model="gate")

        # 2) Slots local
        slots = self._extract_slots(best_intent, raw)

        # 3) Missing slots?
        required = INTENTS[best_intent].required_slots
        missing = [k for k in required if not slots.get(k)]

        # 4) Fallback LLM si:
        #   - baja confianza
        #   - o faltan slots
        if (best_score < self.gate_threshold or missing) and self.llm is not None:
            llm_frame = self._llm_classify(raw, source=source)
            if llm_frame:
                ctx.last_intent = llm_frame.intent
                ctx.last_slots = dict(llm_frame.slots)
                return llm_frame

        frame_type = INTENTS[best_intent].kind  # "action"|"chat"
        if missing:
            return IntentFrame(
                type="clarify",
                intent=best_intent,
                confidence=best_score,
                slots=slots,
                missing=missing,
                raw_text=raw,
                source=source,
                model="gate",
            )

        out = IntentFrame(
            type=frame_type,
            intent=best_intent,
            confidence=best_score,
            slots=slots,
            raw_text=raw,
            source=source,
            model="gate",
        )
        ctx.last_intent = out.intent
        ctx.last_slots = dict(out.slots)
        return out

    def _gate(self, t: str) -> tuple[str | None, float]:
        # 1) Si hay modelo sklearn, úsalo
        if self._clf is not None:
            try:
                proba = self._clf.predict_proba([t])[0]
                classes = self._clf.classes_
                best_idx = int(proba.argmax())
                best_intent = str(classes[best_idx])
                best_score = float(proba[best_idx])
                if best_intent in INTENTS:
                    return best_intent, best_score
            except Exception:
                # si algo falla, caemos al heurístico
                pass

        # 2) fallback heurístico (tu implementación actual)
        best_intent = None
        best_score = 0.0
        for intent, kws in INTENT_KEYWORDS.items():
            hits = sum(1 for k in kws if k in t)
            if hits <= 0:
                continue
            score = min(0.55 + 0.15 * hits, 0.95)
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent, best_score

    def _extract_slots(self, intent: str, raw: str) -> dict[str, Any]:
        fn = SLOT_EXTRACTORS.get(intent)
        if not fn:
            return {}
        try:
            return fn(raw) or {}
        except Exception:
            return {}
        
    def _normalize_llm_slots(
        self,
        intent: str,
        slots: dict[str, Any],
        missing: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Normaliza nombres de slots que el LLM podría inventar (app_name vs app_raw, etc.)
        para que encajen con nuestro catálogo y dispatcher.
        """
        slots = dict(slots or {})
        missing = [str(m) for m in (missing or [])]

        # open_app: acepta app_name como alias de app_raw
        if intent == "open_app":
            if "app_name" in slots and "app_raw" not in slots:
                slots["app_raw"] = slots.pop("app_name")

            # También normaliza missing
            missing = ["app_raw" if m == "app_name" else m for m in missing]

        # memory_store: a veces el LLM usa "content" o "memory"
        if intent == "memory_store":
            if "content" in slots and "text" not in slots:
                slots["text"] = slots.pop("content")
            if "memory" in slots and "text" not in slots:
                slots["text"] = slots.pop("memory")
            missing = ["text" if m in ("content", "memory") else m for m in missing]

        return slots, missing

    def _llm_classify(self, raw: str, source: str) -> IntentFrame | None:
        """
        Usa el LLM para devolver JSON estricto.
        Si el LLM falla, devolvemos None y el engine cae a chat.
        """
        allowed_intents = list(INTENTS.keys())
        prompt = (
            "Eres un clasificador de intención para un asistente de escritorio.\n"
            "Devuelve SOLO JSON válido (sin texto extra).\n\n"
            f"Intents permitidos: {allowed_intents}\n"
            "Esquema:\n"
            "{"
            '"type":"action|chat|clarify",'
            '"intent":"<intent>",'
            '"confidence":0.0,'
            '"slots":{...},'
            '"missing":[...]}'
            "\n\n"
            "Reglas:\n"
            "- Si es conversación, type='chat' e intent='chat'.\n"
            "- Si falta información necesaria para ejecutar, type='clarify' y pon missing.\n"
            "- Para intent 'open_app' el slot requerido se llama EXACTAMENTE 'app_raw'.\n"
            "- Para intent 'memory_store' el slot requerido se llama EXACTAMENTE 'text'.\n"
            "- No inventes apps: usa app_raw tal como el usuario la diga.\n\n"
            f"Texto del usuario: {raw}\n"
        )

        try:
            out = self.llm.ask_stateless(prompt, temperature=0.0)
        except Exception:
            return None

        try:
            data = json.loads(out)
        except Exception:
            return None

        intent = str(data.get("intent", "chat"))
        if intent not in INTENTS:
            intent = "chat"

        frame_type = str(data.get("type", INTENTS[intent].kind))
        if frame_type not in ("action", "chat", "clarify"):
            frame_type = INTENTS[intent].kind

        conf = float(data.get("confidence", 0.50))
        conf = max(0.0, min(conf, 1.0))

        slots = data.get("slots") or {}
        if not isinstance(slots, dict):
            slots = {}

        missing = data.get("missing") or []
        if not isinstance(missing, list):
            missing = []

        # 🔥 NORMALIZACIÓN defensiva de slots del LLM
        slots, missing = self._normalize_llm_slots(intent, slots, missing)

        # Si el LLM dice "missing" pero realmente ya está presente, lo limpiamos
        required = INTENTS[intent].required_slots
        if required:
            missing = [k for k in missing if k in required and not slots.get(k)]

        return IntentFrame(
            type=frame_type,
            intent=intent,
            confidence=conf,
            slots=slots,
            missing=[str(x) for x in missing],
            raw_text=raw,
            source=source,
            model="llm",
        )