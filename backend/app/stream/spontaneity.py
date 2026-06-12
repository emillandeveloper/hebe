from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from app.cognitive.scheduler import InternalEvent
from app.stream.game_profiles import GameProfileStore
from app.stream.state import StreamSessionState


@dataclass(frozen=True)
class StreamSpontaneityConfig:
    companion_silence_sec: float = 20 * 60
    show_silence_sec: float = 9 * 60
    companion_jitter_sec: float = 2 * 60
    show_jitter_sec: float = 60
    global_stream_cooldown_sec: float = 4 * 60
    recent_voice_quiet_sec: float = 20
    max_context_age_sec: float = 5 * 60
    startup_grace_sec: float = 4 * 60
    chat_activity_window_sec: float = 180
    chat_active_message_threshold: int = 3
    chat_active_user_threshold: int = 1
    suppress_when_chat_active: bool = True
    companion_max_per_hour: int = 2
    show_max_per_hour: int = 5
    max_per_stream: int = 6
    require_specific_context: bool = False
    title_marker_ttl_sec: float = 55 * 60
    save_equip_topic_cooldown_sec: float = 60 * 60
    cooldown_key: str = "stream_idle_prompt_next_ts"


class StreamSpontaneityService:
    """
    Decides whether Hebe may proactively speak in stream.

    This service only makes the timing/policy decision. It does not synthesize
    text and does not deliver anything.
    """

    def __init__(
        self,
        *,
        config: StreamSpontaneityConfig | None = None,
        game_profiles: GameProfileStore | None = None,
        rng: random.Random | None = None,
        now_fn: Callable[[], float] | None = None,
    ):
        self.config = config or StreamSpontaneityConfig()
        self.game_profiles = game_profiles or GameProfileStore()
        self.rng = rng or random.Random()
        self._now_fn = now_fn

    def build_due_event(self, stream: StreamSessionState | None) -> Optional[InternalEvent]:
        now = self._now()
        readiness = self.evaluate(stream, now=now, live_override=False, mutate_baseline=True)
        if stream is not None:
            stream.last_stream_spontaneity_blocked_reason = readiness.get("blocked_reason")
        if not readiness["would_send"]:
            return None

        mode = readiness["presence_mode"]
        self._schedule_next(stream, mode, now)
        return self.build_event(stream, mode=mode, topic=readiness.get("candidate_topic"))

    def build_preview_event(self, stream: StreamSessionState | None, *, presence_mode: str | None = None) -> Optional[InternalEvent]:
        if not stream:
            return None
        mode = presence_mode or getattr(stream, "presence_mode", "companion") or "companion"
        if mode in {"silent", "reactive"}:
            mode = "companion"
        return self.build_event(stream, mode=mode)

    def build_event(self, stream: StreamSessionState, *, mode: str, topic: str | None = None) -> InternalEvent:
        payload = self._build_payload(stream, mode, topic=topic)
        return InternalEvent(
            event_type="twitch_idle_prompt",
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def evaluate(
        self,
        stream: StreamSessionState | None,
        *,
        now: float | None = None,
        live_override: bool = False,
        mutate_baseline: bool = False,
    ) -> dict:
        now = self._now() if now is None else float(now)
        result = {
            "would_send": False,
            "blocked_reason": "stream state missing",
            "stream_enabled": False,
            "twitch_live": "unknown",
            "presence_mode": "unknown",
            "context_fresh": False,
            "recent_chat_block": False,
            "recent_hebe_block": False,
            "cooldown_ready": False,
            "grace_block": False,
            "idle_enabled": False,
            "chat_active": False,
            "recent_chat_count": 0,
            "prompts_sent_hour": 0,
            "prompts_sent_stream": 0,
            "last_idle_topic": None,
            "recent_idle_topics": [],
            "candidate_topic": None,
            "next_possible_idle_prompt_ts": 0.0,
            "title_markers_fresh": [],
            "title_markers_stale": [],
            "specific_context_anchors": [],
        }
        if not stream:
            return result

        result["stream_enabled"] = bool(getattr(stream, "enabled", False))
        result["idle_enabled"] = bool(getattr(stream, "idle_spontaneity_enabled", True))
        result["presence_mode"] = (getattr(stream, "presence_mode", "reactive") or "reactive").strip().lower()
        if getattr(stream, "live_status_known", False):
            result["twitch_live"] = bool(getattr(stream, "is_live", False))
        if live_override:
            result["twitch_live"] = True

        if not result["stream_enabled"]:
            result["blocked_reason"] = "stream mode disabled"
            return result
        if not result["idle_enabled"]:
            result["blocked_reason"] = "idle_spontaneity_paused"
            return result
        if result["twitch_live"] == "unknown":
            result["blocked_reason"] = "live status unknown"
            return result
        if result["twitch_live"] is not True:
            result["blocked_reason"] = "stream is offline"
            return result

        context_updated_ts = float(getattr(stream, "stream_context_updated_ts", 0.0) or 0.0)
        result["context_fresh"] = bool(context_updated_ts and now - context_updated_ts <= self.config.max_context_age_sec)
        if not result["context_fresh"]:
            result["blocked_reason"] = "context stale"
            return result

        mode = (getattr(stream, "presence_mode", "reactive") or "reactive").strip().lower()
        if mode in {"silent", "reactive"}:
            result["blocked_reason"] = f"presence mode is {mode}"
            return result
        if mode not in {"companion", "show"}:
            result["blocked_reason"] = f"presence mode is unsupported: {mode}"
            return result

        grace_until = float(getattr(stream, "stream_spontaneity_grace_until_ts", 0.0) or 0.0)
        result["grace_block"] = bool(grace_until and now < grace_until)
        if result["grace_block"]:
            result["blocked_reason"] = "stream_grace_period"
            return result

        title_context = self._title_marker_context(stream, now)
        result["title_markers_fresh"] = title_context["fresh"]
        result["title_markers_stale"] = title_context["stale"]
        anchors = self._specific_context_anchors(stream, now, title_context=title_context)
        result["specific_context_anchors"] = anchors
        if self.config.require_specific_context and self._missing_primer_and_run_context(stream, now, title_context=title_context):
            result["blocked_reason"] = "no_session_primer_or_run_context"
            print("[HEBE][SPONTANEITY] skipped reason=no_session_primer_or_run_context", flush=True)
            return result
        if self._only_weak_recent_context(stream, now):
            result["blocked_reason"] = "no_high_quality_anchor"
            print("[HEBE][SPONTANEITY] skipped reason=no_high_quality_anchor", flush=True)
            return result
        if self.config.require_specific_context and not anchors:
            result["blocked_reason"] = "no_specific_context"
            print("[HEBE][SPONTANEITY] skipped reason=no_specific_context", flush=True)
            return result

        chat_snapshot = self._chat_activity_snapshot(stream, now)
        result["chat_active"] = chat_snapshot["active"]
        result["recent_chat_count"] = chat_snapshot["count"]
        if self.config.suppress_when_chat_active and chat_snapshot["active"]:
            result["blocked_reason"] = "chat_active"
            return result

        last_chat = float(getattr(stream, "last_chat_activity_ts", 0.0) or 0.0)
        if not last_chat:
            if mutate_baseline:
                stream.last_chat_activity_ts = now
            result["blocked_reason"] = "chat activity baseline not ready"
            return result

        required_silence = self._required_silence_sec(stream, mode)
        result["recent_chat_block"] = bool(now - last_chat < required_silence)
        if now - last_chat < required_silence:
            result["blocked_reason"] = "recent_chat_activity"
            return result

        last_spoken = float(getattr(stream, "last_hebe_stream_speak_ts", 0.0) or 0.0)
        result["recent_hebe_block"] = bool(last_spoken and now - last_spoken < self.config.global_stream_cooldown_sec)
        if last_spoken and now - last_spoken < self.config.global_stream_cooldown_sec:
            result["blocked_reason"] = "recent_hebe_message"
            return result

        next_prompt_ts = float((getattr(stream, "cooldowns", {}) or {}).get(self.config.cooldown_key, 0.0) or 0.0)
        result["cooldown_ready"] = not bool(next_prompt_ts and now < next_prompt_ts)
        if next_prompt_ts and now < next_prompt_ts:
            result["blocked_reason"] = "cooldown_active"
            result["next_possible_idle_prompt_ts"] = next_prompt_ts
            return result

        last_voice_ts = float(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        if last_voice_ts and now - last_voice_ts < self.config.recent_voice_quiet_sec:
            result["blocked_reason"] = "Leo spoke recently"
            return result

        recent_idle = list(getattr(stream, "recent_idle_messages", []) or [])
        result["prompts_sent_hour"] = sum(
            1 for item in recent_idle if now - float(item.get("timestamp", 0.0) or 0.0) <= 3600
        )
        result["prompts_sent_stream"] = int(getattr(stream, "idle_prompts_sent_stream", 0) or 0)
        max_hour = self.config.show_max_per_hour if mode == "show" else self.config.companion_max_per_hour
        if result["prompts_sent_hour"] >= max_hour:
            result["blocked_reason"] = "hourly_limit"
            return result
        if result["prompts_sent_stream"] >= self.config.max_per_stream:
            result["blocked_reason"] = "stream_limit"
            return result

        result["last_idle_topic"] = recent_idle[-1].get("topic") if recent_idle else None
        result["recent_idle_topics"] = [item.get("topic") for item in recent_idle[-8:] if item.get("topic")]
        topic = self._choose_topic(stream, now)
        if topic is None:
            result["blocked_reason"] = "topic_recently_used"
            return result
        result["candidate_topic"] = topic
        used_fact = self._recent_run_context_fact(stream, now)
        used_fact_id = used_fact.get("id") if used_fact else None
        if used_fact:
            print(
                "[HEBE][SPONTANEITY] "
                f"selected_anchor category={used_fact.get('category') or used_fact.get('kind')} "
                f"confidence={float(used_fact.get('confidence', 0.0) or 0.0):.2f}",
                flush=True,
            )
        print(f"[HEBE][SPONTANEITY] anchors={anchors}", flush=True)
        print(f"[HEBE][SPONTANEITY] generated topic={topic} used_fact_id={used_fact_id}", flush=True)

        result["would_send"] = True
        result["blocked_reason"] = "ready"
        result["cooldown_ready"] = True
        return result

    def reset_spontaneity_cooldowns(self, stream: StreamSessionState | None) -> int:
        if not stream or not isinstance(getattr(stream, "cooldowns", None), dict):
            return 0
        keys = [
            key for key in stream.cooldowns
            if key.startswith("stream_idle_prompt") or key.endswith("_idle_silence_sec")
        ]
        for key in keys:
            stream.cooldowns.pop(key, None)
        return len(keys)

    def start_grace_period(self, stream: StreamSessionState | None, *, now: float | None = None) -> None:
        if not stream:
            return
        now = self._now() if now is None else float(now)
        stream.stream_spontaneity_grace_until_ts = now + self.config.startup_grace_sec

    def _build_payload(self, stream: StreamSessionState, mode: str, topic: str | None = None) -> dict:
        profile = self.game_profiles.lookup(
            current_category=getattr(stream, "current_category", None),
            current_game=getattr(stream, "current_game", None),
            current_title=getattr(stream, "current_stream_title", None),
        )
        now = self._now()
        title_context = self._title_marker_context(stream, now)
        chat_snapshot = self._chat_activity_snapshot(stream, now)
        recent_idle = list(getattr(stream, "recent_idle_messages", []) or [])
        return {
            "reason": "stream_companion_prompt",
            "presence_mode": mode,
            "idle_topic": topic or self._choose_topic(stream, now) or "game_vibe",
            "title": getattr(stream, "current_stream_title", None),
            "current_game": getattr(stream, "current_game", None),
            "current_category": getattr(stream, "current_category", None),
            "current_tags": list(getattr(stream, "current_tags", []) or []),
            "playthrough_type": getattr(stream, "current_playthrough_type", None),
            "challenge": getattr(stream, "current_challenge", None),
            "stream_slot": getattr(stream, "current_stream_slot", None),
            "language_mode": getattr(stream, "language_mode", None),
            "spoiler_policy": getattr(stream, "spoiler_policy", "no_spoilers"),
            "session_primer": getattr(stream, "session_primer", None),
            "last_voice_event": getattr(stream, "last_voice_event", None),
            "last_voice_summary": getattr(stream, "last_voice_summary", None),
            "leo_mood_hint": getattr(stream, "leo_mood_hint", None),
            "run_context": {
                "objective": getattr(stream, "current_run_objective", None),
                "location": getattr(stream, "current_run_location", None),
                "phase": getattr(stream, "current_run_phase", None),
                "source": getattr(stream, "run_context_source", None),
                "updated_ts": getattr(stream, "run_context_updated_ts", 0.0),
                "facts": [
                    {
                        "id": item.get("id"),
                        "kind": item.get("kind"),
                        "category": item.get("category"),
                        "text": item.get("text"),
                        "summary": item.get("summary"),
                        "confidence": item.get("confidence"),
                        "raw_text": item.get("raw_text"),
                        "normalized_text": item.get("normalized_text"),
                    }
                    for item in list(getattr(stream, "recent_run_context_facts", []) or [])[-8:]
                    if item.get("text")
                ],
                "completed_markers": list(getattr(stream, "completed_run_markers", []) or []),
                "title_markers_fresh": title_context["fresh"],
                "title_markers_stale": title_context["stale"],
            },
            "chat_context": {
                "active": chat_snapshot["active"],
                "recent_count": chat_snapshot["count"],
                "recent_topics": chat_snapshot["topics"],
                "summary": chat_snapshot["summary"],
            },
            "specific_context_anchors": self._specific_context_anchors(stream, now, title_context=title_context, chat_snapshot=chat_snapshot),
            "used_fact_id": (self._recent_run_context_fact(stream, now) or {}).get("id"),
            "recent_idle_topics": [item.get("topic") for item in recent_idle[-8:] if item.get("topic")],
            "recent_idle_messages": [item.get("text") for item in recent_idle[-5:] if item.get("text")],
            "recent_style_motifs": [
                item.get("motif")
                for item in list(getattr(stream, "recent_style_motifs", []) or [])[-12:]
                if item.get("motif")
            ],
            "game_profile": profile.compact_prompt_context(),
        }

    def record_idle_message(self, stream: StreamSessionState | None, text: str, *, topic: str | None = None) -> None:
        if not stream:
            return
        now = self._now()
        entry = {
            "text": str(text or "").strip()[:240],
            "normalized_text": self._normalize_for_similarity(text),
            "topic": topic or "unknown",
            "timestamp": now,
            "game": getattr(stream, "current_game", None) or getattr(stream, "current_category", None),
            "playthrough": getattr(stream, "current_playthrough_type", None),
            "challenge": getattr(stream, "current_challenge", None),
        }
        messages = list(getattr(stream, "recent_idle_messages", []) or [])
        messages.append(entry)
        stream.recent_idle_messages = messages[-30:]
        stream.idle_prompts_sent_stream = int(getattr(stream, "idle_prompts_sent_stream", 0) or 0) + 1
        self._record_style_motifs(stream, text, now=now)

    def is_too_similar_to_recent(self, stream: StreamSessionState | None, text: str) -> bool:
        if not stream:
            return False
        current = set(self._normalize_for_similarity(text).split())
        if not current:
            return False
        repeated_phrases = ("revisa equipo", "guarda", "objetos", "level 1", "ramuh", "lindblum")
        lowered = str(text or "").lower()
        for item in list(getattr(stream, "recent_idle_messages", []) or [])[-8:]:
            previous_text = str(item.get("text") or "").lower()
            if any(phrase in lowered and phrase in previous_text for phrase in repeated_phrases):
                return True
            previous = set(str(item.get("normalized_text") or "").split())
            if previous:
                score = len(current & previous) / max(1, len(current | previous))
                if score > 0.58:
                    return True
        return False

    def _chat_activity_snapshot(self, stream: StreamSessionState, now: float) -> dict:
        window = self.config.chat_activity_window_sec
        messages = [
            item for item in list(getattr(stream, "recent_chat_messages", []) or [])
            if now - float(item.get("ts", 0.0) or 0.0) <= window
        ]
        users = {
            str(item.get("username") or "").strip().lower()
            for item in messages
            if str(item.get("username") or "").strip()
        }
        topics = [item.get("topic") for item in messages if item.get("topic")]
        return {
            "active": len(messages) >= self.config.chat_active_message_threshold
            and len(users) >= self.config.chat_active_user_threshold,
            "count": len(messages),
            "users": sorted(users),
            "topics": topics[-8:],
            "summary": ", ".join(dict.fromkeys(topics[-5:])) if topics else "none",
        }

    def _title_marker_context(self, stream: StreamSessionState, now: float) -> dict:
        markers = list(getattr(stream, "title_context_markers", []) or [])
        completed = {self._normalize_marker(item) for item in list(getattr(stream, "completed_run_markers", []) or [])}
        updated = float(getattr(stream, "title_context_updated_ts", 0.0) or 0.0)
        is_fresh = bool(updated and now - updated <= self.config.title_marker_ttl_sec)
        fresh: list[str] = []
        stale: list[str] = []
        for marker in markers:
            if self._normalize_marker(marker) in completed:
                stale.append(marker)
            elif is_fresh:
                fresh.append(marker)
            else:
                stale.append(marker)
        return {"fresh": fresh, "stale": stale}

    def _choose_topic(self, stream: StreamSessionState, now: float) -> str | None:
        recent_fact = self._recent_run_context_fact(stream, now)
        if recent_fact:
            category_topic = self._topic_for_fact(recent_fact)
            if category_topic:
                recent = list(getattr(stream, "recent_idle_messages", []) or [])
                if not recent or recent[-1].get("topic") != category_topic:
                    return category_topic

        topics = [
            "challenge_comment",
            "jrpg_trope",
            "game_vibe",
            "light_roast",
            "exploration_comment",
            "strategy_without_spoilers",
            "streamer_reaction_hook",
            "hydration_or_break",
            "save_reminder",
            "equipment_check",
            "resource_management",
        ]
        recent = list(getattr(stream, "recent_idle_messages", []) or [])
        last_topic = recent[-1].get("topic") if recent else None
        long_cooldown_topics = {"save_reminder", "equipment_check", "resource_management"}
        blocked: set[str] = set()
        for item in recent:
            topic = item.get("topic")
            if topic in long_cooldown_topics and now - float(item.get("timestamp", 0.0) or 0.0) < self.config.save_equip_topic_cooldown_sec:
                blocked.add(topic)
        available = [topic for topic in topics if topic != last_topic and topic not in blocked]
        if not available:
            return None
        return available[0]

    def _recent_run_context_fact(self, stream: StreamSessionState, now: float) -> dict | None:
        facts = [
            item for item in list(getattr(stream, "recent_run_context_facts", []) or [])
            if item.get("text") and float(item.get("expires_at", 0.0) or 0.0) > now
        ]
        if not facts:
            return None
        high_quality = [fact for fact in facts if self._is_high_quality_fact(fact)]
        if not high_quality:
            return None
        return sorted(
            high_quality,
            key=lambda fact: (float(fact.get("confidence", 0.0) or 0.0), float(fact.get("timestamp", 0.0) or 0.0)),
        )[-1]

    def _is_high_quality_fact(self, fact: dict) -> bool:
        category = str(fact.get("category") or fact.get("kind") or "")
        confidence = float(fact.get("confidence", 0.0) or 0.0)
        high_categories = {
            "combat_risk",
            "rng_dependency",
            "healing_or_recovery",
            "enemy_mechanic",
            "challenge_constraint",
            "progress_marker",
            "failure_or_death",
            "level_gap",
            "resource_management",
            "boss_or_area_difficulty",
            "guide_strategy",
        }
        weak_categories = {"phase", "objective", "ambient_note", "navigation_confusion"}
        if category in weak_categories:
            return confidence >= 0.82 and bool(self._specific_gameplay_terms(fact))
        if category in high_categories and confidence >= 0.62:
            return True
        return bool(self._specific_gameplay_terms(fact)) and confidence >= 0.7

    def _specific_gameplay_terms(self, fact: dict) -> set[str]:
        text = " ".join(str(fact.get(key) or "") for key in ("text", "summary", "raw_text", "normalized_text")).lower()
        terms = {
            "hp", "vida", "counter", "contraataque", "autopocion", "autopotion",
            "cura", "curarse", "rng", "suerte", "dados", "boss", "jefe",
            "enemigo", "ataque", "level", "nivel", "desafio", "challenge",
            "game over", "muerto", "matado", "recargar", "guardar",
        }
        return {term for term in terms if term in text}

    def _only_weak_recent_context(self, stream: StreamSessionState, now: float) -> bool:
        recent_facts = [
            item for item in list(getattr(stream, "recent_run_context_facts", []) or [])
            if float(item.get("expires_at", 0.0) or 0.0) > now
        ]
        if recent_facts and not any(self._is_high_quality_fact(item) for item in recent_facts):
            return True
        ignored_reason = str(getattr(stream, "last_ambient_context_ignored_reason", "") or "")
        ignored_ts = float(getattr(stream, "last_ambient_context_ignored_ts", 0.0) or 0.0)
        if ignored_reason == "generic_filler" and ignored_ts and now - ignored_ts <= self.config.max_context_age_sec and not recent_facts:
            return True
        return False

    def _topic_for_fact(self, fact: dict) -> str | None:
        category = str(fact.get("category") or fact.get("kind") or "")
        mapping = {
            "healing_item_effectiveness": "resource_management",
            "healing_or_recovery": "resource_management",
            "unexpected_attack": "strategy_without_spoilers",
            "guide_strategy": "strategy_without_spoilers",
            "enemy_mechanic": "strategy_without_spoilers",
            "low_hp": "resource_management",
            "combat_risk": "strategy_without_spoilers",
            "rng_dependency": "challenge_comment",
            "challenge_constraint": "challenge_comment",
            "failure_or_death": "challenge_comment",
            "resource_management": "resource_management",
            "boss_or_area_difficulty": "challenge_comment",
            "navigation_confusion": "exploration_comment",
            "progress_marker": "game_vibe",
            "repeated_failure": "challenge_comment",
            "level_gap": "challenge_comment",
        }
        return mapping.get(category)

    def _specific_context_anchors(
        self,
        stream: StreamSessionState,
        now: float,
        *,
        title_context: dict | None = None,
        chat_snapshot: dict | None = None,
    ) -> list[str]:
        anchors: list[str] = []
        if getattr(stream, "current_game", None) or getattr(stream, "current_category", None):
            anchors.append("game")
        if getattr(stream, "current_stream_title", None):
            anchors.append("title")
        title_context = title_context or self._title_marker_context(stream, now)
        if title_context.get("fresh"):
            anchors.append("title_markers")
        if getattr(stream, "current_playthrough_type", None):
            anchors.append("playthrough_type")
        if getattr(stream, "current_challenge", None):
            anchors.append("challenge")
        if getattr(stream, "session_primer", None):
            anchors.append("session_primer")
        run_updated = float(getattr(stream, "run_context_updated_ts", 0.0) or 0.0)
        if run_updated and now - run_updated <= 45 * 60:
            if (
                getattr(stream, "current_run_objective", None)
                or getattr(stream, "current_run_location", None)
                or getattr(stream, "current_run_phase", None)
                or self._recent_run_context_fact(stream, now)
            ):
                anchors.append("run_context")
        chat_snapshot = chat_snapshot or self._chat_activity_snapshot(stream, now)
        if chat_snapshot.get("topics"):
            anchors.append("chat_topic")
        last_voice_ts = float(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        if last_voice_ts and now - last_voice_ts <= 30 * 60 and getattr(stream, "last_voice_event", None):
            anchors.append("recent_voice_event")
        if getattr(stream, "last_raid_event", None):
            anchors.append("recent_event")
        return list(dict.fromkeys(anchors))

    def _missing_primer_and_run_context(self, stream: StreamSessionState, now: float, *, title_context: dict | None = None) -> bool:
        if getattr(stream, "session_primer", None):
            return False
        title_context = title_context or self._title_marker_context(stream, now)
        if title_context.get("fresh"):
            return False
        if getattr(stream, "current_run_objective", None) or getattr(stream, "current_run_location", None) or getattr(stream, "current_run_phase", None):
            return False
        if self._recent_run_context_fact(stream, now):
            return False
        return True

    def _record_style_motifs(self, stream: StreamSessionState, text: str, *, now: float) -> None:
        motifs = self.detect_style_motifs(text)
        if not motifs:
            return
        recent = list(getattr(stream, "recent_style_motifs", []) or [])
        recent.extend({"motif": motif, "timestamp": now, "text": str(text or "")[:160]} for motif in motifs)
        stream.recent_style_motifs = recent[-80:]

    def detect_style_motifs(self, text: str) -> list[str]:
        lowered = str(text or "").lower()
        motifs: list[str] = []
        motif_terms = {
            "cafe": ("cafe", "café", "coffee", "cafeina", "cafeína"),
            "energy": ("energia", "energía", "pilas", "cansancio"),
            "florist": ("florist", "florister", "flores"),
            "creator": ("creador", "creadores"),
            "chaos": ("caos", "caotico", "caótico"),
        }
        for motif, terms in motif_terms.items():
            if any(term in lowered for term in terms):
                motifs.append(motif)
        return motifs

    def motif_on_cooldown(self, stream: StreamSessionState | None, text: str, *, now: float | None = None) -> str | None:
        if not stream:
            return None
        now = self._now() if now is None else float(now)
        cooldown_min = float(__import__("os").environ.get("HEBE_STYLE_MOTIF_COOLDOWN_MINUTES", "90") or 90)
        configured = __import__("os").environ.get("HEBE_STYLE_OVERUSED_MOTIFS", "cafe,coffee,energy,florist,creator")
        overused = {item.strip().lower() for item in configured.split(",") if item.strip()}
        aliases = {"coffee": "cafe"}
        motifs = [aliases.get(motif, motif) for motif in self.detect_style_motifs(text)]
        recent = list(getattr(stream, "recent_style_motifs", []) or [])
        for motif in motifs:
            if motif not in overused:
                continue
            for item in reversed(recent):
                item_motif = aliases.get(str(item.get("motif") or "").lower(), str(item.get("motif") or "").lower())
                if item_motif == motif:
                    if now - float(item.get("timestamp", 0.0) or 0.0) < cooldown_min * 60 or cooldown_min <= 0:
                        print(f"[HEBE][STYLE] motif_blocked motif={motif} reason=cooldown", flush=True)
                        print(f"[HEBE][STYLE] recent_motifs={[entry.get('motif') for entry in recent[-12:]]}", flush=True)
                        return motif
                    print(f"[HEBE][STYLE] motif_blocked motif={motif} reason=same_stream", flush=True)
                    print(f"[HEBE][STYLE] recent_motifs={[entry.get('motif') for entry in recent[-12:]]}", flush=True)
                    return motif
        return None

    def _normalize_marker(self, text: str) -> str:
        return self._normalize_for_similarity(text)

    def _normalize_for_similarity(self, text: str) -> str:
        import re
        normalized = re.sub(r"[^a-z0-9áéíóúüñ ]+", " ", str(text or "").lower())
        return " ".join(normalized.split())

    def _required_silence_sec(self, stream: StreamSessionState, mode: str) -> float:
        cooldowns = getattr(stream, "cooldowns", None)
        key = f"{mode}_idle_silence_sec"
        if isinstance(cooldowns, dict) and key in cooldowns:
            return float(cooldowns[key] or 0.0)

        if mode == "show":
            return self.config.show_silence_sec
        return self.config.companion_silence_sec

    def _schedule_next(self, stream: StreamSessionState, mode: str, now: float) -> None:
        cooldowns = getattr(stream, "cooldowns", None)
        if not isinstance(cooldowns, dict):
            stream.cooldowns = {}
            cooldowns = stream.cooldowns

        base = self.config.show_silence_sec if mode == "show" else self.config.companion_silence_sec
        jitter = self.config.show_jitter_sec if mode == "show" else self.config.companion_jitter_sec
        cooldowns[self.config.cooldown_key] = now + base + self.rng.uniform(0, jitter)

    def _now(self) -> float:
        if self._now_fn is not None:
            return float(self._now_fn())
        from time import time

        return time()
