from __future__ import annotations

import re
import time
import unicodedata
import json
from difflib import SequenceMatcher
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any

from app.stream.game_profiles import GameProfileStore
from app.cognitive.wake_name_resolver import WakeNameResolver


CAP_GAME_GUIDANCE = "game.guidance"


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9\s'-]", " ", text).split())


@dataclass(slots=True)
class GameRunState:
    game: str = ""
    platform_version: str = ""
    playthrough_type: str = "casual"
    spoiler_policy: str = "spoiler_safe_hints"
    current_location: str = ""
    current_character: str = ""
    party_members: list[str] = field(default_factory=list)
    last_confirmed_progress: str = ""
    current_objective: str = ""
    challenge: str = ""
    known_constraints: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    provenance: str = "inferred"
    confidence: float = 0.0

    @classmethod
    def from_value(cls, value: Any) -> "GameRunState":
        if isinstance(value, cls):
            return value
        raw = value if isinstance(value, dict) else {}
        names = cls.__dataclass_fields__
        return cls(**{name: raw[name] for name in names if name in raw})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameGuidanceContext:
    game: str = ""
    normalized_game: str = ""
    game_confidence: float = 0.0
    location_or_area: str = ""
    current_character: str = ""
    party_members: list[str] = field(default_factory=list)
    chapter_disc_act: str = ""
    recent_event: str = ""
    current_objective: str = ""
    playthrough_type: str = "casual"
    challenge: str = ""
    spoiler_policy: str = "spoiler_safe_hints"
    source_context: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguity_reasons: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    should_search_web: bool = False
    should_use_rag: bool = True
    allowed_answer_depth: str = "hint_only"
    forbidden_content: list[str] = field(default_factory=list)
    query_kind: str = "progression"
    search_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameGuidanceDecision:
    context: GameGuidanceContext
    response_mode: str
    reason: str
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    web_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "response_mode": self.response_mode,
            "reason": self.reason,
            "rag_chunks": list(self.rag_chunks),
            "web_results": list(self.web_results),
        }


class GameGuidanceCapability:
    """Builds a spoiler-aware source plan; it never invents guide content."""

    def __init__(self, profile_store: GameProfileStore | None = None, search_provider: Any = None):
        self.profile_store = profile_store or GameProfileStore()
        self.search_provider = search_provider
        self.entity_catalog = self._load_entity_catalog()

    def looks_like_query(self, text: str, state_snapshot: dict[str, Any] | None = None) -> bool:
        normalized = _normalize(text)
        run = GameRunState.from_value((state_snapshot or {}).get("game_run_state"))
        game, confidence = self._resolve_game(text, run)
        guidance = bool(re.search(
            r"\b(?:donde|como|que\s+(?:hago|toca|debo)|siguiente|objetivo|boss|jefe|"
            r"derrot|mat|consig|encuentr|item|objeto|build|deck|estrateg|prepar|ruta|romp|"
            r"explic|historia|final|spoiler)\w*\b",
            normalized,
        ))
        location_state = bool(re.search(r"\b(?:estoy|ando|sigo)\s+en\b", normalized))
        named_game_target = bool(re.search(r"\b(?:contra|boss|jefe)\s+[a-z0-9'-]+", normalized))
        return bool((guidance or location_state) and (
            (game and confidence >= 0.55) or named_game_target
        ))

    def evaluate(self, context: Any) -> GameGuidanceDecision:
        text = str(getattr(context, "input_text", "") or "")
        snapshot = getattr(context, "state_snapshot", {}) or {}
        run = GameRunState.from_value(snapshot.get("game_run_state"))
        game, game_confidence = self._resolve_game(text, run)
        normalized = _normalize(text)
        raw_location = self._extract_slot(
            normalized,
            r"\b(?:(?:estoy|ando|sigo)\s+en|(?:llegado|llegar|llego)\s+a)\s+"
            r"([^,?.]+?)(?=\s+(?:y\s+)?(?:no\s+se|que|cual|como|donde|una\s+vez)\b|$)",
        )
        location = self.normalize_entity(raw_location, "location", game) if raw_location else run.current_location
        character = self._extract_slot(normalized, r"\b(?:controlo|llevo|manejo|juego\s+con)\s+(?:a\s+)?([^,?.]+)") or run.current_character
        query_kind = self._query_kind(normalized)
        playthrough_type = run.playthrough_type
        if query_kind == "mechanics" and playthrough_type == "casual":
            playthrough_type = "break_the_game"
        elif query_kind in {"boss", "build"} and playthrough_type == "casual":
            playthrough_type = "strategy_mode"
        concrete = query_kind in {"item", "boss", "build", "mechanics"}
        recent_event = run.last_confirmed_progress
        objective = run.current_objective
        phase = self._extract_slot(normalized, r"\b(?:disco|disc|acto|act|capitulo|chapter)\s+([a-z0-9]+)")
        ambiguity: list[str] = []
        if not game:
            ambiguity.append("game_unknown")
        if query_kind == "progression":
            if not location:
                ambiguity.append("location_unknown")
            if not character:
                ambiguity.append("character_unknown")
            if not (phase or recent_event or objective):
                ambiguity.append("story_phase_unknown")
        needs_clarification = bool(ambiguity) and (query_kind == "progression" or not game)
        raw_rag_chunks = list(getattr(context, "relevant_chunks", []) or [])
        rag_chunks = self._filter_rag_by_game(raw_rag_chunks, game)
        should_use_rag = True
        should_search = bool(concrete and game and not rag_chunks and not needs_clarification)
        policy = self._spoiler_policy(run, game)
        story_sensitive = bool(re.search(
            r"\b(?:ending|giro|twist|muere|identidad|historia|plot)\w*\b|\bfinal\s+(?:de|del|explicado)\b",
            normalized,
        ))
        spoiler_permission = bool(re.search(r"\b(?:con|permito|acepto|puedes\s+dar)\s+spoilers?\b", normalized))
        if story_sensitive and policy != "full_guide_allowed" and not spoiler_permission:
            ambiguity.append("major_spoiler_permission_required")
            needs_clarification = True
            should_search = False
        mechanics_depth = query_kind in {"boss", "build", "mechanics"} or run.playthrough_type in {
            "challenge_run", "level_1_challenge", "break_the_game", "strategy_mode",
        }
        allowed_depth = "mechanics_detailed" if mechanics_depth else "hint_only" if policy != "full_guide_allowed" else "full_guide"
        forbidden = ["unrequested_story_reveals", "future_plot_events"]
        if allowed_depth == "hint_only":
            forbidden.extend(["exact_progression_steps_without_source", "future_boss_or_item_reveals"])
        search_query = self._build_search_query(
            game, location, character, query_kind, run, policy, playthrough_type
        ) if should_search else ""
        sources = {
            "user_input": text,
            "GameRunState": run.to_dict() if run.game else None,
            "current_live_session": snapshot.get("live_session") or snapshot.get("stream_context"),
            "stream_schedule": snapshot.get("stream_schedule"),
            "memory_rag": bool(rag_chunks),
            "external_search": False,
        }
        confidence = min(1.0, game_confidence + (0.12 if location else 0) + (0.12 if character else 0))
        guidance = GameGuidanceContext(
            game=game, normalized_game=_normalize(game), game_confidence=game_confidence,
            location_or_area=location, current_character=character, party_members=list(run.party_members), chapter_disc_act=phase,
            recent_event=recent_event, current_objective=objective,
            playthrough_type=playthrough_type, challenge=run.challenge,
            spoiler_policy=policy, source_context=sources, confidence=confidence,
            ambiguity_reasons=ambiguity, needs_clarification=needs_clarification,
            should_search_web=should_search, should_use_rag=should_use_rag,
            allowed_answer_depth=allowed_depth, forbidden_content=forbidden,
            query_kind=query_kind, search_query=search_query,
        )
        web_results: list[dict[str, Any]] = []
        if should_search and self.search_provider is not None:
            try:
                web_results = list(self.search_provider.search(search_query) or [])
                guidance.source_context["external_search"] = bool(web_results)
            except Exception as exc:
                print(f"[HEBE][GAME_SOURCE] tier=web status=skipped reason={type(exc).__name__}", flush=True)
        mode = "game_guidance_clarification" if needs_clarification else "game_guidance"
        reason = "ambiguous_run_state" if needs_clarification else "structured_game_guidance"
        self._log(guidance, rag_chunks, web_results)
        return GameGuidanceDecision(guidance, mode, reason, rag_chunks, web_results)

    def parse_clarification_answer(self, pending: dict[str, Any], text: str) -> dict[str, Any]:
        if str(pending.get("kind") or "") != "game_guidance_clarification":
            return {}
        game = str(pending.get("game") or "")
        missing = set(pending.get("missing_fields") or [])
        clean_text, stripped = self._strip_assistant_aliases(text)
        for alias in stripped:
            print(f"[HEBE][GAME_ENTITY] stripped_alias={alias} reason=hebe_alias", flush=True)
        updates: dict[str, Any] = {}
        if missing & {"current_character", "party_members"} and self._looks_like_party_answer(clean_text):
            party = self._extract_party_members(clean_text, game)
            if party:
                updates["party_members"] = party
                updates["current_character"] = party[0]
        raw_location = self._extract_slot(
            _normalize(clean_text),
            r"\b(?:llegando\s+a|entrando\s+en|al|en|a)\s+([a-z0-9'-]+(?:\s+[a-z0-9'-]+){0,3})",
        )
        if raw_location and missing & {"current_location", "story_phase", "recent_event"}:
            updates["current_location"] = self.normalize_entity(raw_location, "location", game)
        event = self._extract_slot(
            _normalize(clean_text),
            r"\b(?:vengo\s+de|acabo\s+de|despues\s+de)\s+(.+)$",
        )
        if event and missing & {"story_phase", "recent_event", "last_confirmed_progress"}:
            updates["recent_event"] = event
            updates["last_confirmed_progress"] = event
        if updates:
            updates.update({
                "game": game,
                "spoiler_policy": str(pending.get("spoiler_policy") or "spoiler_safe_hints"),
                "provenance": "leo_clarification",
                "confidence": 0.92,
                "last_updated": time.time(),
            })
        return updates

    @staticmethod
    def missing_fields(guidance: GameGuidanceContext | dict[str, Any]) -> list[str]:
        reasons = set(
            guidance.ambiguity_reasons if isinstance(guidance, GameGuidanceContext)
            else guidance.get("ambiguity_reasons") or []
        )
        fields: list[str] = []
        if "game_unknown" in reasons: fields.append("game")
        if "location_unknown" in reasons: fields.append("current_location")
        if "character_unknown" in reasons: fields.extend(["current_character", "party_members"])
        if "story_phase_unknown" in reasons: fields.extend(["story_phase", "recent_event", "current_objective"])
        if "major_spoiler_permission_required" in reasons: fields.append("spoiler_permission")
        return list(dict.fromkeys(fields))

    def normalize_entity(self, raw: str, entity_type: str, game: str) -> str:
        value = _normalize(raw)
        section = self._catalog_for_game(game)
        collection = section.get("characters" if entity_type == "party_member" else "locations", {})
        best_name, best_score = str(raw).strip().title(), 0.0
        for canonical, aliases in collection.items():
            for alias in [canonical, *list(aliases or [])]:
                candidate = _normalize(alias)
                score = (
                    1.0 if value == candidate
                    else .98 if candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", value)
                    else SequenceMatcher(None, value, candidate).ratio()
                )
                if score > best_score:
                    best_name, best_score = canonical, score
        threshold = 0.78 if len(value) >= 4 else 0.95
        normalized = best_name if best_score >= threshold else str(raw).strip().title()
        print(
            f"[HEBE][GAME_NORMALIZE] raw={raw} normalized={normalized} confidence={best_score:.2f}",
            flush=True,
        )
        return normalized

    def _extract_party_members(self, text: str, game: str) -> list[str]:
        value = re.sub(
            r"^.*?\b(?:con|controlo|controlando|llevo|manejando|party|grupo)\b\s+(?:a\s+)?",
            "", text, flags=re.IGNORECASE,
        )
        parts = re.split(r"\s*(?:,|;|\by\b|\be\b|\band\b)\s*", value, flags=re.IGNORECASE)
        ignored = {"estoy", "voy", "ahora", "mismo", "grupo", "party", "con"}
        result: list[str] = []
        aliases = self._assistant_aliases()
        for part in parts:
            tokens = [token for token in re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", part, flags=re.UNICODE) if _normalize(token) not in ignored]
            if not tokens:
                continue
            raw_name = " ".join(tokens[-2:] if len(tokens) > 1 else tokens)
            if _normalize(raw_name) in aliases:
                print(f"[HEBE][GAME_ENTITY] stripped_alias={raw_name} reason=hebe_alias", flush=True)
                continue
            normalized = self.normalize_entity(raw_name, "party_member", game)
            print(f"[HEBE][GAME_ENTITY] raw={raw_name} type=party_member normalized={normalized}", flush=True)
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def _looks_like_party_answer(self, text: str) -> bool:
        normalized = _normalize(text)
        if re.search(r"\b(?:con|controlo|controlando|llevo|manejando|party|grupo)\b", normalized):
            return True
        if re.search(r"\b(?:lleg|entr|sal|vengo|acabo|estoy|ando|sigo)\w*\s+(?:a|al|en|de)\b", normalized):
            return False
        names = self._assistant_aliases()
        tokens = [token for token in normalized.split() if token not in names]
        return bool(tokens) and len(tokens) <= 5

    def _strip_assistant_aliases(self, text: str) -> tuple[str, list[str]]:
        aliases = self._assistant_aliases()
        stripped: list[str] = []
        pieces: list[str] = []
        for piece in re.split(r"([,;])", str(text or "")):
            clean = piece.strip(" .!?")
            if _normalize(clean) in aliases:
                stripped.append(clean)
                continue
            pieces.append(piece)
        return "".join(pieces), stripped

    @staticmethod
    def _assistant_aliases() -> set[str]:
        configured = set(WakeNameResolver.canonical_names) | {"heba", "h"}
        return {_normalize(alias) for alias in configured}

    def _filter_rag_by_game(self, chunks: list[dict[str, Any]], game: str) -> list[dict[str, Any]]:
        if not chunks or not game:
            return []
        section = self._catalog_for_game(game)
        aliases = {_normalize(game), *(_normalize(alias) for alias in section.get("game_aliases", []))}
        matched: list[dict[str, Any]] = []
        for chunk in chunks:
            haystack = _normalize(" ".join(str(chunk.get(key) or "") for key in ("game", "subject", "title", "text", "summary_text")))
            if any(alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", haystack) for alias in aliases):
                matched.append(chunk)
        print(
            f"[HEBE][GAME_SOURCE] tier=rag status={'used' if matched else 'skipped'} "
            f"reason={'game_match' if matched else 'game_mismatch'}",
            flush=True,
        )
        return matched

    def _catalog_for_game(self, game: str) -> dict[str, Any]:
        normalized = _normalize(game)
        for section in self.entity_catalog.values():
            if normalized in {_normalize(alias) for alias in section.get("game_aliases", [])}:
                return section
        return {}

    @staticmethod
    def _load_entity_catalog() -> dict[str, Any]:
        try:
            return json.loads(Path(__file__).with_name("game_entity_aliases.json").read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[HEBE][GAME_NORMALIZE] catalog_load_failed={type(exc).__name__}", flush=True)
            return {}

    def _resolve_game(self, text: str, run: GameRunState) -> tuple[str, float]:
        normalized = _normalize(text)
        for profile in self.profile_store.profiles:
            if profile.game_slug == "generic_jrpg_rpg":
                continue
            for name in [profile.canonical_title, *profile.aliases]:
                candidate = _normalize(name)
                if candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized):
                    return profile.canonical_title, max(0.8, profile.confidence)
        if run.game:
            return run.game, max(0.65, run.confidence)
        match = re.search(
            r"\b(?:en|sobre|para|romper|optimizar|optimize|break)\s+"
            r"([A-Z][\w'’-]+(?:\s+(?:of|the|[A-Z][\w'’-]+)){0,4})",
            text,
        )
        return (match.group(1).strip(), 0.6) if match else ("", 0.0)

    @staticmethod
    def _extract_slot(text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _query_kind(text: str) -> str:
        if re.search(r"\b(?:boss|jefe|derrot|mat)\w*\b|\bcombate\b.*\bcontra\b", text): return "boss"
        if re.search(r"\b(?:item|objeto|arma|consig|encuentr)\w*\b", text): return "item"
        if re.search(r"\b(?:build|deck|baraja|equipo|prepar|estrateg)\w*\b", text): return "build"
        if re.search(r"\b(?:romp|optim|mecanic)\w*\b", text): return "mechanics"
        return "progression"

    def _spoiler_policy(self, run: GameRunState, game: str) -> str:
        if run.game and _normalize(run.game) == _normalize(game) and run.spoiler_policy:
            return run.spoiler_policy
        profile = self.profile_store.lookup(current_game=game)
        return "no_story_spoilers" if profile.spoiler_policy == "no_spoilers" else profile.spoiler_policy

    @staticmethod
    def _build_search_query(
        game: str, location: str, character: str, kind: str,
        run: GameRunState, policy: str, playthrough_type: str,
    ) -> str:
        parts = [game, kind]
        parts.extend(value for value in (location, character, run.challenge, playthrough_type) if value and value != "casual")
        if policy != "full_guide_allowed":
            parts.append("spoiler safe mechanics")
        return " ".join(dict.fromkeys(parts))

    @staticmethod
    def _log(guidance: GameGuidanceContext, rag: list, web: list) -> None:
        print(
            f"[HEBE][GAME_GUIDANCE] game={guidance.game or 'unknown'} location={guidance.location_or_area or 'unknown'} "
            f"confidence={guidance.confidence:.2f} needs_clarification={str(guidance.needs_clarification).lower()} "
            f"should_search={str(guidance.should_search_web).lower()} reason={guidance.ambiguity_reasons or 'ready'}",
            flush=True,
        )
        used_context = [name for name, value in guidance.source_context.items() if value and name != "user_input"]
        print(f"[HEBE][GAME_CONTEXT] source={'+'.join(used_context) or 'user_input'}", flush=True)
        print(f"[HEBE][GAME_SPOILER_POLICY] policy={guidance.spoiler_policy} allowed_depth={guidance.allowed_answer_depth}", flush=True)
        print(f"[HEBE][GAME_SOURCE] tier=local_context status=used reason=structured_input", flush=True)
        print(f"[HEBE][GAME_SOURCE] tier=rag status={'used' if rag else 'skipped'} reason={'chunks_available' if rag else 'no_chunks'}", flush=True)
        print(f"[HEBE][GAME_SOURCE] tier=web status={'used' if web else 'skipped'} reason={'results_available' if web else 'not_needed_or_unavailable'}", flush=True)
