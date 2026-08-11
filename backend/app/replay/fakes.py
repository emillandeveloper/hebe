from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np


class UnexpectedFixtureCall(RuntimeError):
    pass


class DeterministicOutcomeQueue:
    def __init__(self, configured: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._queues = {str(key): deque(dict(item) for item in rows) for key, rows in (configured or {}).items()}
        self.attempts: list[dict[str, Any]] = []

    def configure_next(self, operation: str, outcome: dict[str, Any]) -> None:
        self._queues.setdefault(str(operation), deque()).append(dict(outcome or {}))

    def take(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = str(operation)
        if key not in self._queues or not self._queues[key]:
            outcome = {"success": False, "status": "unconfigured", "reason": "external_outcome_missing"}
        else:
            outcome = dict(self._queues[key].popleft())
        attempt = {"operation": key, "payload": dict(payload), "outcome": outcome}
        self.attempts.append(attempt)
        status = str(outcome.get("status") or "").lower()
        if status in {"timeout", "unknown"}:
            raise TimeoutError(str(outcome.get("reason") or status))
        return outcome


@dataclass(slots=True)
class RecordingSpeech:
    outcomes: DeterministicOutcomeQueue
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, text: str, language: str = "es", **metadata: Any) -> bool:
        request = {"text": str(text or ""), "language": language, "metadata": dict(metadata)}
        self.requests.append(request)
        try:
            outcome = self.outcomes.take("tts.speak", {"text": request["text"], "language": language})
        except TimeoutError:
            return False
        return bool(outcome.get("success", True))


class FakeSTT:
    def __init__(self) -> None:
        self.tts_playback = False
        self.last_tts_text = ""

    def set_tts_playback(self, active: bool, text: str = "") -> None:
        self.tts_playback = bool(active)
        self.last_tts_text = str(text or "")

    def init(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeWinAutomation:
    def __init__(self, outcomes: DeterministicOutcomeQueue) -> None:
        self.outcomes = outcomes
        self.attempts: list[dict[str, Any]] = []

    def open_app(self, app_id: str, **payload: Any) -> dict[str, Any]:
        request = {"app_id": str(app_id or ""), **payload}
        self.attempts.append(request)
        return self.outcomes.take("desktop.open_app", request)

    def handle_volume_command(self, *_args: Any, **_kwargs: Any) -> bool:
        outcome = self.outcomes.take("desktop.volume", {})
        return bool(outcome.get("success"))


class FakeTwitch:
    def __init__(self, outcomes: DeterministicOutcomeQueue, resolution_fixtures: dict[str, dict[str, Any]] | None = None) -> None:
        self.outcomes = outcomes
        self.channel_name = "leonifelheim"
        self.bot_username = "HebeNifelheim"
        self.is_connected = True
        self.helix_client = self
        self.last_status_by_endpoint: dict[str, int] = {}
        self._stream: dict[str, Any] | None = None
        self._channel: dict[str, Any] = {}
        self.identities: dict[str, dict[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self.resolution_fixtures = {str(key).lower(): dict(value) for key, value in (resolution_fixtures or {}).items()}

    @property
    def attempts(self) -> list[dict[str, Any]]:
        return self.outcomes.attempts

    def is_available(self) -> bool:
        return True

    def configure_stream_metadata(self, payload: dict[str, Any]) -> None:
        live = bool(payload.get("is_live", True))
        data = {
            "id": str(payload.get("stream_id") or payload.get("id") or "replay-stream"),
            "title": str(payload.get("title") or ""),
            "game_id": str(payload.get("game_id") or ""),
            "game_name": str(payload.get("game") or payload.get("category") or payload.get("game_name") or ""),
            "tags": list(payload.get("tags") or []),
            "started_at": payload.get("started_at"),
        }
        self._stream = data if live else None
        self._channel = dict(data)

    def get_stream(self) -> dict[str, Any] | None:
        self.last_status_by_endpoint["get_streams"] = 200
        return dict(self._stream) if self._stream else None

    def get_current_stream(self) -> dict[str, Any] | None:
        return self.get_stream()

    def get_channel_info(self) -> dict[str, Any]:
        self.last_status_by_endpoint["get_channel_information"] = 200
        return dict(self._channel)

    def remember_chat_message(self, *, username: str, display_name: str = "", text: str = "") -> None:
        login = str(username or "").lower()
        if login:
            self.identities[login] = {"username": username, "display_name": display_name or username, "text": text}

    def remember_identity(self, *, user_id: str, login: str, display_name: str = "") -> None:
        key = str(login or "").lower()
        if key:
            self.identities[key] = {"user_id": str(user_id or ""), "username": login, "display_name": display_name or login}

    @staticmethod
    def normalize_twitch_username(username: str) -> str:
        target = re.sub(r"\s+", "", str(username or "").strip().lstrip("@"))
        return target if re.fullmatch(r"[A-Za-z0-9_]{3,25}", target) else ""

    def resolve_user(self, raw_target: str) -> str | None:
        query = self.normalize_twitch_username(raw_target).lower()
        if not query:
            return None
        if query in self.resolution_fixtures:
            return str(self.resolution_fixtures[query].get("username") or "") or None
        if query in self.aliases:
            return self.aliases[query]
        exact = self.identities.get(query)
        if exact:
            return str(exact.get("username") or query)
        matches = [str(row.get("username") or key) for key, row in self.identities.items() if key.startswith(query)]
        return matches[0] if len(matches) == 1 else None

    def resolve_user_details(self, raw_target: str, intent: str = "") -> dict[str, Any] | None:
        fixture = self.resolution_fixtures.get(self.normalize_twitch_username(raw_target).lower())
        if fixture is not None:
            return dict(fixture)
        username = self.resolve_user(raw_target)
        if not username:
            return None
        row = self.identities.get(username.lower(), {})
        return {
            "username": username,
            "display_name": row.get("display_name") or username,
            "user_id": row.get("user_id") or "",
            "confidence": 0.98,
            "candidates": [username],
            "reason": "replay_stable_identity",
        }

    def send_message(self, text: str) -> bool:
        outcome = self.outcomes.take("twitch.send_message", {"text": str(text or "")})
        return bool(outcome.get("success"))

    def shoutout(self, username: str) -> dict[str, Any]:
        target = self.normalize_twitch_username(username)
        outcome = self.outcomes.take("twitch.shoutout", {"target": target, "command": self.build_shoutout_command(target)})
        return outcome

    def build_shoutout_command(self, username: str) -> str:
        return f"!so {self.normalize_twitch_username(username)}"

    def remember_raid(self, *, username: str, viewer_count: int = 0) -> None:
        self.remember_identity(user_id="", login=username, display_name=username)


class FixtureModel:
    """Semantic-key fixture model. Unknown calls fail closed."""

    def __init__(self, fixtures: dict[str, Any] | None = None, *, label: str = "model") -> None:
        self.fixtures = dict(fixtures or {})
        self.label = label
        self.calls: list[dict[str, Any]] = []

    def _key(self, *, purpose: str = "", system: str = "", user: str = "", schema: Any = None) -> str:
        explicit = str(purpose or "").strip()
        if explicit:
            return explicit
        probe = f"{system}\n{user}".lower()
        semantic = "generic"
        for name, tokens in {
            "promotion_clarification": ("shoutout", "promo", "promotion"),
            "memory_extraction": ("long-term memory", "memories"),
            "stream_response": ("twitch", "stream"),
            "game_research": ("video game", "spoiler"),
        }.items():
            if any(token in probe for token in tokens):
                semantic = name
                break
        schema_digest = hashlib.sha256(json.dumps(schema, sort_keys=True, default=str).encode()).hexdigest()[:8] if schema else "none"
        return f"{semantic}:v1:{schema_digest}"

    def _take(self, key: str, call: dict[str, Any]) -> Any:
        self.calls.append({"key": key, **call})
        if key in self.fixtures:
            return self.fixtures[key]
        prefix = key.split(":v1:", 1)[0]
        if prefix in self.fixtures:
            return self.fixtures[prefix]
        if "*" in self.fixtures:
            return self.fixtures["*"]
        raise UnexpectedFixtureCall(f"unexpected_{self.label}_call:{key}")

    def chat_structured(self, *, system_prompt: str = "", user_prompt: str = "", schema: Any = None, purpose: str = "", **_kwargs: Any) -> dict[str, Any]:
        key = self._key(purpose=purpose, system=system_prompt, user=user_prompt, schema=schema)
        value = self._take(key, {"method": "chat_structured"})
        return dict(value or {})

    def chat(self, messages: Any = None, *, system_prompt: str = "", user_prompt: str = "", purpose: str = "", **_kwargs: Any) -> str:
        text = user_prompt or json.dumps(messages or [], ensure_ascii=False, default=str)
        key = self._key(purpose=purpose, system=system_prompt, user=text)
        return str(self._take(key, {"method": "chat"}) or "")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.chat(user_prompt=prompt, **kwargs)


class FixtureResearchProvider:
    def __init__(self, fixtures: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.fixtures = {str(key): [dict(row) for row in rows] for key, rows in (fixtures or {}).items()}
        self.calls: list[dict[str, str]] = []

    def search(self, query: str, constraints: dict[str, Any] | None = None, *, cache_key: str = "", **_kwargs: Any) -> list[dict[str, Any]]:
        key = str(cache_key or query).strip()
        self.calls.append({"key": key, "query": str(query), "constraints": json.dumps(constraints or {}, sort_keys=True)})
        if key in self.fixtures:
            rows = self.fixtures[key]
        elif "*" in self.fixtures:
            rows = self.fixtures["*"]
        else:
            raise UnexpectedFixtureCall(f"research_fixture_missing:{key}")
        return [dict(row) for row in rows]


class DeterministicEmbedder:
    model_name = "cognitive-replay-hash-v1"
    dim = 32

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        values = values - values.mean()
        norm = float(np.linalg.norm(values)) or 1.0
        return values / norm

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(text) for text in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


def simple_runtime_support(outcomes: DeterministicOutcomeQueue) -> SimpleNamespace:
    return SimpleNamespace(outcomes=outcomes)
