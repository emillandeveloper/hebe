from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.hebe_engine import HebeEngine
from app.stream.state import StreamSessionState
from tests.test_voice_command_pipeline import install_test_continuity, open_test_conversation


def decision(*capabilities: str, authority: str = "owner", source: str = "ui", live: bool = True, simulation: bool = False):
    grants = set(capabilities)
    return SimpleNamespace(
        authority=authority,
        source=source,
        should_stop_pipeline=False,
        allowed_step_types=["state_update", "action", "reply"],
        action_permission_summary={"stream_live": live, "is_simulation": simulation},
        allows_capability=lambda capability: capability in grants,
    )


def engine_with_state():
    engine = HebeEngine.__new__(HebeEngine)
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(
            hebe_sleeping=True,
            mode="sleep",
            tts_enabled=True,
            stream=StreamSessionState(),
        ),
        speak=Mock(),
    )
    install_test_continuity(engine)
    open_test_conversation(engine, kind="appointment_datetime")
    return engine


class ManualHandlerGuardTests(unittest.TestCase):
    def test_wake_requires_router_capability(self):
        engine = engine_with_state()
        blocked = engine._handle_wake_sleep_command(
            "Hebe, despierta", cognitive_decision=decision("hebe.chat_reply"), source="ui"
        )
        self.assertIsNone(blocked)
        self.assertTrue(engine.runtime.state.hebe_sleeping)

        allowed = engine._handle_wake_sleep_command(
            "Hebe, despierta", cognitive_decision=decision("hebe.wake_control"), source="ui"
        )
        self.assertIsNotNone(allowed)
        self.assertFalse(engine.runtime.state.hebe_sleeping)

    def test_ambient_tts_cannot_mutate_audio_state(self):
        engine = engine_with_state()
        result = engine._handle_tts_manual_command(
            "Hebe, desactiva tu voz",
            cognitive_decision=decision("audio.tts_control", authority="ambient", source="ambient_stt"),
            source="ambient_stt",
        )
        self.assertIsNone(result)
        self.assertTrue(engine.runtime.state.tts_enabled)

    def test_stream_manual_command_is_blocked_offline(self):
        engine = engine_with_state()
        result = engine._handle_stream_manual_command(
            "Hebe, activa STT ambiental",
            cognitive_decision=decision("stream.local_state_control", live=False),
            source="ui",
        )
        self.assertIsNone(result)
        self.assertFalse(engine.runtime.state.stream.enabled)

    def test_explicit_simulation_may_cross_offline_manual_guard(self):
        engine = engine_with_state()
        self.assertTrue(engine._manual_handler_guard(
            handler="stream", cognitive_decision=decision(
                "stream.local_state_control", live=False, simulation=True
            ), capabilities={"stream.local_state_control"}, source="ui", require_live=True,
        ))

    def test_time_and_personal_state_routes_cannot_touch_pending(self):
        for text in ("Hebe, qué hora es", "Hebe, tengo hambre"):
            with self.subTest(text=text):
                engine = engine_with_state()
                original = engine._active_current_conversation()
                result = engine._handle_pending_manual_intent(
                    text, cognitive_decision=decision("hebe.chat_reply"), source="ui"
                )
                self.assertIsNone(result)
                current = engine._active_current_conversation()
                self.assertEqual((current.id, current.version), (original.id, original.version))

    def test_ambient_cannot_cancel_pending(self):
        engine = engine_with_state()
        original = engine._active_current_conversation()
        result = engine._handle_pending_manual_intent(
            "cancela esa cita",
            cognitive_decision=decision("pending.cancel", authority="ambient", source="ambient_stt"),
            source="ambient_stt",
        )
        self.assertIsNone(result)
        current = engine._active_current_conversation()
        self.assertEqual((current.id, current.version), (original.id, original.version))


if __name__ == "__main__":
    unittest.main()
