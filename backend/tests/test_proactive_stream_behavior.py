import unittest

from app.stream.game_advice_gate import GameAdviceGate
from app.stream.proactive import StreamPreparationRoutine, cooldown_active, mark_cooldown, semantic_cooldown_key
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.stream.state import StreamSessionState


class ProactiveStreamBehaviorTests(unittest.TestCase):
    def test_persona_5_autopotion_blocked(self):
        result = GameAdviceGate().validate(
            current_game="Persona 5 Royal",
            proposed_advice="Conviene activar autopoción antes del jefe.",
        )

        self.assertFalse(result.allowed)
        self.assertIn("autopotion", result.blocked)
        self.assertEqual(result.reason, "mechanic_not_validated")

    def test_persona_5_valid_mechanic_allowed(self):
        result = GameAdviceGate().validate(
            current_game="Persona 5 Royal",
            proposed_advice="Mira el SP antes de avanzar y aprovecha una sala segura si toca respirar.",
        )

        self.assertTrue(result.allowed)
        self.assertIn("sp_management", result.validated)
        self.assertIn("safe_rooms", result.validated)

    def test_unknown_game_blocks_specific_mechanics_without_source(self):
        result = GameAdviceGate().validate(
            current_game="Mystery RPG",
            proposed_advice="Usa Baton Pass y fusiona una Persona mejor.",
        )

        self.assertFalse(result.allowed)
        self.assertTrue({"baton_pass", "persona_fusion"} & set(result.blocked))

    def test_stream_prep_is_actionable_when_obs_closed(self):
        stream = StreamSessionState(enabled=False, presence_mode="reactive")
        decision = StreamPreparationRoutine().evaluate(
            stream=stream,
            schedule_slot={"game": "Persona 5 Royal", "slot_name": "Persona Week", "category": "Persona 5 Royal"},
            obs_running=False,
            twitch_connected=True,
            chat_connected=True,
            stt_listening=True,
            tts_ready=True,
            game_run_state_ready=False,
            title_category_known=True,
        )

        self.assertEqual(decision.proactive_type, "actionable_routine")
        self.assertTrue(decision.should_speak)
        self.assertTrue(decision.action_available)
        self.assertIn("open_obs", decision.suggested_action)
        self.assertIn("enable_stream_mode", decision.suggested_action)

    def test_empty_activate_stream_mode_blocked_when_already_prepared(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        decision = StreamPreparationRoutine().evaluate(
            stream=stream,
            schedule_slot={"game": "Persona 5 Royal", "slot_name": "Persona Week", "category": "Persona 5 Royal"},
            obs_running=True,
            twitch_connected=True,
            chat_connected=True,
            stt_listening=True,
            tts_ready=True,
            game_run_state_ready=True,
            title_category_known=True,
        )

        self.assertFalse(decision.should_speak)
        self.assertEqual(decision.blocked_reason, "already_prepared")
        self.assertFalse(decision.action_available)

    def test_repetition_cooldown_blocks_second_healing_advice(self):
        stream = StreamSessionState()
        key = semantic_cooldown_key("Cuida el SP y la cura antes del jefe.", "resource_management")
        mark_cooldown(stream, key, now=1000.0, seconds=600)

        self.assertTrue(cooldown_active(stream, key, now=1001.0))

    def test_no_high_quality_anchor_skips_spontaneity(self):
        now = 2_000_000.0
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = now
        stream.last_chat_activity_ts = 0.0
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(show_silence_sec=1, show_jitter_sec=0),
            now_fn=lambda: now,
        )

        readiness = service.evaluate(stream, now=now)

        self.assertFalse(readiness["would_send"])
        self.assertIn(readiness["blocked_reason"], {"chat activity baseline not ready", "weak_anchor"})


if __name__ == "__main__":
    unittest.main()
