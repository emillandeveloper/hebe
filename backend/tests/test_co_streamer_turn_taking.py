from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.stream.companion_loop import StreamCompanionLoop
from app.stream.speech_intents import SpeechIntentManager, SpeechIntentStatus, SpeechIntentType
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.stream.state import StreamSessionState
from app.hebe_engine import HebeEngine


class CoStreamerTurnTakingScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = [1_000_000.0]
        self.voice_active = [False]

    def stream(self, *, game: str | None = "Test RPG") -> StreamSessionState:
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = True
        stream.live_status_known = True
        stream.current_game = game
        stream.stream_context_updated_ts = self.clock[0]
        stream.last_chat_activity_ts = self.clock[0] - 3600
        stream.policies.allow_tts_idle_prompts = True
        stream.current_scene_timeline = {"scene_id": "boss-a", "topic_id": "fight-a"}
        return stream

    def loop(self) -> StreamCompanionLoop:
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                companion_silence_sec=60, companion_jitter_sec=0,
                global_stream_cooldown_sec=60, recent_voice_quiet_sec=20,
                startup_grace_sec=0, companion_max_per_hour=3,
            ),
            now_fn=lambda: self.clock[0],
        )
        return StreamCompanionLoop(
            spontaneity=service,
            now_fn=lambda: self.clock[0],
            owner_voice_active_fn=lambda: self.voice_active[0],
        )

    def fact(self, kind: str = "rng_dependency", *, fact_id: str = "rng-1", role: str = "owner_commentary", confidence: float = 0.86) -> dict:
        return {
            "id": fact_id, "kind": kind, "category": kind,
            "text": "Que ha sido demasiada suerte, Dios.",
            "raw_text": "Que ha sido demasiada suerte, Dios.",
            "confidence": confidence, "timestamp": self.clock[0],
            "expires_at": self.clock[0] + 60, "ttl_sec": 60,
            "utterance_role": role, "proactive_eligible": role == "owner_commentary",
            "scene_id": "boss-a", "topic_id": "fight-a",
        }

    def evaluate(self, loop: StreamCompanionLoop, stream: StreamSessionState):
        with patch("app.stream.companion_loop.log_proactive_decision", lambda _decision: None):
            return loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

    def owner_ends(self, stream: StreamSessionState, *, advance: float = 0.0) -> None:
        self.clock[0] += advance
        self.voice_active[0] = False
        stream.last_voice_event_ts = self.clock[0]
        stream.last_owner_utterance_end_ts = self.clock[0]

    def test_scenario_a_direct_owner_reaction_uses_short_turn_gap(self):
        stream, loop = self.stream(), self.loop()
        stream.recent_run_context_facts = [self.fact()]
        self.owner_ends(stream)
        self.clock[0] += 1.3
        tick = self.evaluate(loop, stream)
        self.assertTrue(tick.should_speak)
        self.assertEqual(tick.speech_intent["type"], "REACTION")
        self.assertNotEqual(tick.blocked_reason, "recent_owner_speech")

    def test_scenario_b_owner_resume_yields_then_fresh_intent_can_emit(self):
        stream, loop = self.stream(), self.loop()
        stream.recent_run_context_facts = [self.fact()]
        self.owner_ends(stream)
        self.clock[0] += 1.3
        first = self.evaluate(loop, stream)
        self.assertTrue(first.should_speak)
        self.voice_active[0] = True
        self.clock[0] += 0.2
        yielded = self.evaluate(loop, stream)
        self.assertFalse(yielded.should_speak)
        self.assertGreaterEqual(loop.intent_manager.metrics["yield_due_owner_resume"], 1)
        self.owner_ends(stream, advance=0.5)
        self.clock[0] += 1.3
        retried = self.evaluate(loop, stream)
        self.assertTrue(retried.should_speak)
        self.assertEqual(first.speech_intent["id"], retried.speech_intent["id"])

    def test_scenario_c_reaction_expires_instead_of_emitting_late(self):
        stream, loop = self.stream(), self.loop()
        stream.recent_run_context_facts = [self.fact()]
        self.voice_active[0] = True
        self.evaluate(loop, stream)
        self.clock[0] += 9.0
        stream.current_scene_timeline = {"scene_id": "menu", "topic_id": "menu"}
        stream.current_game = None
        stream.recent_run_context_facts = []
        self.voice_active[0] = False
        stream.last_owner_utterance_end_ts = self.clock[0]
        tick = self.evaluate(loop, stream)
        self.assertFalse(tick.should_speak)
        self.assertEqual(loop.intent_manager.metrics["intents_expired"], 1)

    def test_scenario_d_idle_chatter_remains_conservative(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.speech_intent_candidates = [{"type": "IDLE_CHATTER", "topic": "idle", "value": 0.7}]
        self.owner_ends(stream)
        self.clock[0] += 10.0
        tick = self.evaluate(loop, stream)
        self.assertFalse(tick.should_speak)
        self.assertEqual(tick.blocked_reason, "turn_gap_too_short")

    def test_scenario_e_contextual_opinion_can_take_turn(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.speech_intent_candidates = [{"type": "OPINION", "topic": "boss-design", "value": 0.82, "material": {"basis": "hebe_self"}}]
        self.owner_ends(stream)
        self.clock[0] += 2.6
        tick = self.evaluate(loop, stream)
        self.assertTrue(tick.should_speak)
        self.assertEqual(tick.speech_intent["type"], "OPINION")

    def test_scenario_f_self_initiated_topic_needs_no_direct_address(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.speech_intent_candidates = [{"type": "SELF_INITIATED_TOPIC", "topic": "earlier-thought", "value": 0.78, "material": {"basis": "open_thread"}}]
        self.owner_ends(stream)
        pending = self.evaluate(loop, stream)
        self.assertFalse(pending.should_speak)
        self.clock[0] += 20.1
        tick = self.evaluate(loop, stream)
        self.assertTrue(tick.should_speak)
        self.assertEqual(tick.speech_intent["type"], "SELF_INITIATED_TOPIC")

    def test_scenario_g_overlapping_intents_are_coalesced(self):
        manager = SpeechIntentManager(now_fn=lambda: self.clock[0])
        for value in (0.7, 0.74, 0.9):
            manager.create(intent_type="REACTION", topic="same-fight", value=value, anchor_ids=[f"a-{value}"])
        self.assertEqual(len(manager.pending()), 1)
        self.assertEqual(manager.pending()[0].value, 0.9)
        self.assertEqual(manager.metrics["intents_superseded"], 2)

    def test_scenario_h_scene_change_invalidates_pending_intent(self):
        manager = SpeechIntentManager(now_fn=lambda: self.clock[0])
        manager.create(intent_type="GAME_COMMENT", topic="boss", value=0.8, scene_relevance={"scene_id": "boss-a"})
        manager.expire(current_scene={"scene_id": "menu"})
        self.assertFalse(manager.pending())
        self.assertEqual(manager.snapshot()["all"][-1]["suppression_reason"], "stale_scene")

    def test_scenario_i_chat_activity_does_not_veto_response_to_chat(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.recent_chat_messages = [{"timestamp": self.clock[0], "username": "viewer", "text": "Hebe, mira esto"}] * 3
        stream.last_chat_activity_ts = self.clock[0]
        stream.speech_intent_candidates = [{"type": "SOCIAL_FOLLOWUP", "topic": "viewer-direct", "value": 0.86, "material": {"directed_to_hebe": True}}]
        self.owner_ends(stream)
        self.clock[0] += 1.6
        tick = self.evaluate(loop, stream)
        self.assertTrue(tick.should_speak)
        self.assertEqual(tick.speech_intent["type"], "SOCIAL_FOLLOWUP")

    def test_scenario_j_read_dialogue_does_not_flood_intents(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.recent_run_context_facts = [
            self.fact("dialogue", fact_id=f"line-{index}", role="quoted_or_read_dialogue")
            for index in range(3)
        ]
        self.owner_ends(stream)
        self.clock[0] += 4.0
        tick = self.evaluate(loop, stream)
        self.assertFalse(tick.should_speak)
        self.assertEqual(loop.intent_manager.metrics["intents_created"], 0)

    def test_scenario_k_presence_cooldown_still_controls_frequency(self):
        stream, loop = self.stream(), self.loop()
        stream.recent_run_context_facts = [self.fact()]
        self.owner_ends(stream)
        self.clock[0] += 1.3
        first = self.evaluate(loop, stream)
        self.assertTrue(first.should_speak)
        loop.record_emitted(stream, "reaction", intent_id=first.speech_intent["id"])
        stream.last_hebe_stream_speak_ts = self.clock[0]
        stream.recent_run_context_facts = [self.fact("enemy_mechanic", fact_id="mechanic-2")]
        self.clock[0] += 3.0
        second = self.evaluate(loop, stream)
        self.assertFalse(second.should_speak)
        self.assertEqual(second.blocked_reason, "recent_hebe_message")

    def test_scenario_l_frequent_streamer_cadence_is_not_permanent_silence(self):
        stream, loop = self.stream(), self.loop()
        emissions = 0
        for index, kind in enumerate(("enemy_mechanic", "failure_or_death", "rng_dependency")):
            self.voice_active[0] = True
            stream.recent_run_context_facts = [self.fact(kind, fact_id=f"frequent-{index}")]
            self.evaluate(loop, stream)
            self.owner_ends(stream, advance=0.7)
            self.clock[0] += 1.3
            tick = self.evaluate(loop, stream)
            if tick.should_speak:
                emissions += 1
                loop.record_emitted(stream, "contextual contribution", intent_id=tick.speech_intent["id"])
                break
        self.assertGreaterEqual(emissions, 1)

    def test_cognitive_discourse_material_projects_without_a_model_call(self):
        stream, loop = self.stream(game=None), self.loop()
        stream.proposed_discourse_contribution = {
            "should_contribute": True, "topic_id": "topic-1",
            "contribution_value": 0.8, "grounded_fragments": ["Leo prefers the risky route"],
            "proposed_claims": ["contrast risk and consistency"],
        }
        engine = SimpleNamespace(
            social_world=SimpleNamespace(last_opportunities=[]),
            hebe_self_model=SimpleNamespace(current=lambda: []),
        )
        HebeEngine._queue_cognitive_speech_intent_candidates(engine, stream, loop=loop, now=self.clock[0])
        self.assertEqual(stream.speech_intent_candidates[0]["type"], "OPINION")
        self.assertEqual(stream.speech_intent_candidates[0]["source_event_ids"], ["discourse:topic-1"])

    def test_tts_commit_is_distinct_from_turn_reservation(self):
        manager = SpeechIntentManager(now_fn=lambda: self.clock[0])
        intent = manager.create(intent_type="REACTION", topic="rng", value=0.9)
        selected = manager.arbitrate(
            owner_voice_active=False, owner_utterance_ended_at=self.clock[0] - 2,
            tts_active=False, now=self.clock[0],
        ).intent
        self.assertEqual(selected.status, SpeechIntentStatus.TURN_RESERVED)
        manager.mark_tts_committed(intent.id)
        self.assertEqual(intent.status, SpeechIntentStatus.TTS_COMMITTED)


if __name__ == "__main__":
    unittest.main()
