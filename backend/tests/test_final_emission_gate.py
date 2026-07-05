import unittest

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute


class FinalEmissionGateTests(unittest.TestCase):
    def make_gate(self):
        calls = {"ui": [], "debug": [], "twitch": [], "tts": [], "logs": []}
        gate = FinalEmissionGate()
        return gate, calls

    def emit(self, gate, calls, **kwargs):
        return gate.emit(
            emit_ui=lambda payload: calls["ui"].append(payload),
            emit_debug=lambda payload: calls["debug"].append(payload),
            send_twitch=lambda text: calls["twitch"].append(text),
            speak=lambda text: calls["tts"].append(text),
            logger=lambda line: calls["logs"].append(line),
            **kwargs,
        )

    def test_candidate_not_broadcast_before_guards(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-1",
            source="twitch",
            final_response="candidate",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": False},
            debug_payload={"candidate_response": "candidate"},
        )

        self.assertEqual(calls["twitch"], [])
        self.assertEqual(calls["ui"], [])
        self.assertEqual(calls["debug"][0]["candidate_response"], "candidate")

    def test_candidate_response_never_ui_broadcast_even_if_guard_marked_passed(self):
        gate, calls = self.make_gate()

        result = self.emit(
            gate,
            calls,
            event_id="evt-stage-candidate",
            source="ui",
            final_response="candidate",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
            debug_payload={"response_stage": "candidate"},
        )

        self.assertFalse(result.emitted)
        self.assertEqual(calls["ui"], [])
        self.assertTrue(calls["debug"][0]["blocked_candidate_ui"])

    def test_failed_guard_response_never_ui_broadcast_by_stage(self):
        gate, calls = self.make_gate()

        result = self.emit(
            gate,
            calls,
            event_id="evt-stage-failed",
            source="ui",
            final_response="failed",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
            debug_payload={"response_stage": "failed_guard"},
        )

        self.assertFalse(result.emitted)
        self.assertEqual(calls["ui"], [])

    def test_repair_attempt_never_ui_broadcast_by_stage(self):
        gate, calls = self.make_gate()

        result = self.emit(
            gate,
            calls,
            event_id="evt-stage-repair",
            source="ui",
            final_response="repair",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
            debug_payload={"response_stage": "repair_attempt"},
        )

        self.assertFalse(result.emitted)
        self.assertEqual(calls["ui"], [])

    def test_failed_guard_response_not_broadcast(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-2",
            source="ui",
            final_response="failed",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": False},
        )

        self.assertEqual(calls["ui"], [])
        self.assertEqual(calls["debug"][0]["failed_guard_response"], "failed")

    def test_repair_attempt_not_broadcast(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-3",
            source="twitch",
            final_response="final",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
            repair_summary={"attempts": [{"cleaned": "repair"}]},
            debug_payload={"repair_attempts": [{"cleaned": "repair"}]},
        )

        self.assertEqual(calls["twitch"], ["final"])
        self.assertNotIn("repair", calls["twitch"])

    def test_one_input_one_final_output(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-4",
            source="ui",
            final_response="once",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
        )
        self.emit(
            gate,
            calls,
            event_id="evt-4",
            source="ui",
            final_response="again",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
        )

        self.assertEqual([payload["text"] for payload in calls["ui"]], ["once"])

    def test_suppress_route_no_public_output(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-5",
            source="twitch",
            final_response="blocked",
            output_route=OutputRoute.SUPPRESS,
            output_targets=["twitch_chat", "stream_tts", "local_ui"],
        )

        self.assertEqual(calls["twitch"], [])
        self.assertEqual(calls["tts"], [])
        self.assertEqual(calls["ui"], [])

    def test_observe_only_no_model_output(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-6",
            source="twitch",
            final_response="",
            output_route=OutputRoute.OBSERVE_ONLY,
            output_targets=[],
        )

        self.assertEqual(calls["twitch"], [])
        self.assertEqual(calls["tts"], [])
        self.assertEqual(calls["ui"], [])

    def test_local_ui_debug_only_not_public(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-7",
            source="simulation",
            final_response="debug",
            output_route=OutputRoute.LOCAL_UI_DEBUG_ONLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
        )

        self.assertEqual(calls["ui"], [])
        self.assertEqual(calls["twitch"], [])
        self.assertEqual(calls["tts"], [])
        self.assertEqual(calls["debug"][0]["final_response"], "debug")

    def test_twitch_text_reply_only_sends_twitch(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-8",
            source="twitch",
            final_response="chat",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
        )

        self.assertEqual(calls["twitch"], ["chat"])
        self.assertEqual(calls["tts"], [])
        self.assertEqual(calls["ui"], [])

    def test_stream_tts_reply_speaks_once(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-9",
            source="twitch",
            final_response="speak",
            output_route=OutputRoute.STREAM_TTS_REPLY,
            output_targets=["stream_tts"],
            guard_result={"passed": True},
        )

        self.assertEqual(calls["tts"], ["speak"])

    def test_same_event_deduped(self):
        gate, calls = self.make_gate()

        first = self.emit(
            gate,
            calls,
            event_id="evt-10",
            source="twitch",
            final_response="one",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
        )
        second = self.emit(
            gate,
            calls,
            event_id="evt-10",
            source="twitch",
            final_response="two",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
        )

        self.assertTrue(first.emitted)
        self.assertTrue(second.deduped)
        self.assertEqual(calls["twitch"], ["one"])

    def test_action_confirmation_requires_final_gate(self):
        gate, calls = self.make_gate()

        result = self.emit(
            gate,
            calls,
            event_id="evt-11",
            source="action",
            final_response="done",
            output_route=OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=["local_ui"],
            guard_result={"passed": True},
            execution_result={"action": "ok"},
        )

        self.assertTrue(result.emitted)
        self.assertEqual(calls["ui"][0]["execution_result"], {"action": "ok"})

    def test_bypass_audit_logs(self):
        gate, calls = self.make_gate()

        self.emit(
            gate,
            calls,
            event_id="evt-12",
            source="action",
            final_response="",
            output_route=OutputRoute.TWITCH_ACTION_ONLY,
            output_targets=[],
        )

        self.assertTrue(any("[HEBE][FINAL_EMISSION_GATE] suppressed=true" in line for line in calls["logs"]))


if __name__ == "__main__":
    unittest.main()
