import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.command_result import CommandResult
from app.cognitive.response_synthesizer import ResponseSynthesizer


class FakeModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.text


class SequenceModel:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.texts:
            return self.texts.pop(0)
        return ""


class CommandResultSynthesisTests(unittest.TestCase):
    def test_generated_reply_is_used_when_valid(self):
        model = FakeModel("Listo, mi señor. Voz local encendida y el directo sigue en texto.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, "Listo, mi señor. Voz local encendida y el directo sigue en texto.")
        self.assertTrue(model.calls)

    def test_question_after_resolved_scope_uses_fallback(self):
        model = FakeModel("¿La quieres también para stream?")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, result.fallback_text)

    def test_stream_enabled_claim_for_local_scope_uses_fallback(self):
        model = FakeModel("Perfecto, también para el stream queda activado.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, result.fallback_text)


    def test_open_application_missing_path_rejects_remote_access_advice(self):
        model = FakeModel("Quieres que abra OBS o que te explique como hacerlo? Pasame acceso remoto.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="open_application",
            success=False,
            user_visible_summary="OBS Studio recognized but path missing.",
            state_changes={"app_id": "obs", "app_name": "OBS Studio", "error_code": "app_path_missing"},
            constraints=["Do not ask for remote access.", "Ask Leo to configure HEBE_APP_OBS_PATH."],
            fallback_text="Reconozco OBS Studio, pero no tengo configurada su ruta ejecutable. Configura HEBE_APP_OBS_PATH o la ruta en el registro de apps.",
            requires_model_response=True,
            metadata={"error_code": "app_path_missing", "message_goal": "Ask Leo to configure HEBE_APP_OBS_PATH."},
        )

        reply = synth.synthesize_command_result(result, input_text="hebe abre obs")

        self.assertTrue(model.calls)
        self.assertEqual(reply, result.fallback_text)

    def test_open_application_success_uses_valid_model_wording(self):
        model = FakeModel("Abriendo OBS Studio.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="open_application",
            success=True,
            user_visible_summary="Abriendo OBS Studio.",
            state_changes={"app_id": "obs", "app_name": "OBS Studio", "error_code": None},
            constraints=["Do not ask whether to open it."],
            fallback_text="Abriendo OBS Studio.",
            requires_model_response=True,
            metadata={"message_goal": "Confirm that OBS Studio is opening locally."},
        )

        reply = synth.synthesize_command_result(result, input_text="hebe abre obs")

        self.assertEqual(reply, "Abriendo OBS Studio.")

    def test_assistant_like_save_publish_offer_uses_fallback(self):
        model = FakeModel("Perfecto, tomo nota. Te lo guardo. Quieres que lo publique en stream?")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="stream_context_note",
            success=True,
            user_visible_summary="Context noted.",
            state_changes={"ok": True},
            fallback_text="Vale, queda hecho.",
            requires_model_response=True,
            metadata={"message_goal": "Confirm briefly."},
        )

        reply = synth.synthesize_command_result(result, input_text="ok")

        self.assertEqual(reply, result.fallback_text)

    def test_style_guard_blocks_creator_and_helper_phrases(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._guard_style("Tomo nota, tú mandas creador. ¿Quieres que lo guarde?", fallback="Corto y claro.")

        self.assertEqual(reply, "Corto y claro.")

    def test_style_guard_trims_bad_final_assistant_offer(self):
        synth = ResponseSynthesizer(conversation_model=None)
        bad = (
            "Me alegro, guapo. A ver si tanta confianza te sirve para algo util hoy. "
            "Que quieres que haga o comente ahora?"
        )
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            reply = synth._guard_style(bad)

        self.assertEqual(reply, "Me alegro, guapo. A ver si tanta confianza te sirve para algo util hoy.")
        self.assertNotIn("Eso son", reply)
        self.assertIn("action=trimmed", "\n".join(logs))

    def test_style_guard_regenerates_when_trim_cannot_save_reply(self):
        model = SequenceModel(["Me alegro, guapo. Hoy vienes subido, pero con gracia."])
        synth = ResponseSynthesizer(conversation_model=model)
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            reply = synth._guard_style("Quieres que haga algo ahora?", source_text="estoy contento")

        self.assertEqual(reply, "Me alegro, guapo. Hoy vienes subido, pero con gracia.")
        self.assertEqual(len(model.calls), 1)
        self.assertIn("action=regenerated", "\n".join(logs))
        self.assertNotIn("Eso sonÃ³ raro hasta para mÃ­, jefe", reply)

    def test_style_guard_minimal_stt_fallback_is_neutral_and_warns_on_repeat(self):
        synth = ResponseSynthesizer(conversation_model=None)
        ctx = SimpleNamespace(source="stt_voice", input_text="ruido")
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            first = synth._guard_style("Tomo nota.", context=ctx, allow_minimal_fallback=True)
            second = synth._guard_style("Tomo nota.", context=ctx, allow_minimal_fallback=True)

        self.assertEqual(first, "No te he entendido bien.")
        self.assertEqual(second, "No te he entendido bien.")
        joined = "\n".join(logs)
        self.assertIn("action=minimal_fallback", joined)
        self.assertIn("[HEBE][STYLE_GUARD][WARN] repeated_fallback_detected", joined)
        self.assertNotIn("Eso sonÃ³ raro hasta para mÃ­, jefe", first + second)

    def test_style_guard_does_not_use_hardcoded_personality_fallback(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._guard_style(
            "Tomo nota.",
            fallback="Eso sonÃ³ raro hasta para mÃ­, jefe.",
        )

        self.assertEqual(reply, "")

    def test_action_offer_phrase_requires_structured_action_permission(self):
        synth = ResponseSynthesizer(conversation_model=None)

        blocked = synth._guard_style("Lo publico en stream.")
        allowed = synth._guard_style("Lo publico en stream.", allow_action_offers=True)

        self.assertEqual(blocked, "")
        self.assertEqual(allowed, "Lo publico en stream.")

    def test_policy_boundary_regenerates_generic_refusal(self):
        model = SequenceModel([
            "No puedo proporcionar esa informacion en directo. Consulta recursos profesionales.",
            "Ese tema se queda fuera del directo. Cambiamos de carril antes de que arda el chat.",
        ])
        synth = ResponseSynthesizer(conversation_model=model)

        result = synth.synthesize_policy_boundary_response(
            policy={
                "policy_decision": "blocked",
                "reason": "sexual_topic_stream_mode",
                "intent": "viewer_unsafe_or_offbrand_request",
                "requested_behavior": "sexual_stream_topic",
                "response_intent": "hebe_playful_boundary",
                "must_not_include": ["explicit_instructions", "generic_ai_refusal"],
            },
            input_text="Hebe, como se usa un condon?",
            speaker="viewer",
            source="twitch_chat",
        )

        self.assertEqual(result["response_source"], "llm_persona_generated")
        self.assertTrue(result["style_guard_triggered"])
        self.assertTrue(result["was_generic_refusal_rewritten"])
        lowered = result["text"].casefold()
        self.assertNotIn("no puedo proporcionar", lowered)
        self.assertNotIn("consulta recursos", lowered)
        self.assertNotIn("como ia", lowered)

    def test_policy_boundary_without_model_is_marked_as_fallback_template(self):
        synth = ResponseSynthesizer(conversation_model=None)

        result = synth.synthesize_policy_boundary_response(
            policy={
                "policy_decision": "blocked",
                "reason": "protected_group_joke",
                "response_intent": "hebe_playful_boundary",
            },
            input_text="Hebe, haz un chiste sobre gitanos",
            speaker="viewer",
            source="twitch_chat",
        )

        self.assertEqual(result["response_source"], "fallback_template")
        self.assertFalse(result["style_guard_triggered"])
        self.assertTrue(result["text"])
        self.assertFalse(synth._generic_refusal_reason(result["text"]))

    def test_sexual_stream_topic_not_generic_refusal(self):
        model = SequenceModel([
            "No puedo dar instrucciones sobre condones aqui; consulta recursos fiables.",
            "Ese tema no entra en directo; siguiente curva, chat.",
        ])
        synth = ResponseSynthesizer(conversation_model=model)

        result = synth.synthesize_policy_boundary_response(
            policy={
                "policy_decision": "blocked",
                "reason": "sexual_topic_stream_mode",
                "requested_behavior": "sexual_stream_topic",
                "behavior_family": "stream_safety",
            },
            input_text="Hebe, como uso un condon?",
            speaker="viewer",
            source="twitch_chat",
        )

        lowered = result["text"].casefold()
        self.assertEqual(result["style_profile"], "sharp_stream_boundary")
        self.assertNotIn("no puedo", lowered)
        self.assertNotIn("recursos fiables", lowered)
        self.assertNotIn("consulta", lowered)

    def test_policy_boundary_no_proxy_profile(self):
        synth = ResponseSynthesizer(conversation_model=None)

        result = synth.synthesize_policy_boundary_response(
            policy={
                "policy_decision": "blocked",
                "reason": "viewer_repeat_to_leo_request",
                "requested_behavior": "message_to_leo",
                "behavior_family": "message_to_leo",
            },
            input_text="Hebe, avisa a Leo de que lea el chat",
            speaker="viewer",
            source="twitch_chat",
        )

        self.assertEqual(result["style_profile"], "no_proxy_boundary")
        self.assertEqual(result["blocked_behavior"], "message_to_leo")

    def test_style_guard_trims_default_followup_question_from_mood_reply(self):
        synth = ResponseSynthesizer(conversation_model=None)
        ctx = SimpleNamespace(message_type="small_talk", input_text="Estoy bastante contento.")
        reply = synth._guard_style(
            "Me alegro, jefe. Que dure la racha. Â¿Contento por quÃ©?",
            context=ctx,
        )

        self.assertEqual(reply, "Me alegro, jefe. Que dure la racha.")

    def test_style_guard_regenerates_single_question_into_statement(self):
        model = SequenceModel(["Me alegro, jefe. Que dure la racha."])
        synth = ResponseSynthesizer(conversation_model=model)
        ctx = SimpleNamespace(message_type="small_talk", input_text="Estoy bastante contento.")

        reply = synth._guard_style("Contento por que?", context=ctx)

        self.assertEqual(reply, "Me alegro, jefe. Que dure la racha.")
        self.assertEqual(len(model.calls), 1)

    def test_style_guard_allows_question_when_explicitly_requested(self):
        synth = ResponseSynthesizer(conversation_model=None)
        ctx = SimpleNamespace(message_type="clarification", input_text="Hebe, haz SO")

        reply = synth._guard_style("A quien le hago el SO?", context=ctx)

        self.assertEqual(reply, "A quien le hago el SO?")

    def test_style_guard_blocks_repeated_tu_que_tal_loop(self):
        model = SequenceModel(["AquÃ­ seguimos, Leo. Sin incendios nuevos."])
        synth = ResponseSynthesizer(conversation_model=model)
        ctx = SimpleNamespace(message_type="small_talk", input_text="Estoy bastante contento.")

        reply = synth._guard_style("Me alegro, jefe. Tu que tal?", context=ctx)

        self.assertEqual(reply, "Me alegro, jefe.")
        self.assertNotIn("que tal", reply.casefold())

    def test_style_guard_trims_customer_support_choice_from_tired_reply(self):
        synth = ResponseSynthesizer(conversation_model=None)
        ctx = SimpleNamespace(message_type="small_talk", input_text="Estoy cansado.")

        reply = synth._guard_style(
            "Normal. Hoy vienes con la barra de energia en rojo. Quieres que te recomiende descansar?",
            context=ctx,
        )

        self.assertEqual(reply, "Normal. Hoy vienes con la barra de energia en rojo.")


if __name__ == "__main__":
    unittest.main()
