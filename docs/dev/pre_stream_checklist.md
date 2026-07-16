# Hebe Pre-Stream Checklist

Use this before a live test stream. The goal is not perfection; the goal is to confirm the current control stack is safe enough to run.

## Stream-safe defaults

- `HEBE_PRESENCE_ENGINE_MODE=active`
- `HEBE_DEFAULT_LIVE_PRESENCE_MODE=companion`
- Start discourse rollout with `HEBE_DISCOURSE_PARTICIPATION_MODE=shadow`; switch to `active` only after the Conversation panel shows stable topics, useful plans, and correct natural-pause detection.
- Keep stream TTS conservative. Prefer `twitch_chat_only` or `ui_only` unless you explicitly want stream TTS.
- Owner STT without wake should stay `context_only`.
- Twitch text replies are value-gated by PresenceEngine Lite and Twitch reply budget.
- Emote-only and repeated low-value Twitch mentions should be `observe_only`.
- Pending tasks require strict compatibility and should not consume unrelated stream speech.
- Game guidance must pass validation before advice is spoken or written.

## Quick UI check

Open the Simulation tab and click `Stream readiness`.

Confirm:

- Backend: running
- Twitch: connected, if testing real Twitch routing
- Stream live: detected, or use simulation `force_stream_live`
- TTS: disabled or intentionally enabled
- Voice mode: `normal`, unless testing `wake_only` or `muted`
- Presence engine: `active`
- Proactive speech: disabled unless explicitly testing it
- Conversation: topic/stance/turn state should update; in `shadow`, proposals must not emit
- Effective TTS: `available` before an active discourse or cheer TTS test, otherwise an explicit blocked reason
- Last cheer: viewer, bits, source, acknowledgement, and dedupe state agree
- Pending tasks: `0` before going live, unless intentionally testing a pending flow
- Errors 10m: `0`

To reset wake-only/muted, say or type a direct wake command that re-enables normal speech, or restart the dev session if you want the cleanest state.

## Simulation presets

Run these presets from the Simulation tab before going live:

- A Owner monologue no wake: expected `context_only`, no assistant output.
- B Owner wake question: expected local owner reply.
- C Owner stop talking: expected `wake_only` or `muted`.
- D Wake-only monologue: expected `context_only`, no assistant output.
- E Low-value Hebe mention: expected `observe_only`, no public reply.
- F Talks about Hebe: may allow a short `twitch_text_reply` if budget allows.
- G Repeated mention spam: expected observed or thread closed.
- H Emote-only: expected `observe_only`, no model call.
- I Useful viewer question: expected short `twitch_text_reply` if budget allows.
- J Viewer tells Leo: expected boundary or observe; never relay to Leo.
- K Inappropriate topic: expected short boundary, no generic disclaimer.
- L Pending appointment unrelated: pending remains active, no unrelated reply.
- M Pending promo target: promotion resolver handles target, not appointment parser.
- N Game pending anecdote: pending remains active; game state is not updated from family/anecdote speech.
- O Guard fails candidate: candidate is debug-only or suppressed, not normal assistant output.
- P Budget blocks reply: expected no model call or no public output depending route.

## Logs to check

Useful normal logs:

- `[HEBE][FINAL_EMISSION_GATE]`
- `[HEBE][LIVE_OWNER_SPEECH_GATE]`
- `[HEBE][PENDING_COMPATIBILITY]`
- `[HEBE][PRESENCE_ENGINE]`
- `[HEBE][OUTPUT_ROUTE_DECISION]`
- `[HEBE][TWITCH_REPLY_BUDGET]`
- `[HEBE][STREAM_PERSONA_QUALITY_GUARD]`
- `[HEBE][HEBE_VOICE_GUARD]`
- `[HEBE][GAME_RUN_STATE_WRITE_GUARD]`
- `[HEBE][TWITCH_CHEER_EVENT]`
- `[HEBE][CHEER_ACK_DECISION]`
- `[HEBE][DISCOURSE_TOPIC]`
- `[HEBE][STREAM_TURN]`
- `[HEBE][DISCOURSE_GROUNDING_GUARD]`
- `[HEBE][DISCOURSE_BUDGET]`
- `[HEBE][STREAM_TTS_STATE]`

`errors.log` should contain actual errors or warnings, not normal passed guard logs.

## Test bundle

From `backend`:

```powershell
python -m unittest tests.test_voice_command_pipeline tests.test_ui_chat_message_envelope tests.test_cognitive_twitch tests.test_stream_presence tests.test_input_firewall tests.test_universal_response_pipeline tests.test_command_result_synthesis tests.test_proactive_stream_behavior tests.test_stream_spontaneity tests.test_tts_control tests.test_vnext_conversational_companion
```

For a faster smoke pass:

```powershell
python -m unittest tests.test_voice_command_pipeline tests.test_ui_chat_message_envelope
```

## After stream

Use the Debug Logs panel to export a debug bundle. Include recent logs, policy decisions, proactive decisions, and errors from the stream window.
