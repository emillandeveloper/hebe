# Cognitive Replay & Verification Harness

Phase 0.5 provides an accelerated, deterministic, offline harness for Cognitive Continuity work. It does not implement Conversation Continuity, beliefs, durable game runs, SocialWorld, consolidation, or other Phase 1+ behavior.

Unlike the older Stream Replay Lab, Cognitive Replay constructs a real `HebeEngine`, uses the production policy/router/domain transaction/persistence/emission machinery, and replaces only external adapters. Owner and ambient transcripts enter the same post-transcription methods as live audio. Normalized Twitch chat, lifecycle, and metadata use shared engine ingress methods called by production adapters too. Social replay events use production-shaped normalized internal events.

## Running it

From the repository root with the backend virtual environment and `PYTHONPATH=backend`:

```powershell
backend/.venv/Scripts/python.exe -m app.replay --scenario ivanxi_resub_promo_restart --output artifacts/cognitive-replay/ivanxi
backend/.venv/Scripts/python.exe -m app.replay --scenario backend/tests/fixtures/cognitive_replay --output artifacts/cognitive-replay/directory
backend/.venv/Scripts/python.exe -m app.replay --suite cognitive-v2 --run-phase-tests --output artifacts/cognitive-replay/phase-0.5
```

The command returns 1 for failure, 2 for incomplete required verification, and 0 when all applicable layers are verified. Declared non-blocking boundary limitations remain visible without mechanically failing the phase. Omitting `--run-phase-tests` intentionally makes the overall report incomplete because unit, integration, and regression layers were not attached.

For a closure/provenance run, `--baseline-differential <json>` attaches an exact baseline comparison. Inherited failures remain visible but only `NEW_PHASE_0_5_REGRESSION` or `FAILURE_CHANGED` fails the phase. The CLI rejects a differential whose current failure count does not match the regression run.

## Scenario format

Scenarios are JSON using schema version 1. They define a stable ID, initial time, seed, optional copied DB fixture, feature flags, semantic model/research fixtures, queued external outcomes, ordered events, checkpoint assertions, and final assertions. Relative event times are offsets from the scenario start; absolute ISO timestamps are supported.

Supported events are:

- `owner_stt`, `ambient_stt`
- `twitch_chat`, including identity, message, reply, mention, and normalization data
- `twitch_follow`, `twitch_sub`, `twitch_resub`, `twitch_raid`, `twitch_cheer`
- `stream_started`, `stream_ended`, `stream_metadata_changed`
- `advance_time`, `maintenance`, `restart_hebe`
- `configure_external_outcome`
- `game_research`

Assertions support equality, existence/absence, collection count, contains/no-match, exactly once, and zero external calls. Assertions tagged `future_phase` are recorded as skipped and make the scenario incomplete rather than pretending later-phase behavior exists.

## Time, workspaces, and restart

`ScenarioClock` controls `time.time`, `time.time_ns`, monotonic time, event scheduling, and the injected stream-context clock while an event runs. `advance_time` does not sleep; it advances the clock and invokes current pending, scheduler, and presence maintenance seams. Live owner audio, live-classified ambient audio, the ambient simulator, and replay all enter `ingest_normalized_stt` before classification, firewall, evidence, and routing.

Every scenario owns a fresh isolated SQLite database, or copies a supplied fixture before normal production schema initialization and replay migrations. Retained workspaces are cleared before a rerun. `restart_hebe` stops and releases the engine, forces collection evidence, constructs a new engine/runtime and repositories, and continues with the same SQLite file. Python runtime state is not copied.

## Deterministic boundaries

Twitch, speech/audio, desktop automation, model calls, embedding, and research are fakes or fixtures. They record attempts and return configured production-shaped outcomes without real external I/O. Model and research fixtures are semantic mappings; an unknown request fails closed. A wildcard fixture is explicit, never a network fallback. Seeds and UUID generation are deterministic.

The Twitch promotion fake sits behind the current production promotion transaction. It does not write receipts itself: production code interprets its result and persists the receipt. Success, failure, and timeout-shaped failures can therefore be tested as action truth.

## State and reports

`CognitiveStateProbe` is read-only. It exposes runtime policy state, stream/session metadata, pending state, current scene/game state, social observations, promotion profiles, external attempts, persisted receipts, minimal redacted emissions, DB counts, and migration records. Empty `open_threads` and `beliefs` sections reserve the verification API for later phases without implementing them.

Each run creates `verification-report.json` and a Markdown rendering derived from the same sanitized data. Reports include repository/environment identity, commands, test counts, scenario checkpoints/final state, action and persistence evidence, external-boundary classification, restart evidence, and limitations. Raw unrestricted transcripts and secret-shaped fields are redacted or represented by digests.

## Adding a scenario

Add a JSON file under `backend/tests/fixtures/cognitive_replay`, provide semantic fixtures for every expected model/research call, use synthetic identities, and express behavior through events and assertions. Do not add scenario-specific branches to production or replay code. If a required event cannot reach the same normalized production seam as a live event, first extract a shared ingress and route both adapters through it.

## Verification boundary

The harness does not test faster-whisper transcription quality, real Twitch/EventSub transport, real network reliability, audio-device output, or desktop application behavior. Some legacy `datetime.now()` persistence timestamps remain wall-clock based and are disclosed in reports. Exact raw natural-language output should only be asserted for guarded deterministic templates.

Human evaluation remains required for naturalness, personality, comedic timing, and whether Hebe feels socially appropriate. The harness verifies the cognitive and state prerequisites for those qualities; it does not objectively score them.
