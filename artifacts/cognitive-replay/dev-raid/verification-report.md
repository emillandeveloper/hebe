# Cognitive Replay Verification Report

Overall status: **FAILED**

## Repository and environment

- Commit: `9f91948003ec545fbf7ae045cf78b3affb3451d7`
- Working tree: `34335982b46d`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `python -m app.replay --scenario raid_transition_foundation --output artifacts/cognitive-replay/dev-raid` → exit 1 (6.179646s)

## Tests

```json
{
  "replay": {
    "passed": 0,
    "failed": 1,
    "skipped": 0
  },
  "required_layer_missing": false
}
```

## Replay scenarios

### raid_transition_foundation

- Status: **FAILED**
- Events: 3
- Restarts: 0
- Duration: 6.091764s
- Assertions passed/failed/skipped: 1/1/0

- Failure: event `final`, path `actions.attempts`, reason `assertion_mismatch`
## External boundaries

- twitch: `fake`
- tts_audio: `fake`
- desktop: `fake`
- game_research_web: `fixture`
- llm_model: `fixture`
- network: `blocked_by_design`

## Persistence

```json
{
  "database_type": "isolated_sqlite",
  "database_paths": [
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\dev-raid\\workspaces\\raid_transition_foundation\\hebe-replay.sqlite3"
  ],
  "restart_points": 0,
  "schema_migrations": [
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T06:46:20.868773+00:00",
        "already_applied": false
      }
    ]
  ]
}
```

## Limitations

- datetime.now() reads in legacy persistence remain wall-clock based; behavioral TTL/cooldown time.time() reads are controlled during replay dispatch
- faster-whisper audio decoding is outside the cognitive replay boundary and requires its separate integration suite

## Human evaluation boundary

This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.
