# Cognitive Replay Verification Report

Overall status: **FAILED**

## Repository and environment

- Commit: `9f91948003ec545fbf7ae045cf78b3affb3451d7`
- Working tree: `9ac275d1cae2`
- Platform: `Windows-10-10.0.26200-SP0`
- Python: `3.11.0`

## Commands

- `python -m app.replay --scenario ivanxi_resub_promo_restart --output artifacts/cognitive-replay/dev-a` → exit 1 (17.205645s)

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

### ivanxi_resub_promo_restart

- Status: **FAILED**
- Events: 7
- Restarts: 1
- Duration: 17.028403s
- Assertions passed/failed/skipped: 3/2/0

- Failure: event `final`, path `receipts`, reason `assertion_mismatch`
- Failure: event `final`, path `promotion_profiles`, reason `assertion_mismatch`
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
    "C:\\Users\\Leo Nifelheim\\Documents\\Hebe\\hebe-ui\\artifacts\\cognitive-replay\\dev-a\\workspaces\\ivanxi_resub_promo_restart\\hebe-replay.sqlite3"
  ],
  "restart_points": 1,
  "schema_migrations": [
    [
      {
        "component": "cognitive_replay",
        "version": 1,
        "name": "replay_metadata",
        "checksum": "847e14d94cbcb5ad15dac895e06910750bdfd4fd44b001f5fff35688fffa5eba",
        "applied_at": "2026-08-11T06:40:55.083853+00:00",
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
