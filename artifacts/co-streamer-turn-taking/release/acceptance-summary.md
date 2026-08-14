# Co-Streamer Turn-Taking Verification Summary

Overall: **CO-STREAMER TURN-TAKING VERIFIED**

## Before / after

| Metric | Before real stream | Deterministic replay |
|---|---:|---:|
| Companion ticks | 486 | 4 |
| Valid emissions | 0 | 1 |
| Intents created | unavailable | 3 |
| Intents emitted | 0 | 1 |

## Performance

- Intent creation p50/p95: 0.057 / 0.057 ms
- Pending queue operation p50/p95: 0.057 / 0.057 ms
- Turn arbitration p50/p95: 0.02 / 0.024 ms
- Presence + turn decision p50/p95: 1.799 / 2.037 ms
- Created-to-emitted p50: 2100.0 ms
- Conversational gap before emission p50: 1300.0 ms

## Regression

- Tests: 577/594 passed; 17 inherited failures; 0 new regressions.
- Scenarios A–L: PASSED.
- Representative replay: VERIFIED (10 assertions).

## Known limitations

- Naturalness, comedic timing, and social appropriateness still require human stream review.
- Replay replaces Twitch, TTS audio, and model/network boundaries with deterministic fakes.
- Owner voice-active uses the production RMS/VAD sample plus normalized STT utterance completion; it is not sample-accurate diarization.
- Hebe yields before TTS commit; already-playing audio is not forcibly cancelled by this change.
- The full historical suite retains 17 baseline failures; the differential confirms none were introduced by this fix.
