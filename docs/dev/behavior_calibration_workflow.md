# Behavior calibration workflow

Fase 2E is shadow-only. Telemetry and reviewer labels must never feed
`BehaviorAdaptationService`, alter a score, or create a constraint.

## Inspect

Use `GET /debug/behavior-calibration` while the DEV backend is running. The
response contains bounded in-memory metrics, recent rotating JSONL events,
current-stream constraints, durable constraints, recent retired constraints,
and the projected episodic fatigue state. It does not expose raw feedback or
raw STT.

Follow one decision using `trace_id`. A proactive trace normally contains:

1. `candidate_policy`
2. `candidate_ranking`
3. `post_generation`
4. `emission`

Feedback uses its existing input `event_id` as `trace_id` and records the
interpretation outcome, referent provenance, semantic motif identity, and
effect without the original transcript.

## Label

Submit one manual QA label with:

`POST /debug/behavior-calibration/{trace_id}/label`

Body:

```json
{"label": "FALSE_POSITIVE"}
```

Allowed values are `CORRECT`, `FALSE_POSITIVE`, `FALSE_NEGATIVE`, and
`UNCERTAIN`. Labels are calibration evidence only. They are stored in their
own bounded rotating JSONL and have no runtime policy consumer.

## Curate a replay

Do not convert logs automatically. Select an interesting labeled trace,
review the surrounding stream context, and manually provide the minimal
candidate wording needed to reproduce it. Use `BehaviorTraceReplayCurator`
to create a `CuratedBehaviorReplayCase`, then review its fixture row before
adding it to a replay test.

The curated fixture may retain the source `trace_id`, motif terms, topic,
expected decision, and reviewer label. Remove names, raw STT, unrelated chat,
and any other personal data. A policy or threshold change requires a reviewed
replay demonstrating the problem; telemetry volume alone is not evidence of
incorrect behavior.
