# Phase 6 Legacy Removal and Retention Report

No modules or authoritative rows were physically deleted. Phase 6 removed semantic ownership, not auditability.

## Deactivated behavior

- Conversation continuity shadow defaults OFF; v2 continuation defaults ON.
- All Phase 1–5 canonical domains default ON behind the master `HEBE_COGNITIVE_V2_ENABLED` kill switch.
- Legacy/vector/general-memory stores are classified as projections, caches, archives, or compatibility and cannot establish canonical v2 truth.

## Retained components

- **LegacyPendingCompatibilityAdapter** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **HebeState compatibility projection** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **wide runtime scene snapshots** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **LegacyMemoryFactAdapter** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **retrieval cache only** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **MemoryExtractor deprecated writer** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **archive/projection** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **archive/projection** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **legacy stream_schedule compatibility** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **GameDossier compatibility cache** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **compatibility projection** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **runtime projection** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **compatibility cache** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **legacy social compatibility archive** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **generic memory action claims** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **legacy CognitiveContextBuilder retained for non-v2 payload** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **broad summary-to-truth deprecated** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
- **domain TTL helpers retained as compatibility** — COMPATIBILITY; runtime or historical reader remains; canonical mutation: False.
