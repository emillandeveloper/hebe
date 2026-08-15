# Phase 1C — Game state canonicalization checkpoint

Status: implemented and verified; intentionally not committed. Phase 1D Social
has not started. Phase 1B is committed locally as `4bf56b6` and has not been
pushed.

## 1. Inventory of Game models and tables

| Representation | Semantic role | Writer / reader | Lifecycle and persistence | Relation to `game_run_id` | Duplicate status |
|---|---|---|---|---|---|
| `game_identities` | Stable game identity and aliases | `GameV2Repository`; Game services | Durable SQLite identity | Parent of runs | Canonical |
| `game_runs` | Run identity and lifecycle | `GameRunService` via `GameV2Repository` | Durable SQLite, active/paused/finished | The canonical id | Canonical |
| `game_run_sessions` | Stream-session-to-run links | `GameRunService` / repository | Durable SQLite link | Required FK | Canonical |
| `game_run_events` | Append-only run changes | `GameRunService` / repository | Durable SQLite event history | Required FK | Canonical |
| `beliefs`, namespace `game_run` | Current mutable run facts | `GameRunService` / `BeliefLifecycleService` | Durable, supersedable beliefs | Scope is the run id | Canonical |
| `GameRunState` | Runtime presentation of current run facts | Projected by `HebeEngine` from `GameRunService` | In-process snapshot only | Carries the active run projection | Not an owner |
| `game_knowledge_facts` | Typed links for stable Game knowledge | `GameKnowledgeService` / repository | Durable SQLite | Deliberately independent | Canonical |
| `beliefs`, namespace `game_knowledge` | Stable mechanics/entities/strategy claims | `GameKnowledgeService` | Durable, evidence-backed beliefs | Game scope, not run scope | Canonical |
| `game_knowledge_v2_gaps` | Typed unresolved research gaps | Game context v2 services | Durable SQLite | Optional contextual relation | Canonical research state |
| `GameProfileStore` | Curated spoiler-safe seed/profile input | Game knowledge resolver | Local curated/cache input | None | Input, not mutable truth |
| `game_research_cache` | Provider response cache | `GameResearchService` | Durable cache with TTL | None | Cache only |
| `GameDossier` | Research assembly/projection DTO | `GameResearchService` | In-memory only after cutover | None | Non-authoritative projection |
| `GameProgressState` | Spoiler-firewall input projection | Research/spoiler policy | In-memory DTO only | No persistence or ownership | Non-authoritative projection |
| `StreamSessionState` | Current stream observation and active run pointer | Stream runtime / engine | Session process state | `active_game_run_id` points to owner | Observation/projection |
| `LiveSessionBrain` / `live_session_timeline` | Events observed during a stream | Live session services | Durable observation timeline | May carry contextual run evidence | Observation, not run truth |
| `game_progress_states` | Historical mutable Game snapshot | Migration only | Retained SQLite source | Optional compatibility link | Legacy historical |
| `game_sessions` | Historical session notes/snapshot | Migration only | Retained SQLite source | Optional compatibility link | Legacy historical |
| `game_dossiers` | Historical assembled dossier | Migration only | Retained SQLite source | None | Legacy historical |

## 2. Domain separation

- Run state: current game/run, party and characters, progress, location,
  objective and challenge constraints. Durable facts are scoped to a canonical
  run and written through `GameRunService`.
- Game knowledge: relatively stable mechanics, entities and strategy. It is
  game-scoped, evidence-backed and owned by `GameKnowledgeService`.
- Stream/Game observation: what was seen or said during a stream. It remains an
  observation until a canonical service admits it as run state or knowledge.
  Stream state and runtime DTOs cannot establish durable truth themselves.

## 3. Ownership before cutover

Stream category changes resolved/created a modern run and also started a legacy
`GameProgressTracker` state. `HebeEngine` constructed and mutated
`GameRunState`, legacy progress snapshots and modern run facts. Manual session
notes wrote `game_sessions`; the knowledge resolver combined stream fields and
legacy session rows; response context consumed legacy progress plus dossiers.
The same event could therefore create a modern run, a legacy snapshot and a
runtime snapshot with no single mutation owner.

## 4. Ownership after cutover

| Concern | Final owner |
|---|---|
| Current game and identity | `GameV2Repository` + `GameRunService.resolve` |
| Run creation, identity and lifecycle | `GameRunService` |
| Session links | `GameRunService` / `GameV2Repository` |
| Run state write/read | `GameRunService` using `game_run` beliefs and events |
| Progress, party, characters, location, objective, challenge | `GameRunService`, run-scoped |
| Provenance/confidence admission guard | `GameRunService` |
| Stable Game knowledge and persistence | `GameKnowledgeService` |
| Knowledge retrieval | `GameContextResolver` |
| Runtime run projection | `GameRunState`, populated from the canonical service |
| Stream observation | `StreamSessionState` and `LiveSessionBrain` |
| Replay/restore | Game v2 services and canonical persisted state |

`HebeEngine` coordinates these owners but is not a Game repository.

## 5. Dual reads and writes found

The removed paths were:

1. stream online/category event → `GameRunService.resolve` and
   `GameProgressTracker.start`;
2. context preparation → canonical run resolution plus legacy progress/dossier
   reads;
3. manual progress/location/objective → runtime `GameRunState` plus legacy
   `game_sessions` notes and/or modern facts;
4. personal Game memory → stream fields plus `game_sessions` fallback;
5. response context → modern context plus legacy dossier/progress projections;
6. cutover flags and telemetry selecting or shadowing both implementations.

All six now have one production route. Research cache/profile data can enrich a
response, but cannot mutate run truth.

## 6. Classification of the 17 historical `game_progress_states`

The audit ran against a read-only source and a temporary copy. None of the
records contained chapter, area, party, character, boss, mechanic or progress
claims; they contained only playthrough/spoiler metadata and confidence.

| Game / stream session | Classification | Migration result and reason |
|---|---|---|
| `baldur_s_gate_3` / `current` | `AMBIGUOUS` | skipped; pseudo-session has no stable identity |
| `final_fantasy_v` / `current` | `AMBIGUOUS` | skipped; pseudo-session has no stable identity |
| `persona_5_royal` / `current` | `AMBIGUOUS` | skipped; pseudo-session has no stable identity |
| `super_robot_taisen_og_saga_endless_frontier` / `current` | `AMBIGUOUS` | skipped; pseudo-session has no stable identity |
| `the_adventures_of_elliot_the_millennium_tales` / `current` | `AMBIGUOUS` | skipped; pseudo-session has no stable identity |
| `super_robot_taisen_og_saga_endless_frontier` / `45`, `52`, `53` | `ORPHANED` | skipped; no canonical run/session link |
| `persona_5_royal` / `47`, `55` | `ORPHANED` | skipped; no canonical run/session link |
| `disgaea_mayhem` / `48` | `ORPHANED` | skipped; no canonical run/session link |
| `the_adventures_of_elliot_the_millennium_tales` / `48`, `49`, `57`, `59` | `ORPHANED` | skipped; no canonical run/session link |
| `final_fantasy_v` / `54` | `ORPHANED` | skipped; no canonical run/session link |
| `persona_5_royal` / `56` | `CURRENT_RUN_STATE` | verified link to `game_run_f693525d883d48fd8218ebf62e07eef7`; no semantic claims to create, so deduplicated |

Totals: 5 `AMBIGUOUS`, 11 `ORPHANED`, 1 `CURRENT_RUN_STATE`, 0
`HISTORICAL_RUN_STATE`, 0 `GAME_KNOWLEDGE`, and 0 `STREAM_OBSERVATION`.
The canonical run count stays one and no run belief is invented.

## 7. Dossier with `v2_projection_version=0`

The only row is `final_fantasy_v`, dossier version 2, status `partial`. Its
non-empty content is identity/aliases, completed research-section markers and
unsafe-topic policy. It has no semantic mechanic claims and no claim-level
sources. Version zero means not yet reviewed by the canonical projection; it is
not evidence that the row is safely migratable.

The migration records `GAME_KNOWLEDGE / skipped / no_semantic_claims`, preserves
the complete legacy row, creates zero beliefs/facts, and changes
`v2_projection_version` to 1 to mean reviewed/processed, not imported. It never
converts the dossier into run state.

## 8. Migrations

Two separate one-shot migrations run after the Game v2 schema:

- `game_run_state_canonicalization:1` validates legacy schemas, resolves only
  explicit or uniquely verified run/session links, migrates supported run facts
  to `game_run` beliefs/events, and audits every source row in
  `legacy_game_run_state_migration_audit`.
- `game_knowledge_canonicalization:1` admits only semantic claims with a
  claim-level source location and supporting excerpt, then writes canonical
  beliefs/evidence/fact links and audits each dossier in
  `legacy_game_knowledge_migration_audit`.

IDs are deterministic, equivalent facts deduplicate, migrations are
transactional and idempotent, and incompatible schemas prevent the completion
marker. A second pass reports both migrations already applied.

## 9. Orphaned and ambiguous data

No missing `game_run_id` is guessed from title or recency and no run is created
for migration convenience. Ambiguous rows are audited as `ambiguous`; orphaned
rows are audited as `skipped`. Historical tables and rows remain available for
manual review, rollback and forensic recovery.

## 10. Remaining legacy runtime reads/writes

Normal runtime reads: zero. Normal runtime writes: zero. Source search finds the
three legacy table names only in the two versioned migration implementations
and in additive historical-schema columns under `replay/migrations.py`.

## 11. Dead code removed

Removed persistent dossier/progress access, `game_sessions` read/write helpers,
legacy progress startup and mutation, legacy response fallbacks, shadow
telemetry, Game cutover branches/environment mappings, stale imports and four
tests that asserted `GameProgressTracker` as a second mutable owner. The tracker
and its in-memory store are gone. Historical replay flag fields remain parser
metadata only and cannot gate production.

`GameDossier` and `GameProgressState` remain only as non-persistent research and
spoiler-policy DTOs. They cannot create or restore run truth.

## 12. `HebeEngine` cleanup

The engine runs both migrations, wires one repository/run service/knowledge
service/context resolver set, resolves stream Game changes through the canonical
service, writes and clears run facts through that service, and projects its
result into runtime state. Manual notes, offline pause, current-game overrides,
guidance context and knowledge diagnostics no longer touch legacy storage. No
general engine refactor was performed.

## 13. Thread lifecycle findings

`GameResearchService` eagerly creates one `ThreadPoolExecutor` with up to two
`hebe-game-research` workers. Previously it had no shutdown owner, so tests and
replay could leave workers alive during interpreter teardown. This is a direct
and credible lifecycle risk for the earlier native teardown crash, although the
historical crash cannot be proven retrospectively from the surviving evidence.

The service now has idempotent `close(wait=False)` and `HebeEngine.stop()` owns
the call. Tests cover repeated close; replay logs confirm worker closure on
restart and teardown. No broader concurrency rewrite was made.

## 14. Tests

Fifteen new Phase 1C tests cover A–N plus worker shutdown: unique run identity,
single canonical current-game write, valid and rejected provenance, progress,
run-scoped location/objective/challenge, restart reconstruction, linked legacy
migration, orphan/ambiguous refusal, claim-level dossier migration,
canonical-only post-cutover writes, knowledge independence and clean game
switching. Existing stream, guidance, knowledge, primer, voice and research
tests were adjusted only where they asserted retired ownership.

## 15. QA

- Focused Game canonicalization/integration: 280 passed, 1 accepted failure,
  14 subtests after the final dead-code sweep.
- Broader Game context/guidance/primer/presence/live-brain run before the final
  sweep: 124 passed, 10 subtests.
- Game research/live regressions: 136 passed before the final sweep.
- Phase 3 replay set: 12 scenarios `VERIFIED`; relevant restart scenarios each
  performed the expected restart and closed Game research workers.
- Full backend before the final four stale-test removals: 1050 passed, 1
  accepted failure, 5 warnings, 84 subtests passed.
- Final full backend after the dead-code sweep: 1046 passed, 1 accepted failure,
  5 warnings, 84 subtests passed.
- `compileall`: passed. `git diff --check`: passed (line-ending notices only).

## 16. Remaining failure

Only
`test_response_synthesizer_handles_game_knowledge_command_result` remains. It is
the accepted R4/Persona renderer/guard behavior: the deterministic command data
contains `Persona 5 Royal`, but the Persona repair path returns the safe fallback
`Te leo, Leo. Recalibro.`. It is not changed in Phase 1C.

## 17. Historical risks

Ambiguous and orphaned rows remain unassigned by design. A migration already
marked complete will not reconsider later hand-edited legacy data. The dossier
cannot be reconstructed into sourced knowledge from section-status metadata.
Historical tables remain physical rollback/audit material and must not be
dropped without a separately approved migration. Research workers must be
closed by every future non-engine owner as well as by the engine lifecycle.

## 18. New-state guarantee

New durable Game state is born only in `game_runs`, `game_run_sessions`,
`game_run_events` and run-scoped canonical beliefs. New stable knowledge is born
only through `GameKnowledgeService` as evidence-backed Game knowledge. Stream
state and research DTOs are observations/projections. Runtime legacy Game state
reads and writes are both zero outside isolated, versioned migrations.

## 19. Proposed Phase 1D Social — not started

Begin with a contract-first inventory of social identity, relationship state,
episodes, viewer/chatter profiles, Twitch presence and proactive behavior.
Separate durable relationship truth from stream observations and delivery
policy; classify historical data before designing migrations; identify dual
owners and write tests before any cutover. Do not reuse Game or Memory migration
assumptions without evidence.
