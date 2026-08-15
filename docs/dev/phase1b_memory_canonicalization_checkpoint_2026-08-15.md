# Phase 1B — Memory canonicalization checkpoint

Status: implemented and verified; intentionally not committed. Phase 1C has not started.

## 1. Legacy model found

Historical structured memory used `memory_facts(id, kind, subject, payload_json,
source_text, confidence, created_at, updated_at, last_used_at, active)`. Phase 2
later added optional `belief_id` and `epistemic_status` compatibility columns.
The row has no explicit owner, source identity, scope, sensitivity, authority, or
lifecycle history beyond `active`; provenance is limited to free-form
`source_text`, payload metadata, and the row itself.

The retired adapter read a row, created `legacy.<kind>` / `legacy_fact`, forced
`owner_local:leo`, used current wall-clock time, and marked it `SUSPECTED`. Its
reverse projection could write a compatibility row. That discarded legacy
timestamps and invented scope and predicate defaults. Only the historical replay
invoked the forward projection; no production caller invoked reverse projection.

## 2. Complete known `memory_facts.kind` inventory

The inventory covers current and reachable code, Git history, schema comments,
migrations, fixtures, tests, docs, scripts, SQL, and replay data.

| Kind | Classification | Canonical treatment |
|---|---|---|
| `preference` | `DIRECT_MAPPING` | `memory.preference`, private owner scope; payload retained |
| `leo_fact` | `DIRECT_MAPPING` | `memory.owner_fact`, private owner scope; payload retained |
| `hebe_identity` | `DIRECT_MAPPING` | `memory.hebe_identity`, assistant-local scope; payload retained |
| `project_fact` | `DIRECT_MAPPING` | `memory.project`, private owner scope; payload retained |
| `habit` | `DIRECT_MAPPING` | `memory.habit`, private owner scope; payload retained |
| `viewer_fact` | `DIRECT_MAPPING` | `memory.viewer`, viewer-local scope keyed by subject |
| `appointment` | `TRANSFORM_REQUIRED` | requires title and `due_at`; becomes `memory.appointment/scheduled_appointment` |
| `fact` | `TRANSFORM_REQUIRED` | requires explicit `predicate` and `value`; becomes `memory.fact` |
| `person` | `TRANSFORM_REQUIRED` | requires subject, explicit `predicate` and `value`; remains private |
| `stream_fact` | `TRANSFORM_REQUIRED` | requires channel or explicit stream/Twitch source evidence before public scope |
| `task` | `OBSOLETE` | skipped; task execution state is not an epistemic belief |
| `misc` | `AMBIGUOUS` | unsupported; no semantic claim is inferred |

Any other dynamically supplied kind is `UNSUPPORTED`. It is audited as
`unknown_kind` and creates no belief.

## 3. Explicit legacy-to-canonical mapping

Each accepted row becomes one `beliefs` row plus one `belief_evidence` row.
`id` is deterministic from migration version and legacy row id; evidence points
to `memory_facts:<id>`. Confidence becomes belief confidence and evidence
weight. `created_at`, `updated_at`, source text, original payload,
`last_used_at`, and `active` are preserved across belief/evidence fields.
Active rows become at most `SUSPECTED` unless the compatibility status was
already `INFERRED`/`SUSPECTED`; inactive rows become `HISTORICAL`. Legacy data
can never manufacture `KNOWN` owner truth.

## 4. Ownership before cutover

- Create/write/update: `MemoryExtractor`, `PlanExecutor` and `MemoryStore` via
  `db_sqlite` CRUD; the adapter also exposed an unused reverse shadow write.
- Read/context: `MemoryStore`, `ContextBuilder`, Twitch viewer lookup, and
  `/debug/memory` read the legacy table.
- Retrieval: model context combined structured legacy rows and vector chunks,
  while the modern coordinator independently retrieved beliefs.
- Persistence/retire: `db_sqlite` and hygiene code directly mutated legacy rows.

This was a real dual ownership path, even though the inspected local DB had zero
rows.

## 5. Ownership after cutover

- Persistence: `BeliefRepository`.
- Admission/update/retire: `BeliefLifecycleService`.
- Retrieval: `MemoryRetrievalCoordinator`.
- Runtime facade: `MemoryStore` projects canonical beliefs into the existing
  `MemoryFact` presentation shape; it does not access legacy storage.
- Writers: `MemoryExtractor` and `PlanExecutor` write only through that facade.
- Readers: private context, Twitch viewer context, appointments and debug memory
  read only canonical retrieval results.
- `HebeEngine` shares these service instances and owns no compatibility adapter.

## 6. Dual paths found and removed

Removed: legacy CRUD, extractor upserts, context fallback/LIKE query, Twitch
legacy lookup, debug legacy reads, hygiene mutation, adapter shadow read/write,
adapter telemetry, replay events and production cutover flags. Vector
`memory_chunks` remain retrieval aids rather than canonical truth.

## 7. Migration design

`memory_canonicalization:1` runs after belief schema setup at structured-memory
startup. It detects the legacy table and required schema, handles every row in a
single transactional migration, and records one row per source record in
`legacy_memory_fact_migration_audit`. A schema-level marker lives in
`schema_migrations`. An unsupported table schema aborts without setting the
marker; corrupt individual rows are audited and do not block valid rows.

## 8. Implemented behavior

Outcomes are `migrated`, `deduplicated`, `skipped`, `unsupported`, or `error`.
Equivalent canonical identity/object pairs are reused and receive the legacy
evidence. An already-valid `belief_id` compatibility link is recorded as
deduplicated. Re-running after success is a no-op through the migration marker.

## 9. UNKNOWN and corrupt rows

Unknown/ambiguous kinds never create a generic or `legacy.*` belief. Missing
semantic fields, invalid JSON/confidence/timestamps, and incomplete records are
visible in the audit with their reason. No error silently chooses owner, scope,
predicate, or value.

## 10. Adapter state

The compatibility adapter source, import, construction, metrics, replay hooks,
and branches are removed. Production source references: zero.

## 11. Physical `memory_facts` state

Historical tables and rows are not dropped or rewritten. Existing DBs retain
them as rollback/audit backup. Fresh databases no longer create the table.
Physical deletion is deferred to a later, separately approved migration.

## 12. Dead code removed

Removed legacy CRUD/touch/deactivate functions, STT cleanup mutation of legacy
rows, hygiene reader/writer, shadow telemetry, replay compatibility projection,
stale stream schema ownership text, and the unused production belief read/write
flags. Historical replay JSON flag keys are tolerated only while parsing old
fixtures; they do not produce runtime attributes or environment flags.

## 13. HebeEngine cleanup

Engine initialization reuses `MemoryStore.repository`, `.lifecycle`, and
`.retrieval`; it no longer constructs a second set of Memory services or any
legacy adapter. No general engine refactor was performed.

## 14. Tests

Twelve Phase 1B tests cover contracts A–L: direct mapping, provenance and
confidence, timestamps, rerun idempotency, canonical deduplication, semantic
transformation, obsolete and unknown kinds, corrupt-row isolation,
canonical-only retrieval, canonical-only new writes, and restart after the
migration marker. The replay compatibility fixture was replaced one-for-one by
`phase2_h_canonical_memory_restart`.

## 15. QA

- Focused Memory/belief/context/reminder tests: 45 passed.
- Phase 2 replay scenarios: 10 `VERIFIED`, including canonical Memory restart.
- Full backend: 1035 passed, 1 accepted failure, 5 warnings, 84 subtests passed.
- Accepted failure only: `test_response_synthesizer_handles_game_knowledge_command_result` (R4 Persona).

## 16. Remaining failures

Only the pre-existing R4 Persona failure remains. No Phase 1B regression exists.

## 17. Historical DB risks

Rows with undocumented kinds, ambiguous semantics, corrupt fields, or an
unsupported table schema require manual review of the audit. `stream_fact`
without public-scope evidence is deliberately not migrated. The table remains
available for rollback and forensic recovery. A successfully marked migration
will not automatically reconsider rows later edited by hand.

## 18. New-memory guarantee

New structured memories are born only as beliefs with evidence. Runtime reads
and writes to `memory_facts` are zero; only the isolated versioned migration can
read it. Reminder links now store `source_belief_id` in reminder payload and do
not write the legacy foreign-key column.

## 19. Proposed Phase 1C — not started

Audit Game ownership and schemas first, define run-state versus durable game
knowledge boundaries, inventory legacy representations, write migration and
retrieval contracts, then consolidate one owner at a time. No Game code or
behavior has been changed in Phase 1B.
