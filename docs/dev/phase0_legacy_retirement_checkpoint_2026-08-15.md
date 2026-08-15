# Phase 0 legacy retirement checkpoint — 2026-08-15

> Historical checkpoint. Its Memory compatibility findings were superseded by
> `phase1b_memory_canonicalization_checkpoint_2026-08-15.md`; do not treat its
> adapter/runtime status as current architecture.

This checkpoint is the boundary before Phase 1. It records what was removed,
what remains, the evidence for each decision, and the migrations that must be
approved before persistent compatibility state can be retired. No production
database was modified during this audit.

## 1. Frozen starting point

- Approved ownership checkpoint: `24be7ef refactor: consolidate phase 0 cognitive ownership`.
- Corrective parent retained: `0d56433 chore: remove accidental runtime dataset changes from phase 0 checkpoint`.
- Remote was not changed; `origin/main` remained at `56e235d` when this pass began.
- Accepted baseline: 1022 passed, one known R4/Persona failure, five warnings,
  84 subtests. R4 is explicitly outside this cleanup.

## 2. Persistent-state evidence

The two repository databases were opened in SQLite read-only URI mode. The
root `hebe.db` contains no v2 cognitive/game rows and four legacy chatter
profiles. The active-looking `backend/hebe.db` contains:

| Store | Rows | Migration/link evidence |
|---|---:|---|
| `memory_facts` | 0 | No rows require local migration; the table still has active production writers/readers |
| `beliefs`, `belief_evidence` | 0, 0 | Modern schema exists, but local legacy facts have not populated it |
| `conversations`, `open_threads` | 0, 0 | No durable pending payload requires local backfill |
| `game_progress_states` | 17 | 0 have `game_run_id` |
| `game_dossiers` | 1 | `v2_projection_version=0` |
| `game_runs`, `game_run_events` | 1, 1 | Modern game state already exists beside legacy state |
| `chatter_profiles` | 39 | 10 logins match modern identities; 31 profiles are unmatched |
| `stream_chatter_summaries` | 184 | 121 summaries belong to logins without a modern identity |
| `people`, `person_identities` | 10, 10 | Modern identity writes are active beside legacy social writes |

The counts prove that Game and Social require data migration even though their
named compatibility adapter classes did not perform that migration.

## 3. Adapter disposition summary

| Adapter | Initial classification | Reads/writes in normal runtime | Final disposition |
|---|---|---|---|
| `LegacyPendingAdapter` | `STILL_ACTIVE` + `MIGRATION_REQUIRED` | Projects every canonical-shaped pending dict into a persisted v2 conversation and closes that conversation with the dict | Retained; still in the normal pending hot path |
| `LegacyMemoryFactAdapter` | `MIGRATION_REQUIRED` | Instantiated in runtime, but called only by the Phase 2 compatibility replay; it can read legacy facts and optionally write them | Retained as migration machinery, not a normal request hot path |
| `LegacyGameCompatibilityAdapter` | `SAFE_TO_REMOVE` | No data conversion; observes two objects and appends discarded in-memory telemetry | Removed |
| `LegacySocialCompatibilityAdapter` | `SAFE_TO_REMOVE` | No caller at all; only an empty telemetry object was exposed in replay probes | Removed |

## 4. LegacyPendingAdapter — A through I

### A/B. Legacy input and canonical output

It reads the runtime `HebeState.pending_clarification` dictionary: `kind`,
`expected_reply_type`, candidates, expiry and event identifiers. It maps these
to `ExpectedReply` and opens a persisted `Conversation` through
`ConversationContinuityService`. It then writes the conversation id back into
the dictionary.

All newly constructed pending payloads, including developer simulation seeds,
now go through `_make_pending_task`, so their format contains ids, authority,
allowed sources, capability, timestamps, expiry, attempts and status. The
remaining problem is ownership direction: the dictionary is written first and
the typed conversation is still a projection.

### C/D. Construction and consumers

`HebeEngine._initialize_conversation_continuity` constructs it. `_set_pending_task`
calls `project_legacy_pending`; `_clear_pending_task` calls `close_for_legacy`.
The v2 continuation resolver consumes the created conversation and maps its
resolution back to the runtime pending state.

### E. Participation

- Read: active runtime pending dictionaries.
- Write: `conversations` plus `pending["conversation_id"]`.
- Shadow write: yes; the dict and conversation are written for one semantic pending action.
- Migration: no one-shot migration exists.
- Fallback: closing a dict-owned pending action closes its projected conversation.

### F/G. Persistent dependency and modern equivalent

No known database currently has an actionable persisted conversation, and
pending dictionaries are process-local. The modern equivalent is
`ConversationContinuityService` plus `ConversationRepository` and
`OpenThreadRepository`. Nevertheless, new runtime actions still depend on the
adapter to reach that modern store.

### H. Protection

`test_conversation_continuity_phase1.py` directly verifies the adapter mapping.
Phase 1 replay fixtures verify continuation, authority, expiry, interruption,
duplicates, cancellation and restart invalidation. Voice pipeline tests cover
appointment, promotion and game-guidance pending behavior.

### I. Removal condition and migration design

1. Make `ConversationContinuityService.open_conversation` the only creation
   write. Produce `HebeState.pending_clarification` as a read-only compatibility
   projection from the returned conversation/domain payload.
2. Add exactly-once contract tests for every pending kind, including no-wake
   owner follow-up, cancellation, TTL, interruption and restart.
3. On startup, archive any old active conversation that lacks the new canonical
   schema/version. There is no process-local legacy state to backfill after a
   restart.
4. Remove the direct projection/close calls, the adapter test, and the adapter
   only after the runtime has no dict-first writer.

## 5. LegacyMemoryFactAdapter — A through I

### A/B. Legacy input and canonical output

`shadow_project(fact_id)` reads one `memory_facts` row (`kind`, `subject`,
`payload_json`, source text and confidence) and proposes a `SUSPECTED` belief in
`legacy.<kind>` with evidence pointing back to that row. `project_to_legacy`
does the inverse optional write by inserting a `memory_facts` row linked through
`belief_id`.

### C/D. Construction and consumers

`HebeEngine._initialize_belief_v2` constructs it. No production input path
calls either method. `CognitiveReplayRunner` is the sole repository consumer of
`shadow_project`; no caller of `project_to_legacy` exists.

### E. Participation

- Read: explicit compatibility replay only.
- Write: belief proposal in replay; optional v2-to-legacy method has no caller.
- Shadow write: demonstrated only by the replay, not wired to `MemoryExtractor`.
- Migration: it is row-level migration machinery, but has no batch, watermark,
  idempotent audit, or cutover.
- Fallback: none in normal runtime.

### F/G. Persistent dependency and modern equivalent

The known databases contain zero `memory_facts`, so no local historical rows
need immediate conversion. The store is still active: `MemoryExtractor` calls
`upsert_memory_fact`, and `ContextBuilder`, `MemoryStore`, API diagnostics and
database helpers still read it. The modern equivalent is
`BeliefLifecycleService`/`BeliefRepository` with explicit evidence and
`MemoryRetrievalCoordinator`.

### H. Protection

`cognitive_replay_phase2/h_legacy_compatibility.json` verifies that a legacy
row becomes a suspected belief with provenance. The Phase 2 epistemics suite
protects belief lifecycle, evidence, restart and supersession behavior.

### I. Removal condition and migration design

1. Inventory every supported `memory_facts.kind` and define a typed namespace,
   predicate, object, authority and evidence mapping. Unknown/ambiguous rows go
   to `NEEDS_REVIEW`; they must not become known facts.
2. Build an idempotent one-shot migration with a durable audit/watermark and
   `memory_facts.belief_id`. Preserve source text and inactive history.
3. Move `MemoryExtractor` to validated belief proposals and move all semantic
   readers to beliefs. Retrieval-only chunks may remain caches.
4. Disable v2-to-legacy writes, prove zero unlinked active rows and zero legacy
   reads over the agreed release window, then remove the runtime adapter and
   compatibility replay event.

The adapter is retained because deleting the only provenance-preserving
conversion before the migration is designed would make non-local historical
databases harder to migrate. It is not a normal hot-path consumer.

## 6. LegacyGameCompatibilityAdapter — A through I

### A/B/C/D/E

The class received a `GameProgressState` or `GameDossier` and returned a small
`COMPATIBILITY_ONLY` dictionary. `HebeEngine` constructed it and ignored both
return values. It performed no reads, writes, shadow writes, migration or
fallback; only replay-probe telemetry observed its internal lists.

### F/G

Persistent legacy data does exist, but it is owned directly by
`GameIntelligenceStore`: `game_progress_states` and `game_dossiers`. The modern
equivalent is `GameRunService`, `GameKnowledgeService` and
`GameContextResolver`. Removing the no-op adapter does not migrate or delete
either model.

### H/I

No test or replay asserted the adapter output. The Phase 3 suite and its replay
fixtures cover the modern game model. The adapter was therefore removed along
with its calls and probe field. The persistent cutover remains:

1. Map every legacy progress row to a stable game identity and run; `current`
   rows become a checkpoint only after provenance/conflict validation.
2. Convert dossier facts individually with source/spoiler/progress metadata;
   do not promote the entire dossier blob to truth.
3. Backfill `game_run_id` and `v2_projection_version` in an audited one-shot
   migration, compare reads, then move `GameIntelligenceService` writes to the
   v2 repositories.
4. Remove legacy reads/tables only after all 17 progress rows and the dossier
   are accounted for and restart/replay parity is green.

## 7. LegacySocialCompatibilityAdapter — A through I

### A/B/C/D/E

The class had a generic `observe(source, value, classification)` method that
would append telemetry. Nothing called it. `HebeEngine` constructed it and the
replay probe exposed its empty telemetry. It performed no read, write, shadow
write, migration or fallback.

### F/G

The data dependency is real but bypassed the adapter. Stream memory directly
writes/reads `chatter_profiles`, `chatter_facts` and
`stream_chatter_summaries`; the modern Twitch path independently calls
`SocialWorldService.resolve_person` and writes `people`/`person_identities`.
Removing the no-op class does not affect either owner.

### H/I

No test or replay asserted adapter behavior. The Phase 4 suite and replay
fixtures protect stable identity, rename handling, privacy, familiarity,
episodes, follow-ups and culture. The adapter was removed. Persistent cutover
must:

1. Match by stable Twitch user id first, normalized login/aliases second; never
   merge ambiguous similar names.
2. Create modern identities for the 31 unmatched profiles while retaining
   first/last seen, aliases and counters as provenance, not personality truth.
3. Convert 121 unmatched historical summaries into bounded social episodes or
   archive references according to privacy/sensitivity and retention policy.
4. Dual-read for comparison, switch chat observation to one canonical social
   writer, then retire legacy writes only after counts, identity isolation and
   cross-session replay are verified.

## 8. ToolSystem, InteractionActions and orchestrator audit

| Surface | Builder / real consumer | Reachability and contract | Disposition |
|---|---|---|---|
| `HebeRuntime.tools` / `ToolSystem` | `build_runtime`; potential external callers through `runtime.tools.call/exec` | Constructed in production and part of the public runtime dataclass. Its `open_app` callback can still invoke `InteractionActions` outside the cognitive input loop | `PUBLIC_RUNTIME_CONTRACT`, retained pending a versioned replacement |
| `HebeRuntime.actions` / `InteractionActions` | `build_runtime`; callback provider for ToolSystem | Constructed and publicly exposed; no internal cognitive caller | `PUBLIC_RUNTIME_CONTRACT`, retained pending a versioned replacement |
| `app.legacy.dispatcher`, legacy intent resolver/catalog | No import, constructor, entrypoint, documentation contract or test | Internally unreachable; manual imports were the only hypothetical consumer | `DEAD`, removed |
| Old orchestrator gates/policy/executor/dispatcher/tool handlers | Only referenced each other; no bootstrap, entrypoint or tests | Internally unreachable. `handle_open_app` was the second hypothetical direct app-open route | `DEAD`, removed |
| `orchestrator/intents/resolver.py`, catalog and two DTOs | `app/scripts/eval_intents.py` | Developer tooling only, not a production input loop | `COMPATIBILITY_ONLY`, retained and trimmed |
| Cognitive app-open chain | `HebeEngine` input loop | `CognitiveRouter -> DeliberationService -> PlanExecutor -> ActionRuntime -> LocalCapabilityResolver -> WinAutomationService` | `STILL_ACTIVE`, canonical |

There is now no second app-open route in the old orchestrator or legacy
dispatcher. One externally invocable compatibility route remains through the
public `HebeRuntime.tools/actions` fields. Removing it now would break the
runtime dataclass contract, so its deprecation needs an explicit replacement:

1. Add a versioned public action facade that always enters `PlanExecutor`.
2. Add contract tests and call telemetry for public tool/action use.
3. Deprecate `runtime.tools` and `runtime.actions` for one release, with the
   facade documented as replacement.
4. Remove the fields only after known external scripts have migrated.

Manual imports of deleted `app.orchestrator` execution modules and
`app.legacy.dispatcher` were not a declared public API. Their migration target
is the cognitive router/plan executor, or the retained intent evaluator for
offline intent evaluation.

## 9. Dead-code sweep

Removed:

- `LegacyGameCompatibilityAdapter` and `LegacySocialCompatibilityAdapter`;
- their construction, observation calls, exports and replay telemetry fields;
- old orchestrator gates, policy, executor, dispatcher, tool handlers, prompts
  and top-level orchestrator;
- the orphan legacy dispatcher, intent resolver and NLU catalog;
- orphan orchestrator decision/execution/pending DTOs and result helpers;
- unused `TOOL_INTENTS` compatibility catalogue.

Retained intentionally:

- `LegacyPendingAdapter`: active dict-first to conversation projection;
- `LegacyMemoryFactAdapter`: migration-only machinery and Phase 2 replay;
- `ToolSystem`/`InteractionActions`: public runtime compatibility contract;
- intent evaluator resolver/catalog/input/result DTOs: real developer script.

## 10. Final hot paths and known duplicate ownership

- Normal Hebe input uses the cognitive router, deliberation and plan executor.
- App opening has one internal production execution path.
- Pending still has a known dual write: runtime dict first, v2 conversation
  second. It is documented and blocks removal of `LegacyPendingAdapter`.
- Game still writes legacy progress/dossiers while a v2 run model exists.
- Social still writes legacy chatter history while modern identity writes exist.
- Memory still writes/reads `memory_facts`; the adapter does not shadow those
  writes automatically.

Therefore it would be false to claim that Phase 0 found no dual writes. The
accurate closure statement is: there are no *unknown* dual decision owners or
second internal app-open routes; the remaining compatibility dual persistence
paths are enumerated above and require approved data migrations.

## 11. QA

Focused verification completed before the final full run:

- Social adapter removal: `test_social_world_phase4.py` — 10 passed, including
  its real-restart replay.
- Game adapter removal: `test_game_context_phase3.py` — 10 passed, including
  its replay coverage.
- App-open/orchestrator removal: app-open architecture plus voice command
  pipeline — 173 passed and 9 subtests.

Final verification:

- full backend suite: 1022 passed, one accepted R4 failure, five warnings and
  84 subtests passed;
- sole failure: `test_response_synthesizer_handles_game_knowledge_command_result`;
- `python -m compileall -q backend/app backend/tests`: exit 0;
- `git diff --check`: exit 0;
- final source scan: no reference to either removed adapter, old orchestrator
  execution modules, legacy dispatcher/resolver, or `handle_open_app` remains.

## 12. Phase 1 proposal — not started

Phase 1 should begin with migration design and ownership cutover, not a new
personality or cognitive system:

1. conversation-first pending creation and removal of `LegacyPendingAdapter`;
2. audited `memory_facts -> beliefs` migration and writer/reader cutover;
3. audited game progress/dossier migration to one game persistence model;
4. stable-id-first social history migration and one canonical observation
   writer;
5. versioned replacement and deprecation of public runtime tools/actions.

No Phase 1 implementation is part of this checkpoint.
