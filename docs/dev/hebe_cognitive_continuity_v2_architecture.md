# Hebe Cognitive Continuity & Learning v2

Status: architecture and migration blueprint only  
Repository reviewed: current working tree on 2026-08-11  
Production behavior changed by this task: **no**

## Executive decision

Hebe should evolve by putting a typed continuity and belief layer between the existing event stores and the existing cognitive pipeline. The model may propose interpretations and consolidations, but deterministic services own state transitions. Existing ingestion, authority, execution, output, evidence, research, presence, and replay boundaries remain in place.

This is a strangler migration, not a second Hebe stack:

```text
STT / UI / Twitch / internal events
                |
        existing firewall + authority
                |
    canonical experience projection
    (extend live_session_timeline)
                |
  Conversation | Scene | Open Threads
                |
  Beliefs + domain repositories + RAG
                |
      existing CognitiveRouter
                |
 deliberation / synthesis / validation
                |
 existing PlanExecutor / domain executor
                |
 receipt -> HistoricalActionLedger projection
                |
       existing FinalEmissionGate
```

The principal ownership rule is: observations are evidence, beliefs are revisable interpretations, and successful receipts are the only proof of actions. Neither prompt text nor an LLM response may directly mutate authoritative state.

## Architectural invariants

1. `InputFirewall`, source authority, `CognitiveRouter`, plan validation, and capability checks remain upstream of action execution.
2. `PlanExecutor` and domain executors remain the only action path. A memory write cannot imply that an action occurred.
3. `FinalEmissionGate` remains the final owner of externally emitted speech/text and exactly-once behavior.
4. `PresenceEngine` remains a proposal source, not an output or authority bypass.
5. Game advice retains spoiler, evidence, and confidence guards. New memory can supply evidence; it cannot weaken those guards.
6. Raw evidence remains immutable and addressable. A semantic label never overwrites the source utterance/event.
7. Runtime contexts remain separate. Private, owner voice, Twitch chat, and autonomous stream behavior receive only the scopes they are allowed to see.
8. Model output is a proposal. Repository methods validate types, provenance, authority, confidence, lifecycle, and idempotency before committing it.
9. Retrieval is purpose-limited. Social continuity must not become exhaustive viewer profiling.
10. Schema changes are additive and versioned until replay and shadow-read parity permit cleanup.

## A. Current architecture mapping

### 1. CurrentConversation

**Existing coverage.** `HebeState.pending_clarification` in `backend/app/core/state.py`, the pending-task helpers and `pending_conversation_turn` helpers in `backend/app/hebe_engine.py`, `ConversationState`/resolver in `backend/app/cognitive/stream_companion_flow.py`, wake evidence in the STT input-envelope path, and expected-reply metadata from response synthesis already support pieces of turn handoff.

**Overlap and gaps.** There are two principal pending representations (`pending_clarification` and `pending_conversation_turn`) plus capability-specific payload shapes. They describe a task continuation more often than a generic conversation. Participant set, turn owner, semantic topic, compatibility rule, closure reason, and durable thread linkage are not represented consistently.

**Legacy/duplicate path.** Promotion, appointment, and game-guidance compatibility helpers encode domain-specific follow-ups. The dormant `backend/app/orchestrator/gates.py` has another pending model and must not be reconnected.

**Migration direction.** Introduce one `ConversationContinuityService` and one typed `CurrentConversation` projection. Initially it reads existing pending dictionaries and writes compatibility projections back to them. A wake word acquires attention only when no eligible conversation exists. When Hebe explicitly hands the turn to Leo, a compatible reply within the relevance window is addressed to Hebe without another wake word. Compatibility is deterministic first (source, participant, expected reply type, expiry, interruption state), with semantic classification as bounded supporting evidence.

### 2. CurrentScene

**Existing coverage.** `LiveSessionBrain`/`LiveSessionState` and `live_session_timeline` in `backend/app/stream/live_session.py` hold the richest persistent live context and provenance. `SceneTimelineManager`/`SceneTimelineState` in `backend/app/stream/scene_timeline.py` model scene versions and terminal transitions. `AmbientFact` in `backend/app/stream/ambient_context.py` retains literal text, semantic fields, confidence, evidence spans, and inference level. Stream state also carries current game, activity, combat, location, objective, boss, failure, and resource fields.

**Overlap and gaps.** Scene truth is split between a persistent timeline, a wide live-state snapshot, an in-memory scene manager, ambient facts, and stream state. Referents and belief status are not common types. Some derived labels risk becoming more prominent than their literal evidence.

**Migration direction.** Make `CurrentScene` a materialized projection over canonical timeline evidence, not a new source-of-truth event store. Every `SceneAssertion` references one or more evidence rows and contains semantic interpretation, referents, confidence, observed time, validity, provenance, and belief status. Snapshot fields remain compatibility projections until all readers move.

### 3. OpenThreads

**Existing coverage.** Pending clarifications, expected replies, spontaneity anchors, `LastHebeUtterance`, recent participant/topic state, game knowledge gaps, and schedule hypotheses each approximate a kind of unresolved thread.

**Missing.** There is no shared lifecycle for a question, promise, social follow-up, unresolved observation, or game objective across sessions. Expiration, snoozing, resolution evidence, and privacy/scope are inconsistent.

**Migration direction.** Add a typed `open_threads` store used by conversation, game, and social adapters. Threads are deliberately small records pointing to evidence and domain objects; they are not transcripts or free-form dossiers.

### 4. Memory hierarchy

| Target category | Existing repository mapping | Required evolution |
|---|---|---|
| WorkingMemory | `HebeState`, `LiveSessionState`, `StreamSessionState`, conversation history, rolling summaries | Treat as bounded projections with explicit TTL/session scope; do not copy every item into long-term memory. |
| EpisodicMemory | `chat_log`, `stream_chat_messages`, `stream_events`, `live_session_timeline`, `stream_summaries`, `memory_chunks` | Preserve raw events; create only salient episode records/chunks with source links. |
| SemanticMemory | `memory_facts`, `memory_chunks`, game dossiers/profiles, schedule hypotheses | Add belief lifecycle, validity, evidence links, and supersession; keep vector chunks as retrieval aids. |
| ProceduralPreferenceMemory | behavior constraints, promotion profiles, some `preference`/`hebe_identity` facts | Add typed, high-authority owner preferences; compile them into guards/policy rather than relying on recalled prose. |
| GameRunMemory | `GameRunState`, `game_progress_states`, `game_sessions`, live run fields | Introduce a run identity spanning stream sessions and typed run assertions/events. |
| SocialMemory | chatter profiles/facts/presence/summaries, stream chat, promotion profile, viewer language inference | Separate observed episodes from hypotheses and retain only socially useful continuity. |
| HistoricalActionLedger | promotion transactions/events/receipts, executor results, internal logs | Add an append-only cross-domain receipt index; retain each domain transaction as authoritative detail. |

`backend/app/cognitive/memory_store.py` is the structured `memory_facts` access layer; `backend/app/cognitive/memory/memory_store.py` is the vector `memory_chunks` layer. Their similar names obscure their different responsibilities and should be clarified later, not merged into one generic store.

### 5. Beliefs and hypotheses

**Existing coverage.** `schedule_observations` and `schedule_hypotheses` in `backend/app/stream/session_primer.py` already demonstrate tentative, active, weakening, superseded, and historical temporal reasoning. `chatter_facts` has confidence and expiry. Memory facts have confidence and active state.

**Missing.** There is no shared epistemic vocabulary or evidence graph. `active` alone cannot distinguish historical truth from discredited inference. Owner correction and contradictory evidence do not have a general transition protocol.

**Migration direction.** Reuse the schedule lifecycle as the reference behavior, then introduce common `BeliefStatus` values: `KNOWN`, `INFERRED`, `SUSPECTED`, `HISTORICAL`, `SUPERSEDED`. Status is not confidence: a high-confidence historical claim is still historical. Corrections append evidence and supersede prior beliefs; they do not erase source history.

### 6. GameKnowledge versus GameRun

**Existing coverage.** `backend/app/stream/game_intelligence.py` contains `GameDossier`, `RetrievedGameFact`, `GameProgressState`, gaps, research jobs, advice policy, and persistence in `game_dossiers`, `game_research_cache`, `game_progress_states`, and `game_knowledge_gaps`. `backend/app/stream/game_knowledge.py` resolves local profile plus personal session memory and optional research. `backend/app/stream/game_research.py` and `game_profiles.py` provide an older profile-oriented research path. `backend/app/cognitive/game_guidance.py` has another `GameRunState` and guidance/retrieval path. `session_primer.game_sessions` stores prior run summaries.

**Overlap and gaps.** General facts, profiles, run state, and research exist, but several paths own overlapping decisions and cache formats. A run is often keyed to one stream session, so an ongoing Persona 5 or FFV run lacks a first-class identity spanning sessions. Claim-level provenance and supersession inside JSON dossiers are limited.

**Migration direction.** Keep the current GameResearch provider abstraction and evidence/advice guards. Establish `GameContextResolver` as the one read hierarchy and repositories as adapters over existing stores. Normalize validated reusable claims gradually; add persistent run identity and run-session links.

### 7. Memory-first game research

**Existing coverage.** Game Intelligence already caches research, tracks gaps, has provider-neutral jobs, and applies spoiler/evidence rules. The older knowledge resolver checks personal memory and local profiles before optional web lookup.

**Gap.** The complete hierarchy is not enforced through one gateway, and successful research may update a profile/cache without becoming claim-level reusable knowledge across all callers.

**Migration direction.** All game consumers call:

```text
CurrentScene -> active GameRun -> validated GameKnowledge
             -> scoped memory/RAG -> research only for a typed gap
```

Research results enter quarantine as proposed claims. Citation, spoiler, source-quality, and contradiction validation occur before promotion into GameKnowledge. Cache freshness and a normalized gap key prevent repeated broad searches.

### 8. SocialWorld

**Existing coverage.** `stream_chat_messages`, `chatter_presence`, `chatter_profiles`, `chatter_facts`, `stream_chatter_summaries`, Twitch events, `viewer_profiles.py`, `social_events.py`, and promotion profiles provide identity clues, observations, aggregates, and authorized preferences.

**Overlap and gaps.** Records are frequently keyed by mutable username; observations and inferred summaries can be mixed; `relationship_level` and notes can drift toward an opaque CRM model. There is no shared social episode/hypothesis lifecycle.

**Migration direction.** Add minimal stable person identity/aliases and use ordinary beliefs/threads for interpretations. Keep raw chat and events as observations. Chatter aggregates remain operational materialized views, not personality truth. Apply retention, scope, sensitivity, and social-usefulness filters at write and retrieval time.

### 9. SharedCulture and running jokes

**Existing coverage.** Prompt examples and hard-coded voice material contain channel flavor, while stream summaries/chunks can retain episodes.

**Missing.** There is no evidence-backed shared-culture object with participants, origin, reactions, tone, reuse count, cooldown, or retirement state.

**Migration direction.** Add `shared_culture_items` linked to episode evidence and lightweight use/reaction records. Creation requires explicit owner confirmation or repeated positive social evidence. Selection enforces context, participants, cooldown, negative evidence, and maximum frequency. Never branch on a viewer name to emit a fixed phrase.

### 10. Scene transitions

**Existing coverage.** Scene timeline terminal states, stream phase, Twitch raid events, promotion/action execution, and presence proposals provide the ingredients.

**Missing.** Action consequences are not consistently projected as narrative scene transitions. Raid count telemetry can leak into wording because event rendering and social meaning are insufficiently separated.

**Migration direction.** A successful raid receipt emits a canonical `raid_started` action fact. The scene reducer derives `stream_ending` and `social_transition(destination_person)` from that receipt, then opens a bounded farewell/comment opportunity. Relationship/shared-culture retrieval is confidence-limited. Viewer count stays telemetry unless a specific policy explicitly asks for it; the default raid acknowledgement template does not expose it.

### 11. Action-history truth

**Existing coverage.** `PlanExecutor`, execution results, promotion `ActionReceipt`, `PromotionCommandTransaction`, promotion events, and executor logs already enforce much of the correct boundary.

**Missing.** There is no normalized append-only cross-domain ledger query for cognition. General memory could otherwise incorrectly repeat an intended action as completed.

**Migration direction.** Add a receipt-backed `HistoricalActionLedger` projection. Only the executor/transaction persistence boundary can append a successful action. Requested, attempted, failed, unknown, and succeeded remain distinct. Conversational claims query the ledger and cite the receipt/domain event ID.

### 12. HebeSelfModel

**Existing coverage.** `backend/app/cognitive/persona/hebe_identity.py` contains the stable identity; `hebe_voice.py` carries voice, preferences, opinions, examples, and channel flavor. Memory extraction can store `hebe_identity` and `preference`. Behavior constraints encode explicit owner boundaries.

**Overlap and gaps.** Stable identity, Hebe tastes, Leo-like phrasing, and shared culture are partially conflated in prompts. Learned identity facts do not have drift constraints or a reviewable lifecycle.

**Migration direction.** Keep `StableHebeCore` as version-controlled, owner-reviewed configuration that learning cannot overwrite. Put evolving opinions/preferences in typed beliefs with evidence and allowed domains. `LeoLanguageModel` records Leo's lexical/interaction preferences only; `HebeVoice` renders from core plus evolving Hebe preferences; `SharedCulture` supplies contextual channel references. Language adaptation cannot change authority, identity, or personality boundaries.

### 13. Learning and correction

**Existing coverage.** Memory extraction is conservative, schedule hypotheses accumulate observations, live-session voice relevance recognizes corrections, scene facts can be annotated/revalidated, and promotion profiles learn only after successful transactions.

**Gap.** There is no common proposal/validation/consolidation protocol. The current extractor can upsert stable facts directly from a model result with a relatively thin lifecycle.

**Migration direction.** Use `Observation -> Episode -> Hypothesis -> Evidence/Correction -> Consolidation -> ReusableMemory`. Owner correction is a privileged evidence type: when unambiguous and in scope it immediately supersedes the active run/scene belief, while preserving both old belief and correction event. It may bypass repetition but never bypass domain validation or authority checks.

### 14. Post-session consolidation

**Existing coverage.** Stream summaries, chatter summaries, rolling summaries, session primer records, and memory extraction persist session material.

**Gap.** Summary generation is broader than change detection, and outputs do not share an idempotent delta/audit format.

**Migration direction.** `SessionConsolidator` reads a closed session's evidence and the pre-session state, proposes only typed deltas, validates them, and commits accepted deltas transactionally. Supported delta domains are game, social, Leo preference, schedule, Hebe belief, thread, and culture. “No useful change” is a valid and preferred result.

### 15. Forgetting and temporal relevance

**Existing coverage.** Pending-turn TTLs, expiring facts, recent-window retrieval, cache freshness, schedule state, scene versions, and active flags provide isolated decay mechanisms.

**Missing.** Relevance, validity, archival status, and deletion are not separated. No common policy explains why an item stopped appearing.

**Migration direction.** Use four independent concepts: `valid_until` (world truth), `relevance_until` (active retrieval), `status` (epistemic lifecycle), and retention policy (physical storage). Decay lowers retrieval priority or archives; it does not silently change observed history. Persistent owner preferences do not decay automatically. Sensitive/low-value social material may be deleted under retention policy.

## B. Recommended target architecture

### Components and ownership

| Component | Owns | Must not own |
|---|---|---|
| `ExperienceTimelineRepository` | Canonical normalized event projection and links to raw evidence | Source ingestion, beliefs, action success |
| `ConversationContinuityService` | Active participants, attention, turn handoff, expected reply, closure | Capability-specific execution |
| `OpenThreadRepository` | Cross-turn unresolved continuity and lifecycle | Full transcripts or arbitrary LLM notes |
| `SceneProjector` | Current scene materialized from evidence and active beliefs | Raw evidence mutation |
| `BeliefRepository` | Typed beliefs, evidence, supersession, validity | Inferring without evidence; action receipts |
| `MemoryRetrievalCoordinator` | Purpose/scope-based retrieval across typed stores and RAG | Writing facts during retrieval |
| `GameContextResolver` | Ordered scene/run/knowledge/RAG/research resolution | Independent web calls by downstream consumers |
| `SocialWorldService` | Minimal identities, episodes, hypotheses, social threads | Exhaustive profiles or hidden objective judgments |
| `SharedCultureService` | Evidence-backed culture items, use policy, reaction/cooldown | Hard-coded viewer catchphrases |
| `HistoricalActionLedger` | Readable receipt-backed action history | Executing or declaring success itself |
| `HebeSelfModelService` | Stable-core view plus validated evolving beliefs | Modifying core identity from model output |
| `SessionConsolidator` | Idempotent proposed and accepted state deltas | Bulk summarization or direct unvalidated writes |
| `TemporalRelevanceService` | Retrieval decay, expiry, archival/supersession jobs | Deleting authoritative receipts/evidence |

### Runtime read model

`ContextBuilder` should eventually receive a `ContinuityContext`, not a larger undifferentiated prompt dump:

```text
ContinuityContext
  conversation: CurrentConversation?
  scene: CurrentScene
  open_threads: ranked, scope-filtered subset
  game: GameRunContext + selected GameKnowledge claims
  social: selected episodes/hypotheses with confidence labels
  self: StableHebeCore + selected evolving preferences
  action_evidence: only relevant ledger receipts
  provenance_manifest: IDs/status/age for every supplied claim
```

The synthesizer receives compact rendered views, while validators retain the structured objects. Prompt rendering may change wording, never lifecycle or authority.

### Attention and turn-handoff algorithm

For STT input, evaluate in this order:

1. Preserve source and owner authority checks.
2. If the wake name is present, acquire attention and optionally start a conversation.
3. Otherwise load a non-expired current conversation for the same context.
4. Accept the utterance as addressed to Hebe only if the turn is owned by Leo and the reply is compatible with `expected_reply` (yes/no, entity selection, value, free response), or if a high-confidence direct continuation rule applies.
5. Reject/ignore unrelated ambient speech; do not use an LLM guess alone to seize it.
6. Consume or advance the turn atomically. Close on success, cancellation, incompatible interruption, timeout, scene boundary, or explicit handoff elsewhere.

This makes “sí” in the Ivanxi clarification generic while retaining current wake and firewall safety.

## C. Data model changes

### Reuse without replacement

- Keep `chat_log`, `stream_chat_messages`, and `stream_events` as raw/transcript sources.
- Extend `live_session_timeline`; treat it logically as the canonical experience timeline for stream and private contexts. Do not copy all old events into a new parallel event table.
- Keep `memory_chunks` for semantic retrieval and `memory_facts` as a compatibility structured store.
- Keep all promotion tables, receipts, and viewer promotion profiles authoritative for their domain.
- Keep game dossiers/cache/gaps and expose them through repository adapters.
- Keep schedule observations/hypotheses and align their transitions with the common belief vocabulary.
- Keep stream/session summaries as archived projections, not truth sources.

### Extend existing schemas

`live_session_timeline` additive columns:

- `context_kind` (`private`, `stream`, `system`)
- `source_record_type`, `source_record_id`
- `authority`
- `evidence_json` (literal spans/transport metadata; never semantic replacement)
- `valid_from`, `valid_until`
- `supersedes_event_id`
- `schema_version`

`memory_facts` additive compatibility columns:

- `belief_id`, `source_event_id`, `last_confirmed_at`, `valid_from`, `valid_until`
- `epistemic_status`, `superseded_by`, `owner_confirmed`, `schema_version`

New v2 writers use `BeliefRepository`; legacy facts are backfilled and remain readable during migration. `memory_chunks` should gain optional `belief_id`/`episode_id`, not become the canonical fact store.

`chatter_profiles`/`chatter_facts`:

- Add stable `person_id` and evidence/belief links.
- Treat existing aggregate profile fields as operational projections.
- Stop adding new untyped inferred facts once SocialWorld is active; migrate useful records through reviewable confidence/retention rules.

Game JSON tables:

- Add schema version and canonical game/run identifiers.
- Keep dossiers as materialized game views while claim-level knowledge migrates to normalized facts.
- Stop equating `stream_session_id` with a durable run.

### New models genuinely required

#### `conversations`

`id`, `context_kind`, `context_id`, `participant_ids_json`, `attention_state`, `turn_owner`, `expected_reply_type`, `expected_reply_schema_json`, `topic`, `opened_at`, `last_turn_at`, `expires_at`, `status`, `closure_reason`, `origin_event_id`, `version`.

Only one active conversation per mutually exclusive context, enforced by repository transaction. A conversation can own several sequential expected replies but each handoff is versioned.

#### `open_threads`

`id`, `thread_type`, `scope_kind`, `scope_id`, `participant_ids_json`, `subject_ref`, `summary`, `origin_event_id`, `latest_event_id`, `status`, `priority`, `created_at`, `relevance_until`, `valid_until`, `resolved_at`, `resolution_event_id`, `sensitivity`, `version`.

Statuses: `OPEN`, `WAITING_ON_LEO`, `WAITING_ON_HEBE`, `SNOOZED`, `RESOLVED`, `EXPIRED`, `ARCHIVED`.

#### `beliefs` and `belief_evidence`

`beliefs`: `id`, `namespace`, `scope_kind`, `scope_id`, `subject_ref`, `predicate`, `object_json`, `epistemic_status`, `confidence`, `authority_class`, `created_at`, `last_confirmed_at`, `valid_from`, `valid_until`, `relevance_until`, `superseded_by`, `owner_confirmed`, `sensitivity`, `schema_version`.

`belief_evidence`: `id`, `belief_id`, `source_event_id`, `source_record_type`, `source_record_id`, `relation` (`SUPPORTS`, `CONTRADICTS`, `CORRECTS`), `weight`, `observed_at`, `extractor`, `extractor_version`, `literal_span_json`.

Uniqueness is domain-specific (`namespace + scope + subject + predicate + active validity`), not a lossy global text hash. Evidence rows are append-only. Supersession is transactional.

#### Game run and knowledge tables

- `game_runs`: durable run ID, game ID, owner, rules/challenge, status, start/end, current checkpoint version.
- `game_run_sessions`: run-to-stream-session link.
- `game_run_events`: typed event/belief reference for progress, job roll, boss, objective, choice, resource state.
- `game_knowledge_facts`: normalized validated reusable claim, provenance, spoiler class, source quality, validity/status, dossier link.

The existing dossier remains a cached projection assembled from `game_knowledge_facts`; the existing progress JSON remains a compatibility snapshot assembled from active run beliefs/events.

#### Minimal SocialWorld tables

- `people`: opaque stable internal ID and minimal channel-scope metadata.
- `person_identities`: platform, stable platform user ID, current display/login, aliases, first/last seen.
- `social_episodes`: salient event bundle, participant IDs, tone/reaction observations, relevance/retention, source IDs.
- Social interpretations use `beliefs(namespace='social')`; do not add objective personality columns.
- `shared_culture_items`: label/meaning, origin episode, participants, tone, state, confidence, last used, cooldown, reuse count.
- `shared_culture_evidence`: episode/event, observed reaction, polarity, weight.

#### `action_ledger`

`id`, `request_event_id`, `plan_id`, `capability`, `domain`, `domain_transaction_id`, `receipt_id`, `target_ref`, `status`, `requested_at`, `attempted_at`, `confirmed_at`, `external_reference`, `result_digest`, `schema_version`.

Rows are append-only status events or an immutable record plus append-only transitions. `SUCCEEDED` requires a validated receipt/external confirmation. Promotion data is referenced, not duplicated as a competing transaction source.

#### Consolidation audit

- `consolidation_runs`: session/context, input watermark, pre-state version, consolidator version, status, started/completed.
- `consolidation_deltas`: typed proposal, evidence IDs, validator result, committed object reference, idempotency key, rejection reason.

### Schema migration and versioning

The current code relies mostly on `CREATE TABLE IF NOT EXISTS` and `ensure_column`. Before v2 data, add a small `schema_migrations(component, version, checksum, applied_at)` runner and per-record `schema_version` where JSON is stored.

Migration rules:

1. Migrations are ordered, transactional where SQLite permits, checksum-verified, restart-safe, and tested against copied old databases.
2. Add tables/columns/indexes first; never rename/drop in the same release that introduces v2 readers.
3. Backfills use stable idempotency keys and preserve old source IDs.
4. Dual-read with comparison telemetry precedes read cutover. Dual-write is used only at explicit adapters, never independently throughout business logic.
5. Failed validation leaves a proposal/rejection audit, not a half-written belief.
6. Destructive cleanup occurs only in Phase 6 after backup, replay parity, and at least one release of v2-only reads.

## D. Incremental implementation plan

The requested phase order is sound, but repository inspection of the Replay Lab shows that a verification foundation is required before Phase 1. Add Phase 0.5 below; it does not change the dependency order of Phases 1–6.

### Phase 0.5 — Cognitive Replay & Verification Harness

The current `StreamReplayLab` is a useful deterministic event runner, result container, I/O recorder, research-fixture provider, and version comparator. It is **not currently an end-to-end Hebe replay harness**: callers supply an arbitrary processor, and the repository has no processor that dispatches replay events through Hebe's production STT, Twitch, lifecycle, action, persistence, restart, or consolidation seams. Current replay tests use a toy callback. The separate developer simulators reach selected production handlers, but use wall-clock time, deliberately suppress/skip some real persistence and actions, and cannot orchestrate restarts or multiple sessions.

Phase 0.5 must therefore precede conversation continuity implementation:

1. Define a versioned scenario schema and canonical replay event vocabulary.
2. Add a production replay adapter that enters the same normalized ingress functions used by live STT, Twitch chat/EventSub, metadata, and lifecycle events.
3. Inject one controllable clock into all continuity-relevant TTL, cooldown, session, memory, promotion, scene, and research code reached by scenarios.
4. Add an isolated scenario workspace with temporary SQLite databases, restartable engine/runtime factories, and the same repository migrations as production.
5. Replace only external boundaries—Twitch network, TTS/audio, desktop, web research, and model calls—with deterministic fakes. Run real firewall, router, conversation, memory, plan, domain transaction, receipt, and final-emission code.
6. Add configured fake action outcomes that generate production-shaped receipts while recording attempted Twitch actions.
7. Add structured state probes and assertions for runtime projections and persisted tables.
8. Generate JSON plus Markdown verification reports from every run.

Phase 0.5 exit criteria are defined in the replay assessment and verification contract below. No Phase 1 feature should be described as verified until the relevant end-to-end deterministic scenario enters canonical production paths and passes across restart where applicable.

### Phase 1 — conversation continuity and generic OpenThreads

1. Add migration runner, feature flags, typed models, repository contracts, and replay observability. No routing change by default.
2. Build `ConversationContinuityService` in shadow mode from current pending structures and `LastHebeUtterance`.
3. Add `conversations` and `open_threads`; project current capability pending tasks into them and compare match/expiry/consume decisions.
4. Route wake-free compatible replies through the service behind a source-scoped flag, first for owner STT clarifications. Keep existing pending dictionaries as write-through compatibility projections.
5. Generalize yes/no, entity choice, value, correction, and free-response schemas. Capability handlers receive a resolved continuation payload, not raw pending dictionaries.
6. Expand to UI and stream contexts only after ambient-speech false-positive replays pass.

Exit criteria: Ivanxi-style clarification works without a second wake word; unrelated ambient “sí” does not; expiry/interruption/consumption is atomic; current tests remain green.

### Phase 2 — evidence timeline and Memory/Belief v2

1. Extend `live_session_timeline` and create the timeline adapter over raw chat/Twitch/private inputs.
2. Add beliefs/evidence and implement deterministic status transitions and owner-correction authority.
3. Shadow-project existing `memory_facts`, scene annotations, schedule hypotheses, and selected chatter facts; report mismatches.
4. Introduce `MemoryRetrievalCoordinator` with scope, purpose, epistemic status, age, sensitivity, and provenance ranking.
5. Render structured claim manifests into current `BuiltContext` without widening action authority.
6. Cut writes for selected low-risk namespaces to beliefs, maintaining `memory_facts` compatibility rows.

Exit criteria: every v2 claim has evidence or an explicit authoritative source; correction/supersession is auditable; vector chunks cannot overwrite belief truth.

### Phase 3 — GameKnowledge, GameRun, and memory-first research

1. Inventory and nominate `game_intelligence.py` as the principal implementation seam because it already owns gaps, research jobs, evidence, advice, and persistence.
2. Place adapters around `game_knowledge.py`, `game_research.py`, `game_profiles.py`, `cognitive/game_guidance.py`, and `session_primer.game_sessions`; do not delete them yet.
3. Add durable game/run identity and map existing sessions/progress snapshots to runs.
4. Implement the single ordered `GameContextResolver` and forbid direct provider calls outside GameResearch.
5. Persist validated researched claims into `game_knowledge_facts`; rebuild dossier/profile projections from them.
6. Teach owner correction and explicit run statements (jobs rolled, objective, progress) to update run beliefs immediately with evidence.

Exit criteria: known knowledge does not trigger broad research; separate runs of the same game remain distinct; spoiler/evidence guards pass unchanged; research replay is fixture-only and deterministic.

### Phase 4 — SocialWorld, continuity, and SharedCulture

1. Introduce stable people/identity aliases using Twitch stable IDs where available; backfill without merging ambiguous identities.
2. Define salience, sensitivity, and retention policies before enabling writes.
3. Convert selected chatter observations into social episodes and selected interpretations into labeled hypotheses.
4. Use `open_threads` for short-lived wellbeing/exam/follow-up continuity with explicit relevance windows.
5. Add SharedCulture candidate, reinforcement, cooldown, negative reaction, and retirement workflows.
6. Project owner-authorized promotion preferences and recent interaction receipts by reference; do not duplicate their authority.

Exit criteria: observations and hypotheses are distinguishable in storage and language; expired personal follow-ups stop active retrieval; no per-viewer catchphrase rule is introduced.

### Phase 5 — consolidation, self model, and temporal supersession

1. Separate StableHebeCore, HebeVoice, LeoLanguageModel, and SharedCulture inputs while preserving current rendered voice behind a compatibility snapshot.
2. Implement `SessionConsolidator` in proposal-only mode and compare deltas with human-reviewed real-stream fixtures.
3. Enable idempotent commits one domain at a time: game, schedule, threads, owner preferences, social, Hebe evolving beliefs.
4. Add temporal relevance jobs for expiry, archive, weakening, and supersession; never physically delete authoritative evidence here.
5. Add scene/action consequence reducers, beginning with successful raid -> ending/social transition.

Exit criteria: rerunning consolidation produces no duplicate delta; “nothing changed” is common; owner preferences persist; old schedules become historical rather than silently rewritten; StableHebeCore cannot be mutated.

### Phase 6 — legacy cleanup and database hygiene

1. Cut remaining readers from pending dictionaries, wide duplicated scene fields, legacy fact writes, and overlapping game resolvers after telemetry proves parity.
2. Rename confusing service modules/interfaces with compatibility imports for one release.
3. Archive or migrate useful legacy rows; validate foreign references and orphan counts.
4. Remove dormant code only after import/runtime scans and replay coverage prove it is unused.
5. Backup and compact SQLite after migration verification; add documented restore and downgrade boundaries.

Exit criteria: one owner exists for each concern, no compatibility write path remains, schema migration inventory matches all supported databases, and removal has an explicit rollback release/tag.

## E. Risk analysis

| Area | Risk | Control and release gate |
|---|---|---|
| STT routing | Generic continuation captures ambient speech or another speaker's reply | Deterministic source/participant/turn/expiry checks; shadow decisions; real-stream false-positive fixtures; fail closed to wake-word behavior. |
| Twitch authority | Learned preference or social thread becomes a command | Preserve source authority in the event and context; beliefs never synthesize owner authority; capability tests across owner/mod/viewer/system. |
| Action execution | A planned or conversational action is remembered as successful | Ledger success only from receipt boundary; reconciliation tests for timeout/unknown/failure/retry; no memory-to-ledger writer. |
| Stream output | New scene/culture path bypasses final emission or duplicates speech | All proposals stay upstream of `FinalEmissionGate`; dedupe/idempotency tests; kill switch per proposal source. |
| Memory persistence | Dual writes diverge, migrations corrupt old DBs, or inference hardens into fact | One adapter transaction; checksummed migrations; copied-DB tests; evidence/status required; comparison telemetry and backup. |
| PresenceEngine | More threads create excessive interruptions | Presence remains a ranked opportunity source; existing cooldown/budget/safety gates remain; thread relevance is not emission permission. |
| Game commentary | Stale run state, spoilers, or unsupported researched facts reach speech | Run IDs, validity, citation/spoiler classification, existing GameAdvice/Evidence guards, fixture providers, abstention on conflict. |
| Social memory | Surveillance-like accumulation, sensitive recall, or false social claims | Minimal schema, salience/retention/sensitivity filters, confidence language, provenance, no personality truth fields, expiry and deletion policy. |
| Self model | Voice drift or imitation of Leo changes Hebe's identity | Immutable stable core, domain allowlist for evolving beliefs, owner authority, bounded renderer, snapshot/replay voice tests. |
| Consolidation | Model hallucinates deltas or repeats writes | Proposal/validator/commit split, evidence IDs mandatory, per-domain schemas, idempotency keys, dry-run review, “no delta” default. |
| Scene projection | Reducer overwrites literal evidence or stale facts survive transitions | Append-only evidence, versioned reducers, belief validity, scene-version guards, no destructive source update. |

Operationally, each phase needs a global v2 read flag plus narrower flags for continuation, belief writes, game resolver, social writes, consolidation commits, and scene consequences. Disable should fall back to the existing path without deleting v2 data.

## F. Compatibility strategy

Use adapters at current seams rather than scattering conditionals:

- `LegacyPendingAdapter`: typed conversation/thread <-> `pending_clarification` and `pending_conversation_turn`.
- `LegacySceneAdapter`: `CurrentScene` -> existing live/stream state snapshot fields.
- `LegacyMemoryFactAdapter`: v2 belief -> compatibility `memory_facts`; old fact -> shadow belief proposal.
- `LegacyGameAdapter`: GameRun/Knowledge views -> `GameProgressState`, `GameRunState`, dossier/profile, and session-primer shapes.
- `LegacySocialAdapter`: people/episodes -> current chatter lookup payloads without exporting hypotheses as facts.
- `ReceiptLedgerProjector`: existing domain receipts -> read-only ledger projection.

Migration pattern for each seam:

```text
observe old behavior
  -> compute v2 shadow result
  -> compare and record reasoned diff
  -> dual-write through one adapter
  -> canary v2 reads with legacy fallback
  -> v2 primary / legacy shadow
  -> stop legacy writes
  -> remove after one compatibility release
```

Compatibility projections are intentionally lossy toward v1; v2 remains complete. A rollback can use old readers because required v1 rows/snapshots continue to be written until the cleanup phase.

## G. Testing strategy

### Unit tests

- Conversation lifecycle, participant/source matching, expected-reply compatibility, timeouts, closure, interruption, and optimistic version conflicts.
- Belief status transitions, confidence versus status, evidence requirements, owner correction, contradiction, validity, supersession, and retrieval decay.
- Scene projection keeps raw evidence and referents across semantic reclassification.
- Thread ranking, sensitivity, expiry, resolution, and cross-session reopen behavior.
- Game resolution order, gap keys, cache reuse, run separation, spoiler filtering, citation validation, and abstention.
- Social observation/hypothesis separation, identity alias ambiguity, salience/retention, SharedCulture reinforcement/cooldown/negative reaction.
- Receipt-ledger rules for requested/attempted/failed/unknown/succeeded.
- StableHebeCore immutability and evolving-layer allowlist.
- Consolidation validation and idempotency.

### Conversation-continuity fixtures

At minimum:

1. “Hebe, haz una promo a Ivanxi” -> Hebe entity clarification -> “sí” succeeds without wake.
2. Same flow after TTL -> “sí” is not captured.
3. Hebe asks a yes/no question, unrelated TV/ambient “sí” with incompatible source/context -> ignored.
4. A second owner command interrupts and closes/suspends the old thread deterministically.
5. Two candidate entities require a selection; a non-matching answer does not execute.
6. UI and STT conversations do not accidentally share current turns.

### Replay tests

Use `backend/app/stream/replay.py`, `ReplayIOBoundary`, and fixture research providers. Convert selected real-stream JSONL/events into versioned golden fixtures covering:

- the 2026-08-09 temporal schedule correction and scene evolution;
- live research timeout/retry/partial completion;
- raid/promotion transactions and receipts;
- owner voice correction during a game run;
- spontaneous opportunity and final-emission dedupe;
- chatter follow-up whose relevance expires next session.

Replay comparisons should assert decisions, state deltas, evidence IDs, status transitions, emitted output count/target, and absence of external side effects—not exact prose unless testing a guarded template.

### Database migration tests

- Fresh database reaches latest schema.
- Representative pre-v2 databases migrate and reopen.
- Each migration is idempotent after interruption/restart.
- Backfill preserves source IDs, timestamps, Unicode, JSON, and active/inactive semantics.
- Unknown future schema versions fail safely.
- Rollback reader can use compatibility projections.
- Foreign-key/orphan, duplicate idempotency key, and index/query-plan checks.

### Cross-session scenarios

- An FFV job roll stated by Leo persists in the same run next stream but not a new run.
- Persona 5 progress is independent from general Persona 5 knowledge.
- A corrected crystal number supersedes the old belief immediately and consolidation does not restore it.
- Repeated Monday observations strengthen a schedule hypothesis; the old schedule becomes historical.
- Yesterday's illness opens a short social thread, permits one appropriate follow-up, then expires from active retrieval.
- Owner raid wording preference persists across sessions and prevents automatic viewer-count language.
- Validated game research is reused without a new web call; a genuinely missing typed gap may research once.
- A failed or unknown action is never narrated as completed; a successful receipt can be recalled.

### Non-functional gates

- Measure p50/p95 context-build latency, SQLite lock time, retrieval size, research calls, and emitted proposals.
- Fuzz malformed evidence/model proposals and corrupted/old JSON schema versions.
- Verify privacy scope: private memories do not leak to Twitch; social hypotheses are rendered with uncertainty.
- Run existing cognitive, stream, promotion, presence, guidance, research, and replay suites unchanged before enabling any feature flag.

### Existing Replay Lab capability assessment

#### Overall verdict

The current infrastructure cannot reliably validate Phases 1–5 offline as an integrated system. It provides good low-level building blocks but no canonical production `ReplayProcessor`.

`backend/app/stream/replay.py` currently provides:

- generic, timestamp-ordered `ReplayEvent` values;
- accelerated, real-time, step, shadow, and compare modes;
- deterministic fingerprinting and golden serialization;
- a replay-local `simulated_time` value;
- an arbitrary runtime-state snapshot after each event;
- a fail-closed research fixture provider;
- a side-effect recorder that blocks Twitch, desktop, and TTS calls.

However, `StreamReplayLab._process_event` invokes the caller-supplied processor directly. There is no repository implementation connecting it to `HebeEngine._process_stt_voice_transcript`, the live Twitch callback/`handle_twitch_chat_event`, `process_internal_event`, metadata polling, lifecycle handling, executors, or persistent repositories. In `backend/tests/test_hebe_live_v1.py`, `ReplayLabTests.processor` only assigns `runtime.state.last_event` and returns a constructed `ReplayDecision`. Thus an event named `stt` or `stream_online` currently has no production meaning inside Replay Lab.

The developer simulation methods in `backend/app/hebe_engine.py` are a separate partial facility:

- `simulate_leo_message(..., source='stt_voice')` enters `_process_stt_voice_transcript` and therefore exercises real wake/firewall/routing logic.
- `simulate_ambient_stt` enters the real ambient classification/context path.
- `simulate_twitch_message` calls `process_internal_event(twitch_chat_react)`, but bypasses the real chat callback and `handle_twitch_chat_event`; this omits portions such as automatic-promotion observation and raw IRC normalization.
- `simulate_internal_twitch_event` calls the real internal-event path, but supplies one fixed raider-like payload for every Twitch event type.
- simulation uses `time.time()`/`datetime.now()`, temporarily changes live flags, marks payloads `_simulated`, blocks some actions, and intentionally skips creation of real stream sessions/chat persistence in `backend/app/stream/memory.py`.

Those simulators are valuable ingress tests; they are not a deterministic multi-event, persistent replay lab.

#### Capability matrix

The status below is intentionally strict. “The event can be put in a generic `ReplayEvent` payload” does not count as support unless the harness dispatches it through the system being verified.

| Capability | Status | Existing implementation responsible | Assessment |
|---|---|---|---|
| Owner STT utterances | **PARTIAL** | `HebeEngine.simulate_leo_message`; `_process_stt_voice_transcript`; generic `ReplayEvent` | The dev simulator reaches the canonical post-transcription STT path. Replay Lab has no adapter, injected clock, scenario persistence, or restart. Audio/faster-whisper itself is outside the text replay boundary and needs separate integration fixtures. |
| Wake and non-wake speech | **PARTIAL** | `simulate_leo_message`, `simulate_ambient_stt`, wake resolver and STT gate in `hebe_engine.py` | Individual inputs exercise real wake classification. A timed conversational sequence cannot be deterministically replayed through the lab. |
| Twitch viewer messages | **PARTIAL** | `simulate_twitch_message`; `process_internal_event`; real callback and `handle_twitch_chat_event` | Simulation enters the canonical internal cognitive path but bypasses earlier real chat ingress, chat observation, raw IRC parsing, and automatic-promotion observation. |
| Replies and mentions | **PARTIAL** | `_twitch_reply_metadata`, `_twitch_direct_priority`, conversation ownership gate; pass-through simulator payload | Fields can be manually supplied and downstream logic can classify them, but there is no typed fixture/event contract proving parity with IRC tags or emitted-message reply IDs. |
| Follows | **PARTIAL** | `simulate_internal_twitch_event`; `process_internal_event`; Twitch event callback | Reaches the post-normalization production event handler, but uses an unrealistic fixed payload and lacks deterministic persistence/output assertions. |
| Subs and resubs | **PARTIAL** | Same internal-event simulator and `process_internal_event` | Sub event labels can be dispatched, but resub-specific fields and identity/history are not modeled by the harness. |
| Raids | **PARTIAL** | `simulate_internal_twitch_event`; `_handle_twitch_raid_event`; raid dedupe | Reaches the production raid handler with a fixed payload. It cannot presently configure/inspect a real receipt-backed action sequence or durable scene transition. |
| Twitch title/game changes | **MISSING** | Real stream-context polling and `LiveSessionBrain.observe_stream_metadata` exist outside replay | No replay event adapter applies metadata through the same production update/poll path. Direct state assignment would be insufficient. |
| Stream start/end | **MISSING** | `_handle_stream_lifecycle_event` handles `stream_online`/`stream_offline` in production | Generic replay fixtures use the labels, but no replay processor calls this handler. The dev Twitch-event endpoint rejects non-`twitch_` event types. |
| Passage of time | **PARTIAL** | `ReplayRuntime.simulated_time`; event ordering; optional real-time sleep | Replay-local time is updated, but production TTL/cooldown code mostly calls wall-clock functions. Sleeping is not deterministic verification. |
| Multiple distinct stream sessions | **MISSING** | `LiveSessionStateManager` can begin/reset a session | Replay resets state once at run start; it does not interpret lifecycle boundaries as multiple sessions or preserve selected state between them. |
| Hebe process restart | **MISSING** | No Replay Lab restart primitive | No engine disposal/recreation or repository reconnection exists in a scenario. |
| Persistent DB state across sessions | **MISSING** | Production SQLite repositories exist; simulation memory guards skip simulated stream/session writes | Replay defaults to `SimpleNamespace`, has no isolated DB workspace, migration lifecycle, or durable-state probe. |
| Action receipts | **MISSING** | `ReplayDecision.action_type/action_status`; `ReplayIOBoundary`; production promotion receipts elsewhere | A processor may assert arbitrary status, but the replay boundary returns failed/blocked booleans and is not connected to the actual executor/transaction receipt path. This cannot prove action truth. |
| Social events | **PARTIAL** | `simulate_internal_twitch_event`, `social_events.py`, internal Twitch pipeline | Selected normalized event labels can reach production handlers, but payload fidelity, multi-event identity, persistence, time, and final social-state inspection are absent. |
| Game-run events | **PARTIAL** | `simulate_ambient_stt`, voice relevance/run-context updates, Game Intelligence unit tests | Individual owner speech can update current in-memory run context. No durable cross-session run identity or canonical scenario adapter exists. |
| Memory consolidation | **MISSING** | Existing summaries/extractor operate independently; v2 consolidator does not yet exist | Replay has no session-close consolidation hook, deterministic model fixture, proposal audit, or delta assertions. |
| Memory decay/supersession | **MISSING** | Isolated TTL/status mechanisms exist | Production time is not controlled and there is no replay maintenance event or final belief-lifecycle probe. |
| Inspection of final cognitive/world state | **PARTIAL** | `ReplayResult.state_snapshots`; `_simulation_debug_payload` | Snapshots serialize only the arbitrary runtime state object; simulation debug exposes a hand-picked slice. Persistent beliefs, threads, social state, receipts, DB watermarks, and provenance are not jointly inspected. |

#### Canonical-path parity requirements

The Phase 0.5 adapter must not reproduce production logic in a replay-only processor. It should call shared ingress seams also called by live adapters:

| Scenario event | Required shared production seam |
|---|---|
| `owner_stt` | The post-faster-whisper normalized transcript ingress that calls the same `_process_stt_voice_transcript` logic. Keep separate audio/STT-engine tests for transcription quality. |
| `ambient_stt` | The same voice classification, firewall, evidence recording, and scene/run observation path as live ambient transcripts. |
| `twitch_chat` | A factored normalized Twitch chat ingress used by both the real IRC callback and replay; it must include bot/self filtering, firewall, automatic-promotion observation, chat persistence, reply metadata, and internal dispatch. |
| `twitch_follow/sub/resub/raid/cheer` | `process_internal_event` with production-shaped normalized EventSub/IRC payloads and stable IDs. Payload builders should be shared with real adapter normalization tests. |
| `stream_metadata_changed` | The same metadata application and `LiveSessionBrain.observe_stream_metadata` seam used after real Twitch polling/EventSub updates. |
| `stream_started/stream_ended` | `process_internal_event` -> `_handle_stream_lifecycle_event`, including session open/close and consolidation hook. |
| `advance_time` | An injected `Clock` advanced without dispatching a fake user event; due expiry/maintenance jobs run through their production scheduler seam. |
| `restart_hebe` | Close engine/repositories, clear volatile process objects, create a new engine against the same scenario DB, and run normal state restoration. |
| `action_outcome` | Configure the fake external adapter's next outcome; the owner command still traverses firewall/router/plan/domain executor and writes a production-shaped receipt. |

If live and replay currently cannot share a seam cleanly, first extract a common ingress function and make both adapters call it. Do not call a private downstream reducer merely to make the replay pass.

#### Phase 0.5 harness design

`CognitiveReplayScenario` should be a versioned JSON/YAML document:

```yaml
schema_version: 1
scenario_id: ivanxi_resub_promo_restart
initial_time: 2026-08-11T18:00:00+02:00
external_outcomes:
  twitch.shoutout: [{success: true, message_id: replay-so-1}]
model_fixtures:
  promotion_clarification: {...}
events:
  - {at: +0s, type: stream_started, session_id: stream-1}
  - {at: +2s, type: twitch_resub, user_id: '42', login: ivanxi}
  - {at: +5s, type: twitch_chat, user_id: '42', login: ivanxi, text: hola}
  - {at: +10s, type: owner_stt, text: 'Hebe, hazle una promo a Ivanxi.'}
  - {at: +12s, type: owner_stt, text: 'sí'}
  - {at: +10m, type: stream_ended, session_id: stream-1}
  - {at: +11m, type: restart_hebe}
  - {at: +1d, type: stream_started, session_id: stream-2}
assertions:
  - {after_event: 4, path: conversation.expected_reply_type, equals: confirmation}
  - {after_event: 5, path: action_ledger[0].status, equals: SUCCEEDED}
```

Exact fixture syntax can differ, but scenarios need stable event IDs, absolute/relative time, source identity/authority, production-shaped payloads, configured external outcomes, model/research fixture keys, checkpoint assertions, final assertions, and a declared schema version.

The scenario workspace owns:

- an isolated temporary directory and explicit SQLite paths;
- schema migration setup and a copied-old-DB option for migration scenarios;
- an injected `ScenarioClock` used for wall time and scheduler time;
- deterministic IDs or an ID recorder so reruns compare stable semantic output;
- a restartable engine factory using the same repositories and restoration path as production;
- fixture model/research providers that fail on unknown calls;
- recording Twitch/TTS/desktop adapters that never access the network but return configured production-shaped results;
- a `CognitiveStateProbe` that reads runtime plus repositories without mutating them.

State probes must expose, at minimum:

- current conversation and turn ownership;
- open threads and lifecycle;
- current scene assertions with evidence IDs;
- active GameRun and selected GameKnowledge;
- social people/episodes/hypotheses and SharedCulture items;
- learned beliefs/preferences with status, confidence, validity, and provenance;
- authoritative ledger entries and underlying receipt references;
- attempted/emitted Twitch operations and FinalEmissionGate results;
- final Hebe response and output targets;
- consolidation runs/deltas and DB schema/data watermarks.

Model determinism is essential. Structured fixture responses should be keyed by semantic call purpose and schema version rather than call order alone. The harness may optionally capture a real model interaction for later redacted replay, but standard verification must not require network/model availability. Exact natural-language output should only be asserted for deterministic guarded templates; most scenarios should assert speech act, referenced evidence, forbidden claims, routing, and state change.

#### Required Phase 0.5 acceptance scenarios

1. **Ivanxi resub/promotion/restart:** the nine-step scenario from the requirement, including a wake-free “sí”, one successful configured Twitch action, a persisted receipt/ledger entry, closed conversation, stream closure, restart, and later-session state inspection.
2. **Ambient false positive:** an active yes/no handoff followed by incompatible ambient/non-owner speech; no continuation or action occurs.
3. **FFV durable run:** Leo states/rolls White Mage, unrelated events occur, later reference resolves to the active challenge run, stream closes, Hebe restarts, and the next FFV session restores White Mage without re-explanation.
4. **Owner correction:** scene/run belief says third crystal; Leo corrects it to second; old belief becomes superseded/historical immediately and stays corrected after restart/consolidation.
5. **Temporal social thread:** a viewer illness/exam episode opens a bounded thread, permits an appropriate later-session follow-up, then expires after clock advancement without erasing the episode.
6. **Research learning:** a typed game gap uses fixture research once, validates and persists the claim, then a later session resolves from GameKnowledge with zero new research calls.
7. **Receipt truth:** success, failure, timeout/unknown, and retry outcomes prove that only a successful receipt permits a “performed” claim.
8. **Raid transition:** a successful/observed raid transition closes the scene/session, records destination social context, emits at most one guarded farewell, and does not automatically mention viewer count.
9. **Consolidation idempotency:** session close proposes useful deltas, a second consolidation at the same watermark writes none, and low-value chatter produces no reusable memory.

Phase 0.5 is complete only when these scenario types can be expressed without per-scenario production code, the first two pass against existing behavior/baselines, engine restart genuinely reloads the same scenario DB, and the report identifies every path that remains mocked or unverified.

### Per-phase verification contract

Every implementation phase must include all six layers below. A layer may be marked not applicable only with a reason in the generated report.

1. **Unit tests.** Pure lifecycle, validation, reducer, ranking, authority, and repository behavior with boundary/error cases.
2. **Integration tests.** Real component wiring across canonical ingress, firewall/router, repositories, executor/receipt, and final emission; only external systems are faked.
3. **Deterministic replay scenarios.** Versioned scenario fixtures with checkpoint and final-state assertions, stable clocks/IDs/model fixtures, and no real external I/O.
4. **Cross-session/persistence scenarios.** Required whenever a phase writes durable state or has temporal lifecycle. At least one close/restart/reopen scenario must demonstrate restoration and expiry/supersession where relevant.
5. **Existing-behavior regression tests.** Run the affected existing suites plus explicit authority, STT false-positive, Twitch output, receipt, presence, and game-evidence guards. Baseline changes require an explained golden diff, never blind regeneration.
6. **Machine-generated verification report.** Produced from test/replay result files, not manually inferred from console snippets.

A phase is not verified merely because code compiles, unit tests pass, or a mocked reducer returns the expected structure. Minimum verification requires applicable integration and replay contracts to pass through canonical production seams. If the development environment cannot run an applicable layer, phase status is **implemented, verification incomplete**, not verified.

#### Machine-generated report contract

Emit both `verification-report.json` and a derived `verification-report.md` containing:

- repository commit/worktree identity, platform, Python/schema/scenario versions, feature flags, and deterministic seed;
- exact commands executed and exit codes;
- tests passed, failed, skipped, expected-failed, and duration by suite/layer;
- replay scenarios passed/failed and failed assertion paths;
- relevant before/after decisions and behavior changes, including baseline golden diffs;
- checkpoint and final cognitive/world-state projections where applicable;
- action attempts, receipts, ledger status, and final emissions;
- DB migration versions, restart boundaries, consolidation watermarks, and persistence assertions;
- fixture/model/research/external boundaries used;
- known limitations, unavailable dependencies, nondeterministic fields, and anything not verified in the environment;
- overall result: `VERIFIED`, `FAILED`, or `VERIFICATION_INCOMPLETE` with machine-evaluable reasons.

Secrets, raw sensitive social content, access tokens, and unrestricted transcripts must be redacted from reports. Store stable evidence IDs and minimal assertion excerpts instead.

#### Verification expectations by migration phase

| Phase | Mandatory offline contract beyond unit tests |
|---|---|
| 0.5 | Canonical-ingress parity tests; clock/restart/DB/action fake tests; Ivanxi and ambient-negative scenarios; report self-test. |
| 1 | Wake/non-wake continuation matrix, interruption/expiry, multiple sources, entity clarification, action receipt, and restart/thread persistence scenarios. |
| 2 | Evidence preservation, owner correction, contradiction/supersession, retrieval scope/privacy, DB backfill/restart, and old/new shadow parity scenarios. |
| 3 | FFV/P5 independent durable runs, memory-first hierarchy, research-once reuse, spoiler/evidence abstention, correction, and restart scenarios. |
| 4 | Stable identity aliases, observation versus hypothesis, bounded follow-up expiry, SharedCulture cooldown/negative reaction, privacy/retention, and promotion-profile reference scenarios. |
| 5 | Delta-only consolidation, no-change/idempotency, owner preference persistence, schedule weakening/supersession, decay/archive behavior, immutable StableHebeCore, and raid scene-transition scenarios. |

### Verification performed for this architecture assessment

Commands attempted on 2026-08-11:

```text
backend/.venv/Scripts/python.exe -m pytest ...
.venv/Scripts/python.exe -m pytest ...
```

Both environments lack the `pytest` module, so these commands could not run. The relevant tests use `unittest`, so they were then run with the backend virtual environment and `PYTHONPATH=backend`.

```text
backend/.venv/Scripts/python.exe -m unittest \
  backend.tests.test_hebe_live_v1.ReplayLabTests \
  backend.tests.test_cognitive_twitch.CognitiveTwitchTests.test_twitch_simulation_defaults_to_forced_live \
  backend.tests.test_cognitive_twitch.CognitiveTwitchTests.test_twitch_simulation_can_force_offline_or_use_real_state
```

Result: 10 tests passed. A combined first invocation also named two test classes incorrectly and therefore reported two loader errors; the corrected command below passed both selected tests.

```text
backend/.venv/Scripts/python.exe -m unittest \
  backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_simulated_owner_stt_monologue_is_context_only \
  backend.tests.test_ui_chat_message_envelope.UiChatMessageEnvelopeTests.test_dev_simulate_twitch_message_uses_engine_simulation
```

Result: 2 tests passed. These results verify the existing low-level runner behaviors and selected simulator ingress behavior only. They do **not** verify a multi-event cognitive scenario, persistence, restart, receipts, consolidation, or Phases 1–5. No real Twitch, external research, TTS, or desktop action was attempted.

### Human evaluation boundary

Automated verification should establish continuity prerequisites: correct attention/turn ownership, relevant evidence retrieval, temporal state, safe authority, action truth, bounded social memory, and exactly-once output. It must not claim to objectively score naturalness, comedic timing, personality, warmth, or social appropriateness. Those remain human review dimensions, supported by replay transcripts and state/provenance views rather than replaced by automated pass/fail metrics.

## H. Cleanup candidates (do not remove in this task)

1. The dual pending systems and promotion/appointment/game-specific continuation helpers after `ConversationContinuityService` owns all continuations.
2. Dormant `backend/app/orchestrator` routing/gates after import scans confirm no runtime dependency. It must never be reintroduced as a parallel cognitive owner.
3. The overlapping game knowledge/research/guidance entry points after `GameContextResolver` and repository adapters reach parity.
4. Duplicated run state across `GameRunState`, `game_progress_states`, `game_sessions`, live-session fields, and stream-state fields after durable run projections are primary.
5. Duplicated scene truth in wide snapshots and the in-memory scene manager after all readers use the versioned `CurrentScene` projection. Retain caches where they have a measured latency purpose.
6. Ambiguous module naming between `cognitive/memory_store.py` (structured facts/reminders) and `cognitive/memory/memory_store.py` (chunks/vector retrieval).
7. Direct `MemoryExtractor` upserts for namespaces moved to validated v2 belief writers.
8. Untyped `chatter_facts`/inferred-summary writes after useful data is migrated to observation plus hypothesis records.
9. Hard-coded evolving tastes, jokes, and channel culture in `hebe_voice.py` after equivalent validated data exists. Stable voice constraints and safe examples remain code/config.
10. Ad hoc `ensure_column` calls after all supported schemas are registered with the migration runner.
11. Compatibility columns, adapters, shadow telemetry, and legacy JSON snapshots only after the Phase 6 rollback window closes.

## Concrete first implementation commits

The smallest safe commit sequence following this report is:

1. Add schema migration infrastructure, v2 feature flags, models, and shadow-decision telemetry with no behavior change.
2. Add `conversations`/`open_threads` plus `LegacyPendingAdapter`, still shadow-only.
3. Add replay fixtures for wake-free compatible replies and ambient false positives.
4. Enable generic continuation for owner STT clarification behind a flag; preserve the current executor and emission route.
5. Extend the timeline and add belief/evidence repositories in shadow mode.

No game, social, self-model, or consolidation behavior should be changed before those commits establish the reusable evidence, thread, migration, and replay foundations.
