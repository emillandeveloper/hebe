# Phase 1D — Social identity canonicalization checkpoint

Status: implemented and verified. The Phase 1D working tree is intentionally
uncommitted. Phase 1E ToolSystem /
InteractionActions has not started.

## 1. Social model and table inventory

| Representation | Role and primary identity | Writer / reader | Lifecycle and persistence | Stable Twitch ID relationship | Duplication status |
|---|---|---|---|---|---|
| `people` | Provider-independent person container, keyed by `person_id` | `SocialWorldRepository` / `SocialWorldService` | Durable SQLite | Reached through `person_identities` | Canonical |
| `person_identities` | External identity, aliases and current presentation | SocialWorld repository/service | Durable and rename-aware | `platform='twitch' + platform_user_id` is the strongest Twitch key | Canonical |
| `person_sessions` | Per-person/session familiarity aggregate | SocialWorld presence write and retrieval | Durable session aggregate | Always attached through canonical `person_id` | Canonical |
| `social_presence_events` | Idempotent raw presence admissions | SocialWorld repository | Durable observation receipts | Carries stable identity indirectly through `person_id` | Canonical |
| `social_summaries` | Contextual per-person stream summaries | SocialWorld service/repository | Durable, person-scoped | Attached only after safe identity resolution | Canonical |
| `social_episodes` | Meaningful interactions such as raid/follow/sub | SocialWorld service/repository | Durable social history | Attached to canonical person | Canonical |
| `beliefs`, social namespaces | Hypotheses with provenance/confidence | Belief lifecycle through SocialWorld | Durable, lifecycle-managed | Scoped to person, never an identity key | Canonical knowledge |
| `open_threads`, person scope | Follow-up continuity | Continuity service / SocialWorld retrieval | Durable with expiry | Scoped to canonical person | Canonical continuity |
| shared-culture tables | Community references and reinforcement | SocialWorld culture methods | Durable community scope | Not an identity store | Canonical culture |
| `viewer_linguistic_profiles` | Specialized language/style preference | Viewer profile service | Durable specialized state | Stable ID when known; explicit `login:` placeholder when not verified | Active, not identity owner |
| `viewer_promotion_profiles` | Owner-controlled promotion automation | Promotion service | Durable operational config | Keyed by stable ID; owner-locked rows can be migration evidence | Active, not identity owner |
| `promotion_events` | Promotion action receipts | Promotion service | Durable operational history | References promotion profile | Active, not identity owner |
| `StreamSessionState` and live timeline | Current stream observations | Twitch/stream runtime | In-process plus observation timeline | Can transport identity evidence | Observation only |
| `stream_chat_messages` / stream events | Raw observed chat/events | Stream memory | Durable observation history | Tags may contain stable ID, but the row is not identity truth | Observation only |
| `stream_summaries` | Overall stream aggregate | Stream memory | Durable per-stream aggregate | None | Canonical stream aggregate; new rows do not embed viewer summaries |
| `chatter_profiles` | Username-keyed historical viewer profile | Migration only after cutover | Retained SQLite source | Does not contain stable IDs | Legacy historical |
| `chatter_presence` | Username-keyed historical session aggregate | Migration only after cutover | Retained SQLite source | No stable ID | Legacy historical |
| `chatter_facts` | Historical inferred chatter facts | No runtime writer/reader | Retained SQLite source | No stable ID | Legacy historical, not migrated automatically |
| `stream_chatter_summaries` | Username-keyed historical viewer summaries | Migration only after cutover | Retained SQLite source | Indirectly linkable only through audited profile mapping | Legacy historical |

`display_name` is presentation, `username/login` is a mutable alias, raw stream
data is observation, summaries are contextual projections, and beliefs require
their own admission lifecycle. None of those may independently establish
identity.

## 2. Canonical identity hierarchy

The enforced resolution order is:

1. Twitch stable user ID (`platform='twitch'`, `platform_user_id`);
2. another explicit `provider + provider_user_id` pair;
3. an equivalent canonical external identity with explicit provenance;
4. username/login as a mutable, unverified alias;
5. `display_name` as presentation only.

The resolver never merges two stable IDs because they currently share a login.
An unseen stable ID does not absorb a username-only person automatically. A
login-only observation can reuse exactly one unambiguous unverified identity,
but remains confidence `0.6`; multiple candidates fail closed with
`SocialIdentityConflict`.

For the legacy audit, owner-locked promotion identity is `VERIFIED_IDENTITY`.
One modern stable identity observed in the exact same stream event window is a
deterministic `STRONG_MATCH`. Name similarity, display name, co-occurrence or
Hebe inference is never sufficient.

## 3. Classification of the 39 historical profiles

All 39 source rows have username, display name and first/last timestamps; none
has a stable ID or useful stored alias. Interaction history is in
`chatter_presence`, not in the profile row. The table below uses irreversible
technical hashes. `Presence` and `summaries` are linked legacy-row counts;
`modern evidence` is the pre-migration number of matching modern people and
stable identities.

| Technical source | Classification | Presence | Summaries | Modern evidence | Decision |
|---|---:|---:|---:|---|---|
| `legacy_profile_2f43dd3ab571b2227015` | `ALREADY_CANONICAL` | 21 | 18 | 1 person / 1 stable | deduplicate into verified person |
| `legacy_profile_49d141991b92ef14cd44` | `ALREADY_CANONICAL` | 6 | 6 | 1 / 1 | deduplicate into verified person |
| `legacy_profile_60e46cdc297050ef8dc7` | `ALREADY_CANONICAL` | 1 | 0 | 1 / 1 | deduplicate into verified person |
| `legacy_profile_cca962d0c2dbed7eb151` | `ALREADY_CANONICAL` | 20 | 17 | 1 / 1 | deduplicate into verified person |
| `legacy_profile_f8af4121fba1bb4b6529` | `ALREADY_CANONICAL` | 1 | 0 | 1 / 1 | deduplicate into verified person |
| `legacy_profile_f8ddb748378dad481953` | `ALREADY_CANONICAL` | 22 | 21 | 1 / 1 | deduplicate into verified person |
| `legacy_profile_0958332e1babf5dd0e51` | `SAFE_TO_MIGRATE` | 8 | 8 | owner-locked stable ID | create one verified canonical identity |
| `legacy_profile_067e1c851bf1d8249a51` | `CONFLICT` | 1 | 0 | 2 people / 1 stable | retain, do not merge |
| `legacy_profile_2decdffc59888cbb3cda` | `CONFLICT` | 2 | 1 | 2 people / 1 stable | retain, do not merge |
| `legacy_profile_016b8c231fe12f6bfb61` | `ORPHANED` | 3 | 3 | none | retain legacy |
| `legacy_profile_079c2b5293eb53208a07` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_140dba98bb5d5ebe2d80` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_14d260d3e36951bf0d8a` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_1e5623dbd8fb4c160357` | `ORPHANED` | 3 | 2 | none | retain legacy |
| `legacy_profile_2937a45a57d70baf6a79` | `ORPHANED` | 1 | 0 | none | retain legacy |
| `legacy_profile_2a683d0d972f1090be8d` | `ORPHANED` | 7 | 4 | none | retain legacy |
| `legacy_profile_2af313808530fa710265` | `ORPHANED` | 7 | 7 | none | retain legacy |
| `legacy_profile_39a28e2a8998aba70133` | `ORPHANED` | 4 | 3 | none | retain legacy |
| `legacy_profile_49b42182d513c011e9e6` | `ORPHANED` | 22 | 14 | none | retain legacy |
| `legacy_profile_67c345decc7d60787624` | `ORPHANED` | 1 | 0 | none | retain legacy |
| `legacy_profile_6ef9ceaa1f797b6227bf` | `ORPHANED` | 16 | 16 | none | retain legacy |
| `legacy_profile_7980264033d71355dafb` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_8e28dc8d07b43f50d89e` | `ORPHANED` | 6 | 6 | none | retain legacy |
| `legacy_profile_955649e411236c11ff27` | `ORPHANED` | 1 | 0 | none | retain legacy |
| `legacy_profile_a99e96ac649d7b858e6e` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_af184bcc04c1faba786e` | `ORPHANED` | 2 | 2 | none | retain legacy |
| `legacy_profile_b0678026b55dd5f0741f` | `ORPHANED` | 26 | 25 | none | retain legacy |
| `legacy_profile_c28bb9cc10dcc66536ec` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_c7b8fa9392fa88fcc46a` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_d02fc661f0a8ed48530e` | `ORPHANED` | 3 | 0 | none | retain legacy |
| `legacy_profile_d16d043b2152465bf985` | `ORPHANED` | 18 | 11 | none | retain legacy |
| `legacy_profile_d35ca5051b82ffc326a3` | `ORPHANED` | 9 | 2 | none | retain legacy |
| `legacy_profile_d9298a10d1b0735837dc` | `ORPHANED` | 8 | 2 | none | retain legacy |
| `legacy_profile_d974012599726486780e` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_e0085cd22446dab66cdd` | `ORPHANED` | 2 | 2 | none | retain legacy |
| `legacy_profile_e8e8bd4dbd02a6b38964` | `ORPHANED` | 2 | 2 | none | retain legacy |
| `legacy_profile_ee512adb08ed702f35f0` | `ORPHANED` | 1 | 1 | none | retain legacy |
| `legacy_profile_f71bf347ba529d6d5633` | `ORPHANED` | 2 | 1 | none | retain legacy |
| `legacy_profile_fa93d7e62f81cff5ca98` | `ORPHANED` | 2 | 2 | none | retain legacy |

Totals: 6 `ALREADY_CANONICAL`, 1 `SAFE_TO_MIGRATE`, 0 `ALIAS_UPDATE`,
0 `AMBIGUOUS`, 30 `ORPHANED`, 2 `CONFLICT`. The six deterministic matches
have exact login, one stable identity, the same stream session and first-seen
timestamps less than 0.1 seconds apart. The two conflicts each have a uidless
raid-created person and a stable chat-created person; no automatic merge is
performed.

## 4. Classification of the 184 historical summaries

The summaries are keyed by legacy username/profile, not stable ID. They contain
contextual summary text, topics, counts and legacy notable-quote fields;
inferred-fact fields are empty in the audited data. Migration copies only the
contextual summary, topics and counts. It never copies raw quotes or promotes
casual observations to beliefs.

| Classification | Count | Reason / result |
|---|---:|---|
| `MIGRATABLE` | 70 | 62 belong to the six deterministic canonical profiles; 8 belong to the owner-verified migration |
| `KEEP_LEGACY_REFERENCE` | 113 | owner profile has no safe stable identity; preserved in place |
| `AMBIGUOUS_OWNER` | 1 | owner profile is one of the modern-identity conflicts; not assigned |
| `ORPHANED` | 0 | every summary has an audited legacy profile row |

This evidence supersedes the earlier provisional “121 pending” estimate: the
row-by-row migration audit finds 114 summaries that must stay legacy (113
unresolved plus 1 conflict), while 70 can be linked safely.

## 5. Ownership before cutover

Twitch chat could write SocialWorld conditionally and then independently call
stream memory, whose presence observer created/updated `chatter_profiles`,
`chatter_presence` and inferred chatter facts. Raids similarly wrote a modern
episode and legacy presence. Session close wrote `stream_chatter_summaries`,
while commands and voice target discovery read legacy profiles and summaries.
Feature flags selected or shadowed both routes. Identity, presence, familiarity
and summary ownership were therefore split between engine, stream memory and
SocialWorld.

## 6. Ownership after cutover

| Concern | Final owner |
|---|---|
| Identity resolution and profile creation | `SocialWorldService` + `SocialWorldRepository` |
| Stable identity persistence | `person_identities` |
| Username/alias rename | SocialWorld resolver, on the same stable identity |
| Display name | SocialWorld identity presentation field |
| Presence observation and deduplication | SocialWorld `observe_presence` + `social_presence_events` |
| Familiarity/session aggregates | `person_sessions` through SocialWorld |
| Interaction history | `social_episodes`, canonical person-scoped |
| Social hypotheses/facts | belief lifecycle with provenance, never raw presence |
| Summary write/read | `social_summaries` through SocialWorld |
| Social retrieval | `SocialWorldService.retrieve` |
| Raw chat/event observation | stream memory, explicitly non-authoritative |
| Coordination | `HebeEngine`, with no repository ownership |

Presence decides that an observation occurred and supplies evidence. It does
not invent or persist a second identity model.

## 7. Dual reads and writes found

The cutover removed these production paths:

1. direct/ambient chat → conditional modern identity plus unconditional legacy
   profile/presence/fact write;
2. raid → modern episode plus username-only legacy presence write;
3. stream close → legacy chatter-summary write plus later social use;
4. manual/latest summary and “qué dijo” → legacy chatter summary read;
5. voice known-target discovery and profile/last-seen commands → legacy profile
   reads;
6. username-indexed and stable-ID-indexed identity creation in parallel;
7. cutover/shadow flags in production and replay environment mapping;
8. per-viewer summaries embedded in `stream_summaries` as well as written to
   SocialWorld.

New `stream_summaries` rows keep the historical column physically compatible
but write `[]`; per-viewer candidates exist only transiently until SocialWorld
accepts them.

## 8. Migrations

Two transactional migrations run after the SocialWorld v2 base schema:

- `social_identity_canonicalization:1` adds canonical presence schema, audits
  every legacy profile, creates only owner-verified identities, deduplicates
  deterministic matches and migrates presence aggregates/events;
- `social_summary_canonicalization:1` adds person-scoped summaries, requires
  the identity audit, migrates only safely mapped rows and records every
  migrated/deduplicated/skipped/ambiguous/conflict/error result.

Audit tables are `legacy_social_identity_migration_audit` and
`legacy_social_summary_migration_audit`. Source profile IDs are hashed; target
IDs are technical canonical IDs. IDs and observation keys are deterministic.
Schema incompatibility raises before a completion marker is committed.

Simulation against a temporary copy of the real DB produced 11 people and 11
identities (one safe new identity), 79 canonical presence events and 70
canonical summaries. A second pass reported both migrations already applied
and created no duplicate.

## 9. Aliases, usernames and renames

The same stable ID with a new login updates the current login and appends both
old and new login to ordered alias history; it does not create a new person.
Changing only display name updates presentation. The same current login with
two different stable IDs creates two distinct people. Alias lookup can discover
rename history, but can never override contradictory stable-ID evidence.

## 10. Ambiguous, conflict and orphan handling

Ambiguous or multiple candidates raise/fail closed and remain visible in audit.
No third person is silently created to hide ambiguity. Conflicts are not
overwritten or merged. Orphaned legacy profiles and their summaries remain in
their source tables. No stable ID is invented, no summary is assigned by name
alone, and the two pre-existing modern duplicate pairs remain deliberately
unmerged for manual adjudication.

## 11. Remaining runtime legacy reads and writes

- legacy social profile writes: **0**;
- legacy social profile reads: **0**;
- legacy social summary writes: **0**;
- legacy social summary reads: **0**.

Source search finds `chatter_profiles`, `chatter_presence`, `chatter_facts` and
`stream_chatter_summaries` only in the isolated versioned migration and its
tests. `chatter_highlights_json` remains a physical historical stream-summary
column, but production writes it empty and no longer reads it.

## 12. Physical state of legacy data

No historical profile, presence, fact, summary or raw chat row is deleted.
Legacy tables remain rollback/audit material on existing databases. Fresh
stream-memory schemas no longer create the four legacy social tables. Physical
deletion requires a separate approved migration after retention and manual
conflict decisions.

## 13. Dead code removed

Removed legacy profile/presence/fact creation and mutation, username-only
profile status helpers, legacy profile/summary getters and formatters, session
summary inserts/deletes, legacy summary-count evidence, raid dual writes,
profile fallback reads, shadow comparisons, cutover flags and environment
mappings, stale imports and unreachable branches. Historical replay flag DTO
fields remain parser metadata only and cannot gate production.

## 14. `HebeEngine` cleanup

The engine now initializes the base plus both Social migrations, forwards Twitch
tags, coordinates one SocialWorld presence observation, delegates profile and
summary queries, and persists transient stream-close summary candidates only
through SocialWorld. Ambient plus direct delivery is idempotent through the
Twitch message/observation ID. No general engine refactor was performed.

## 15. Tests

Seventeen Phase 1D tests cover A–P plus incompatible-schema refusal: stable-ID
resolution, rename aliasing, display-name changes, login reuse across stable
IDs, verified and deterministic legacy mappings, ambiguous/conflict refusal,
safe and unsafe summary assignment, idempotency, canonical-only new viewers,
restart-safe history, explicit uidless behavior, presence exactly once and the
familiarity/owner-authority boundary.

Existing SocialWorld, Twitch coordinator/presence, cognitive Twitch, stream
data, voice, architecture and replay tests exercise the integrated routes.

## 16. QA

- Focused Social/Twitch/stream/voice/architecture/replay run: 353 passed, 19
  subtests.
- Temporary-copy migration: exact 39/184 audit counts above; second pass
  idempotent.
- Full backend: 1063 passed, 1 accepted failure, 5 warnings, 84 subtests.
- `compileall backend/app backend/tests`: passed.
- `git diff --check`: passed (line-ending notices only).

## 17. Remaining failure

The only accepted baseline failure remains
`test_response_synthesizer_handles_game_knowledge_command_result`, the existing
R4/Persona renderer/guard behavior. Phase 1D does not change Persona.

## 18. Historical risks

Thirty profiles and 113 summaries lack stable evidence and remain unavailable
to canonical runtime retrieval until a separately approved, externally verified
mapping exists. Two legacy profiles expose pre-existing duplicate modern people
that need manual adjudication. A completed one-shot migration does not
automatically reconsider later edits to legacy source data. Username reuse
without stable tags remains intentionally low-confidence/ambiguous. Historical
tables must not be dropped while they are the only recovery source for these
rows.

## 19. New-viewer guarantee

After cutover, a new viewer is created only in `people` +
`person_identities`, presence is recorded only through SocialWorld, and new
per-viewer summaries are born only in `social_summaries`. Stream memory records
the raw observation but cannot create a legacy social profile or summary.

## 20. Stable-ID guarantee

When Twitch supplies a stable user ID, it is the primary identity key. Login is
updated as an alias and display name remains presentation. Neither can replace,
merge or contradict the stable ID silently.

## 21. Proposed Phase 1E — not started

Audit ToolSystem, InteractionActions and `runtime.tools/actions` contract-first:
map construction, consumers, production reachability, public/runtime
compatibility obligations and app-open ownership; then add characterization
tests before choosing `DEAD`, `COMPATIBILITY_ONLY`, `PUBLIC_RUNTIME_CONTRACT` or
`STILL_ACTIVE`. Do not remove or refactor those components until that checkpoint
is reviewed.
