# Phase 1E — ToolSystem / InteractionActions retirement checkpoint

Status: implemented and verified. The Phase 1E working tree is intentionally
uncommitted. No push was performed.

The approved Phase 1D checkpoint was already present as local commit `9039298`
(`Hebe - Refactor 1D`). Its parent is `61268c6`, which remains `origin/main`.
The commit contains only Social/Presence migrations, cutover, replay, tests and
documentation; no DB, logs, datasets, WAV, `.claude` or runtime artifacts were
included.

## 1. ToolSystem consumers found

| Consumer/reference | Classification | Evidence and decision |
|---|---|---|
| `core/runtime.py::build_runtime` | `PRODUCTION_INTERNAL` construction only | Constructed `ToolSystem`; no runtime code read the resulting field. Removed. |
| `InteractionActions.open_app_from_text` | Internal callback provider | Registered as the `open_app` callback; this was the second executable app-open route. Removed. |
| `InteractionActions.store_memory_from_text` | Internal callback provider | Registered as `memory_fn`, although `ToolSystem` never registered a memory tool. Removed. |
| Phase 0 checkpoint documents | `HISTORICAL` | Audit records, not usage documentation or supported API. Preserved as history. |
| Tests, frontend, CLI, scripts, replay and package exports | No consumer | Repository-wide source search found no `call()`/`exec()` caller and no import outside construction. |

`ToolSystem` registered six independently executable tools: `open_app`,
`type_text`, `press_keys`, `open_url`, `close_window` and `volume`. It maintained
an in-memory callable registry and emitted start/end/error telemetry, but had no
authorization, cognitive-decision or capability guard. `open_app` parsed raw
command text indirectly; the other tools could invoke OS operations directly.
It could therefore bypass both `PlanExecutor` and `ActionRuntime`.

## 2. InteractionActions consumers found

| Consumer/reference | Classification | Evidence and decision |
|---|---|---|
| `core/runtime.py::build_runtime` | `PRODUCTION_INTERNAL` construction only | Created the object and exposed it as `runtime.actions`; removed. |
| `ToolSystem` callback bindings | Legacy internal bridge | `open_app_from_text` and `store_memory_from_text`; removed with ToolSystem. |
| Tests, scripts, frontend, replay and docs-as-API | No real consumer | No method call or supported example existed. |

The class was a historical callback collection: natural-language app lookup
and execution, legacy memory store/recall, interactive app learning and an STT
confirmation prompt. It duplicated identity/resolution and execution owners and
held no durable state of its own.

## 3. Public contracts found

None. `HebeRuntime` is importable Python, but README, frontend, CLI, examples,
packaging metadata and tests never documented `runtime.tools` or
`runtime.actions` as supported APIs. The two Phase 0 documents called them a
possible public contract only because an unknown importer could access them;
those same audits recorded no known consumer and no contract test. Mere Python
accessibility is not evidence of a supported external API.

No deprecation shim is justified.

## 4. Ownership before cutover

The canonical internal path already existed:

`CognitiveRouter → DeliberationService → PlanExecutor → ActionRuntime → LocalCapabilityResolver → WinAutomationService`

In parallel, a programmatic caller could use:

`runtime.tools.call/exec → ToolSystem registry → InteractionActions or direct OS operation`

For app-open specifically, `InteractionActions` performed a second
natural-language lookup against `app_commands` and invoked
`WinAutomationService` without a cognitive decision, plan guard or canonical
capability resolution.

## 5. Ownership after cutover

| Concern | Canonical owner |
|---|---|
| Intent and permission grant | `CognitiveRouter` |
| Action plan | `DeliberationService` |
| Guarded dispatch and structured step receipt | `PlanExecutor` |
| Action execution | `ActionRuntime` |
| App identity/discovery/ambiguity | `LocalCapabilityResolver` |
| OS launch/focus primitive | `WinAutomationService` → active `app.tools` primitives |
| Runtime dependency construction | `build_runtime` |

Any legitimate internal caller must submit intent through the cognitive input
loop. Code already holding a deliberated, authorized plan uses `PlanExecutor`;
only `PlanExecutor` invokes `ActionRuntime.execute` in production.

## 6. Components eliminated

- `services/tool_system.py` and `ToolContext`;
- `services/interaction_actions.py`;
- `runtime/tools.py` and empty `runtime/dispatcher.py`;
- `orchestrator/models.py`, `orchestrator/intents/catalog.py` and the duplicate
  1,058-line intent resolver;
- the old `eval_intents.py`, unguarded `hebe_test_intent.py`, obsolete
  `traint_intent_gate.py`, its JSONL datasets and `intent_gate.joblib`;
- an unguarded, hard-coded Twitch developer request script;
- four large historical filesystem inventories under `app/legacy` that had no
  importer, runtime role or documentation contract;
- unused duplicate app lookup/learning helpers in `db_sqlite.py` and unused
  lookup methods in `app.tools.registry`.

## 7. Deprecated components

None. There is no wrapper, alias, optional field, warning layer or scheduled
compatibility window. A shim would preserve the exact second authority this
phase removes, without serving a demonstrated consumer.

## 8. Final HebeRuntime contract

All remaining fields are `CORE_RUNTIME`:

- `stt`, `stt_enabled`;
- `llm`, `intent_llm`;
- `win`, `speak`, `state`;
- `twitch`, `twitch_events`, `twitch_chat_bot`.

They are constructed by `build_runtime`, consumed by `HebeEngine`/startup, and
reproduced by cognitive replay. No field is marked optional merely for old code.
There are no `PUBLIC_SUPPORTED`, `COMPATIBILITY` or known dead fields in this
dataclass after the cutover.

## 9. runtime.tools / runtime.actions final state

- `HebeRuntime.tools`: removed;
- `HebeRuntime.actions`: removed;
- production `runtime.tools` references: 0;
- production `runtime.actions` references: 0;
- `ToolSystem` references: 0 in production source;
- `InteractionActions` references: 0 in production source.

Replay constructs the same reduced runtime contract and no longer fabricates
empty compatibility objects or callbacks.

## 10. Orchestrator and legacy state

`backend/app/orchestrator`, `backend/app/runtime` and `backend/app/legacy` now
contain no tracked source files. The sole DEV consumer of the old resolver was
migrated to `app/scripts/eval_cognitive_router.py`, which evaluates the real
`CognitiveRouter` and cannot execute actions. Old intent datasets/model files
whose only producer/consumer was the removed stack were deleted.

The new evaluator imports and runs independently; four cases cover canonical
open-app recognition plus reported-command and ambient false-positive rejection.

## 11. Dead-code sweep

Source searches after cutover return zero for:

- `ToolSystem`, `ToolContext`, `InteractionActions`;
- `runtime.tools`, `runtime.actions`, `tools.call`;
- `open_app_from_text`;
- `app.orchestrator`, `IntentResolver`, `OrchestratorInput`, `IntentResult`;
- removed DB/registry app lookup and learning helpers.

The only production call to `action_runtime.execute` is in `PlanExecutor`. The
only production call to `runtime.win.open_app` is in `ActionRuntime`.

## 12. Tests added or modified

`test_runtime_surface_phase1e.py` adds nine tests for B–K:

- known target resolves and launches once;
- ambiguous and unknown targets never launch;
- `HebeRuntime` has no tools/actions surface;
- authorized plans execute and unauthorized plans preserve guard receipts;
- `build_runtime` constructs the reduced contract;
- production source contains none of the five legacy surface markers;
- the migrated CognitiveRouter DEV evaluation imports and passes.

Existing `test_app_open_architecture.py` supplies A and the end-to-end known app
case: normal text → router/deliberation → one plan → one resolution → one
execution. Existing voice, execution-guard, external-app lifecycle and replay
tests continue to protect the rest of the chain. No modern test monkeypatches a
removed compatibility API.

## 13. QA

- New runtime-surface + app-open + execution-guard tests: 26 passed.
- Canonical router script smoke: 4 cases passed.
- Broader action/voice/lifecycle/replay run before the single fixture correction:
  215 passed and 9 replay subtests; its only failure was the new evaluator
  expecting `general_conversation` where the canonical contract is
  `unknown_chat`. The fixture was corrected, not production behavior.
- Full backend after correction: 1072 passed, 1 accepted failure, 5 warnings,
  84 subtests.
- `compileall backend/app backend/tests`: passed.
- `git diff --check`: passed (line-ending notices only).

## 14. Remaining failure

Only `test_response_synthesizer_handles_game_knowledge_command_result` remains.
It is the accepted R4/Persona renderer/guard behavior: the deterministic result
contains `Persona 5 Royal`, but the repair route returns
`Te leo, Leo. Recalibro.`. Phase 1E does not change Persona or Game.

## 15. Single internal execution route

Confirmed. Normal app-open enters the cognitive router, produces a deliberated
plan, passes `PlanExecutor` authorization, executes through `ActionRuntime`,
resolves exactly once in `LocalCapabilityResolver`, and reaches the OS only
through `WinAutomationService`. Ambiguous and unknown results stop before the OS
primitive. There is no second internal app-open route.

## 16. Compatibility execution guarantee

There is no compatibility execution API. Consequently no compatibility layer
can parse language, resolve an app, decide permission or execute independently.
Historical documentation mentions old names only as an audit record and is not
importable or executable.

## 17. Remaining legacy-named code and exact reason

| Remaining piece | Classification | Reason retained |
|---|---|---|
| `app/tools/base.py`, `windows_apps.py`, `windows_input.py` | `ACTIVE` | Low-level OS primitives used by `WinAutomationService`; they do not parse intent or own capability authorization. |
| `app/tools/registry.py` | `ACTIVE` | Records app usage and learned process/window metadata after canonical resolution/execution. Unused lookup functions were removed. |
| `OllamaIntentClient` and `backend/models/ModelFile.intent` | `DEV_ONLY` / active model support | Structured extraction remains used by temporal/deliberation components; neither belongs to the deleted orchestrator nor executes actions. |
| replay `FakeWinAutomation.open_app` | `TEST` | External-boundary fake used to prove receipts without touching the OS. |
| Phase 0 checkpoint references | `HISTORICAL` | Immutable evidence of the earlier audit and migration rationale. |
| Versioned Memory/Game/Social legacy migrations | `HISTORICAL` but required | One-shot migration, rollback and audit of retained user data; outside the 1E execution surface. |

No executable ToolSystem/InteractionActions/orchestrator compatibility code
remains.

## 18. Proposed Phase 1 closure — not started

After review, commit Phase 1E alone and declare Phase 1 architectural
canonicalization closed. Record one consolidated ownership map and retention
plan for physical legacy tables, without dropping historical data yet. The next
functional work should be a separately scoped R4/Persona response-guard phase
or release-hardening pass; it must not be mixed into the Phase 1E commit. No such
work has started.
