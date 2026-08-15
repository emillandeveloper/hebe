# Phase 0 ownership checkpoint — 2026-08-15

This checkpoint records the 16-failure triage before changing the affected
contracts, plus the app-open compatibility audit. "Current" in the triage
table means the behavior observed at the 1003-pass/16-failure baseline.

> Final audit update: the internally dead legacy dispatcher and old
> orchestrator execution/tool-handler stack described below were subsequently
> removed. The retained public runtime and migration blockers are recorded in
> `phase0_legacy_retirement_checkpoint_2026-08-15.md`.

## Failure triage by root cause

| Test | Expected by test | Current behavior at baseline | Owner | Root cause / shared group | Classification | Recommended action and disposition |
|---|---|---|---|---|---|---|
| `small_talk_prompt_blocks_recaps_and_memory` | Small talk excludes retrieved memory and recap instructions | The universal response path ignored `inject_memory/context_policy` and exposed chunks | `ResponseSynthesizer` context assembly + universal speech-act bundle | R1: canonical pipeline returned before historical invariants | `DEUDA_DE_OWNERSHIP` | Move memory eligibility into canonical scene construction; done, regression green |
| `fallback_chat_cannot_claim_pending_action_without_execution` | No completion claim without a successful execution receipt | Leading "Apuntado" escaped the old post-return guard | `final_response_guard` / action-claim guard | R1 | `DEUDA_DE_OWNERSHIP` | Make the canonical final guard own action claims and deterministic failure repair; done |
| `banter_prompt_blocks_planning_topic_shift` | Banter neither retrieves schedule/game memories nor changes topic to planning | Universal prompt lacked the old limited-context constraints | `ResponseSynthesizer` context assembly | R1 | `DEUDA_DE_OWNERSHIP` | Carry message policy into canonical prompt/scene; done |
| `jotun_prompt_tells_model_dog_first` | Resolved Jotun entity facts reach the prompt, dog first | `resolved_entities` were resolved upstream but omitted from the universal prompt | Entity resolver → `ResponseSynthesizer` prompt adapter | R1 | `DEUDA_DE_OWNERSHIP` | Add resolved entity lines as canonical allowed/required facts; done |
| `fallback_chat_blocks_ungrounded_walkthrough_claim` | Fallback chat cannot invent concrete game directions without sources | Universal return bypassed the historical walkthrough guard | `final_response_guard` with structured game state | R1 | `DEUDA_DE_OWNERSHIP` | Move the invariant to the final guard using `game_guidance_query/has_game_guidance_source`; done |
| `fallback_chat_is_blocked_while_game_pending_is_active` | An active game clarification owns the next compatible response | Universal bundle did not receive the active pending task | `final_response_guard` + canonical scene | R1 | `DEUDA_DE_OWNERSHIP` | Carry `active_pending_task` and guard generic replies; done |
| `stt_answer_without_wake_is_owner_followup` | A fresh, explicit game-state answer is an owner follow-up | Fixture had no `created_at`, so TTL treated it as ancient | Pending compatibility contract | R2: stale fixture | `TEST_STALE` | Add real `created_at`; use an utterance satisfying the documented explicit-progress gate; done, no production change |
| `successful_state_update_mutates_runtime_game_run_state` | Semantic run fields are written from an unqualified fixture update | Runtime correctly rejected character/party without owner provenance and confidence | Game run state write guard | R3: test omitted evidence required by write contract | `EXPECTATIVA_LEGACY` | Split into rejection-without-evidence and acceptance with `leo_clarification`/0.95; done, no production change |
| `response_synthesizer_handles_game_knowledge_command_result` | A factual Persona profile reply retains `Persona 5 Royal` | Game advice gate reads the plural word "personas" as the Persona mechanic and repairs a valid reply | `GameAdviceGate` claim detector | R4: lexical false positive, independent of cleanup | `BUG_REAL` | `BUG_FUNCIONAL_POST_FASE_0`; intentionally unchanged |
| `owner_stop_compliments_creates_behavior_block` | Persist owner restriction and expose legacy `compliments_to_leo` fields | Canonical compiler persists `behavior_family=compliment`, `recipient_scope=owner` | `BehaviorConstraintCompiler` | R5: assertions target compatibility naming | `EXPECTATIVA_LEGACY` | Assert canonical constraint and semantic match; done |
| `semantic_owner_stop_mode_creates_behavior_block` | Vague "modo baboso" immediately persists a compliment block | Canonical compiler requests clarification because no explicit behavior family is present | `BehaviorConstraintCompiler` | R5 | `TEST_STALE` | Assert clarification and no persisted constraint; done |
| `semantic_owner_stop_halogos_creates_behavior_block` | Explicit halagos restriction is visible under legacy name | Canonical constraint is `compliment/owner` | `BehaviorConstraintCompiler` | R5 | `EXPECTATIVA_LEGACY` | Assert canonical family/scope and `constraint_matches`; done |
| `behavior_block_applies_to_semantic_viewer_variant` | A block created by the vague phrase blocks a later viewer request | No constraint is created for the vague phrase, so there is nothing to match | Compiler + viewer constraint matcher | R5 | `TEST_STALE` | Seed with an explicit owner restriction, then prove canonical matching; done |
| `twitch_normal_no_mention_chat_reaches_presence_observe` | Every valid unaddressed viewer message reaches Presence, usually as observe-only | Router-level `viewer_context_only` returned before Presence | Presence decision boundary | R6: Router and Presence both owned intervention | `DEUDA_DE_OWNERSHIP` | Remove the special early return; done |
| `high_value_game_tip_can_reply_without_hebe_mention` | Presence may intervene on a valuable unaddressed tip | Same early return prevented value evaluation | Presence decision boundary | R6 | `DEUDA_DE_OWNERSHIP` | Let Presence evaluate all valid viewer inputs, without tip-specific exception; done |
| `twitch_pipeline_health_counts_messages` | Health counts show the message was Presence-evaluated | Early return skipped the evaluation/counter | Presence decision boundary | R6 | `DEUDA_DE_OWNERSHIP` | Same ownership consolidation; done |

All 16 failures existed before this pass. R1, R2, R3, R5 and R6 share causes
within their named groups; R4 is independent.

## Final ownership

- Universal response invariants: `HebeResponsePipeline` and
  `final_response_guard`, fed by the canonical scene assembled by
  `ResponseSynthesizer`.
- Unaddressed Twitch intervention: Presence. `CognitiveRouter` retains source,
  authority, addressing and general input classification; it does not decide
  observe versus intervene for a valid viewer message.
- App-open invocation from Hebe input: `DeliberationService` creates the plan and
  `PlanExecutor` is the only cognitive execution entry. The remaining chain is
  `ActionRuntime -> LocalCapabilityResolver -> WinAutomationService`.

## ToolSystem / InteractionActions / orchestrator audit

| Component | Constructed by | Actual internal consumer | Production reachability / contract | Removal impact | Tests | Classification |
|---|---|---|---|---|---|---|
| `HebeRuntime.tools` / `ToolSystem` | `build_runtime()` | No active Hebe cognitive path; the public object can be called by an external runtime consumer | Constructed in production and exposed as a `HebeRuntime` field; `open_app` delegates to `InteractionActions` | Breaks `HebeRuntime` construction/API and unknown external calls to `runtime.tools.call/exec`; also removes other default tools | No direct contract test | `PUBLIC_RUNTIME_CONTRACT` |
| `HebeRuntime.actions` / `InteractionActions` | `build_runtime()` | Bound as `ToolContext.open_app_fn` and `memory_fn`; otherwise no active cognitive consumer | Reachable through the public `runtime.actions` field and through public `runtime.tools` | Breaks external action consumers and the compatibility callbacks used by ToolSystem | No direct contract test | `PUBLIC_RUNTIME_CONTRACT` |
| `legacy/dispatcher.py` | No constructor/import in active app | None | Only usable by an external importer; it calls the public ToolSystem contract | Unknown external legacy integrations would break | No tests | `COMPATIBILITY_ONLY` |
| `orchestrator/orchestrator.py`, gates, policy, executor, dispatcher | No active bootstrap or constructor | Internal package references only | Not reachable from `HebeEngine`; `legacy_flow` delegates directly to `cognitive_flow` | External importers of the old package would break | No orchestrator tests | `COMPATIBILITY_ONLY` |
| `orchestrator/intents/resolver.py` + catalog/models | Dev script `app/scripts/eval_intents.py`; old orchestrator module also imports it | Intent evaluation script | Not in production input path, but still used by developer tooling | Breaks intent evaluation script and external imports | No direct tests | `COMPATIBILITY_ONLY` |
| `orchestrator/tool_handlers.handle_open_app` | Only returned by `build_tool_handlers`; neither has a caller | None | No internally reachable production route; an external importer could construct it and call Windows directly | Breaks unknown external old-orchestrator consumers | No tests | `COMPATIBILITY_ONLY` (internally dead) |
| `ActionRuntime` app-open path | `HebeEngine` | `PlanExecutor` | Active canonical production path | App opening stops | Architecture, voice, routing and resolver tests | `STILL_ACTIVE` |

There is no second app-open route reachable from Hebe's internal input loop after
the shortcut removal. There are still two externally invocable compatibility
surfaces capable of opening an app without `PlanExecutor`:

1. `runtime.tools.call("open_app", ...) -> InteractionActions -> WinAutomation`;
2. externally constructed `orchestrator.tool_handlers.handle_open_app`.

They are not internal consumers, but the first is an explicit public runtime
contract and the second cannot be declared safe to delete without ruling out
external imports.

## Deprecation / migration plan

1. Publish `runtime.execute_plan(plan)` or a narrower canonical app-action facade
   whose implementation always enters `PlanExecutor`; retain `tools/actions` as
   deprecated read-only compatibility fields for one release.
2. Instrument calls to `ToolSystem.call/exec`, `InteractionActions` methods and
   orchestrator tool handlers with a caller/contract version so real external
   usage can be measured.
3. Change `ToolSystem.open_app` to delegate to the canonical facade while
   preserving its input/result schema; add contract tests for events, failures
   and exactly-once execution.
4. Move `eval_intents.py` to the canonical router or freeze it as a standalone
   developer package. Announce removal of `app.orchestrator` and
   `app.legacy.dispatcher` with an import-level deprecation warning.
5. Remove compatibility fields/modules only after a release window with zero
   observed callers and a documented replacement. Do not silently remove fields
   from `HebeRuntime`.

## Deferred functional bug

`GameAdviceGate` currently treats the ordinary plural word `personas` as a
Persona-series mechanics claim. This makes a valid Persona 5 knowledge response
fail `final_response_guard`. It is deliberately left as
`BUG_FUNCIONAL_POST_FASE_0`; fixing its language/game disambiguation is outside
this ownership cleanup.
