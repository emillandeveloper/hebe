# Cognitive routing cleanup review

Reviewed against the current pipeline on 2026-06-20. The repository already contains
`CognitiveRouter`; this pass audits remaining decision owners without redesigning them.

## Safe cleanup completed

- Removed the first `DeliberationService._plan_twitch_event` definition. A later method with
  the same name shadowed it at class creation, so the removed body was unreachable.
- Removed `DeliberationService._looks_like_appointment`. It had no callers after appointment
  recognition moved into `CognitiveRouter`.
- Removed the duplicate local capability constant in `deliberation_service.py`; plans now use
  `CAP_OPEN_APP` from the router contract.
- Removed `backend/app/cognitive/service.py`. No import or dynamic reference exists in the
  repository. It was an older copy of `PlanExecutor` with duplicate result models and a fake
  `open_app` success fallback. The active implementation is `cognitive/plan_executor.py`.

## Decision ownership inventory

| File | Function/class | What it decides | Keep? | Behind CognitiveRouter? | Risk | Recommended action |
|---|---|---|---|---|---|---|
| `cognitive/cognitive_router.py` | `CognitiveRouter.route` | User intent priority, pending compatibility, capability and step grants | Yes; central owner | It is the owner | Low | Extend contracts here, not in downstream handlers. |
| `cognitive/context_builder.py` | `_classify_message_type`, `_build_context_policy` | Memory/history retrieval depth | Yes; advisory context selection | No action authority | Low | Keep read-only; never let message type authorize tools. |
| `cognitive/capabilities/goal_extractor.py` | `GoalExtractor.extract` | Goal metadata and slots | Yes | Already defers to `context.cognitive_decision` | Medium | Keep as assistant to routing; do not restore independent route priority. |
| `cognitive/deliberation_service.py` | `_handle_user_input` | Converts a decision into a plan and guards step types | Yes | Yes, currently enforced | Low | Keep fallback router adapter only for direct unit callers. |
| `cognitive/deliberation_service.py` | `_parse_relative_reminder`, `_plan_appointment`, `_resolve_pending_appointment` | Parses/constructs authorized scheduling plans | Yes | Yes, currently selected by decision intent | Medium | Preserve temporal internals; require the decision grant on every new entry point. |
| `cognitive/temporal/*` | parsers/interpreter/rules | Temporal facts, not user intent | Yes | Called only from an authorized scheduling route | Low | No routing logic should be added here. |
| `cognitive/scheduler.py` | `poll_due_events`, `_fire_reminder`, `push_event` | Emits due/system events | Yes | Not user-intent routing | Low | Keep; event delivery still needs the central event adapter noted below. |
| `cognitive/plan_executor.py` | `execute`, `_execute_*` | Performs plan side effects and trusts the plan producer | Yes | Indirectly; no independent decision verification | High | Add decision/capability proof validation before accepting additional plan producers. |
| `hebe_engine.py` | `cognitive_flow` wake/sleep branch | Executes wake/sleep before ContextBuilder/router | Yes for behavior | Not yet | High | Move behind a small router-recognized system-control intent in a dedicated change. |
| `hebe_engine.py` | manual pending/TTS/stream handlers in `cognitive_flow` | Mutates pending, audio, stream state and can execute stream commands | Temporarily | Routed first, but handlers do not inspect grants | High | Convert each family to capabilities; until then preserve the added guard comment. |
| `hebe_engine.py` | `_plan_and_execute_local_app_action` compatibility branch | Executes app actions for incomplete legacy/test engines | Compatibility only | Router runs first, but capability grant is not checked | Medium | Remove when all harnesses construct the real deliberation stack. |
| `hebe_engine.py` | `legacy_flow` + `orchestrator/*` | Complete alternate intent/policy/tool pipeline | Flag; current `handle_command` does not call it | No | High | Deprecate explicitly, then remove after external callers are ruled out. |
| `orchestrator/gates.py` | `check`, `_handle_pending_clarification` | Consumes pending replies before semantic new-request checks | Only for legacy flow | No | Critical if re-enabled | Do not reconnect to `handle_command`; adapt it to `CognitiveDecision` before reuse. |
| `hebe_engine.py` | `process_internal_event` | Twitch firewall, viewer policy, response decision, then event plan | Yes | No event decision adapter yet | High | Introduce an event-specific CognitiveDecision adapter; retain all safety gates. |
| `stream/input_firewall.py` | `InputAuthorityFirewall.decide` | Ingress trust, bot/media filtering, allowed output/action envelope | Yes; security boundary | Must remain before router | Low | Never merge this into language intent routing. |
| `cognitive/wake_name_resolver.py` | `WakeNameResolver.resolve` | Addressing/wake-name evidence | Yes | Evidence should feed router | Medium | Stop executing wake actions directly from resolver output. |
| `cognitive/stream_companion_flow.py` | classifiers and `ResponseDecisionResolver` | Conversation relevance and output recommendation | Yes, advisory | CognitiveDecision should dominate user routes | Medium | Rename/document as advisory when central event routing is added. |
| `stream/policy.py` | owner/viewer semantic policies | Safety veto, viewer authority, reply permission | Yes; safety layer | May veto after routing, never grant blocked capabilities | Medium | Preserve; make veto-only relationship explicit in a later adapter. |
| `integrations/twitch/chat_bot.py` | mention/follow-up and ignored-user gates | Determines which chat reaches callback | Yes; ingress eligibility | Before router/firewall | Medium | Keep bot firewall downstream because mention logic alone does not reject every bot. |
| `cognitive/response_synthesizer.py` | mode dispatch and style guards | Wording and safety cleanup | Yes | Must not choose actions/intents | Medium | Structured modes should fail visibly rather than silently becoming chat. |
| `frontend/src/App.tsx` | Simulation Lab | Displays policy and cognitive traces; dev-only pending setup calls backend | Yes | Observability only | Low | Keep route fields aligned with `CognitiveDecision`; never use presets as classifiers. |
| `stream/live_session.py`, `session_primer.py`, summary code | Context/session persistence and summaries | Chooses context facts, not commands | Yes | No action authority | Low | Keep separate from routing and preserve provenance. |

## Dangerous routes and current status

- Generic time vocabulary no longer triggers appointments. Appointment recognition is centralized
  and the removed legacy detector had no callers.
- Appointment pending resolution in the active cognitive flow requires router compatibility,
  authority, TTL, and no stronger new intent. The legacy orchestrator gate does not; it is flagged.
- Reminder parsing occurs only for `reminder_create_request` in active deliberation. The scheduler
  only fires stored reminders and does not infer user intent.
- Twitch viewer traffic is filtered by the firewall and viewer policy before model generation.
  Internal Twitch events still bypass a formal CognitiveDecision and are flagged for an adapter.
- Known bot messages are rejected by the firewall (`would_call_llm=false`); chat observation also
  refuses blocked firewall decisions.
- Ambient STT marked `ignore` or context-only cannot enter direct cognition/action in the current
  STT path. Regression coverage asserts the cognitive flow is not invoked.
- Structured time/date, app, appointment, reminder, catalogue, and personal-state intents are
  selected before fallback chat. Regression coverage makes fallback invocation fail for these routes.
- Response/persona fallbacks remain because they are failure output, not intent classifiers. Some
  are stylistically template-like; changing them is a persona refactor and is intentionally deferred.

## Obsolete artifacts reviewed but retained

- `backend/app/legacy/*` and `HebeEngine.legacy_flow` are isolated from `handle_command`, but external
  callers cannot be disproved from repository search alone. They are flagged rather than deleted.
- `backend/ollama/Modelfile.dolphin-old` is not runtime code. Its deployment use cannot be proven
  from this repository, so it remains.
- Database migrations, schemas, debug routing logs, temporal parsers, stream gates, capability
  catalogue, and Simulation Lab were intentionally retained.

## Next bounded cleanup

1. Add a CognitiveDecision adapter for wake/sleep and the three manual command families.
2. Add an event decision envelope for Twitch/system events while preserving firewall/policy vetoes.
3. Add capability proof validation at `PlanExecutor.execute`.
4. Confirm no external caller uses `legacy_flow`, then remove the alternate orchestrator pipeline.
