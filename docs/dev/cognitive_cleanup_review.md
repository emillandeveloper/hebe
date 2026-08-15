# Cognitive routing cleanup review

Reviewed against the current pipeline on 2026-06-20. The repository already contains
`CognitiveRouter`; this pass audits remaining decision owners without redesigning them.

## Central router implementation status

The high-risk follow-up is now implemented. Wake/sleep and owner manual command families require
explicit router capabilities; `legacy_flow` delegates to `cognitive_flow`; Twitch internal events
carry a decision after the input firewall and before policy/deliberation; and `PlanExecutor` rejects
steps without decision, step-type, capability, authority, risk, and live-stream authorization.
Safety policy remains veto-only and cannot grant a capability blocked by the router.

There are no known active high-risk manual-handler bypasses after this pass. The remaining
critical item is dormant legacy code in `orchestrator/gates.py`: it is not reachable from
`legacy_flow` or `handle_command`, but must not be reconnected without a decision adapter.

Simulation now exposes the decision and executor guard and includes owner/pending, ambient, bot,
viewer-authority, and live/offline raid scenarios. Manual handler entry points also validate the
decision themselves, so a caller-side routing mistake cannot mutate pending, audio, wake, or stream
state. Compatibility tests provide an explicit decision instead of bypassing that boundary.

STT now builds one `InputEnvelope` before the input firewall and router. Wake-name evidence,
trusted no-wake command evidence, pending compatibility, source, authority, and trust are carried
forward instead of being independently reclassified by the stream companion layer. Compatible
appointment datetime answers use `owner_stt_followup`; high-confidence whitelisted local commands
without a wake name use `owner_stt_command`; neither is treated as ambient stream speech.

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
- Game Guidance clarification replies now use a real pending contract. Compatible UI and owner-STT
  follow-ups route through `CognitiveRouter`, update `GameRunState` through an authorized
  `state_update`, and re-evaluate the original question without asking for already supplied fields.
  Assistant aliases are stripped before party parsing, known entities use a data-backed alias
  catalogue with conservative fuzzy matching, and cross-game RAG chunks are excluded.

## Decision ownership inventory

| File | Function/class | What it decides | Keep? | Behind CognitiveRouter? | Risk | Recommended action |
|---|---|---|---|---|---|---|
| `cognitive/cognitive_router.py` | `CognitiveRouter.route` | User intent priority, pending compatibility, capability and step grants | Yes; central owner | It is the owner | Low | Extend contracts here, not in downstream handlers. |
| `cognitive/context_builder.py` | `_classify_message_type`, `_build_context_policy` | Memory/history retrieval depth | Yes; advisory context selection | No action authority | Low | Keep read-only; never let message type authorize tools. |
| `cognitive/capabilities/goal_extractor.py` | `GoalExtractor.extract` | Goal metadata and slots | Yes | Already defers to `context.cognitive_decision` | Medium | Keep as assistant to routing; do not restore independent route priority. |
| `cognitive/game_guidance.py` | `GameGuidanceCapability.evaluate` | Resolves game/run context, ambiguity, spoiler depth and source tier after router authorization | Yes | Yes; requires `game.guidance` | Low | Keep walkthrough claims grounded in RAG/web results and clarify ambiguous progression first. |
| `cognitive/deliberation_service.py` | `_handle_user_input` | Converts a decision into a plan and guards step types | Yes | Yes, currently enforced | Low | Keep fallback router adapter only for direct unit callers. |
| `cognitive/deliberation_service.py` | `_parse_relative_reminder`, `_plan_appointment`, `_resolve_pending_appointment` | Parses/constructs authorized scheduling plans | Yes | Yes, currently selected by decision intent | Medium | Preserve temporal internals; require the decision grant on every new entry point. |
| `cognitive/temporal/*` | parsers/interpreter/rules | Temporal facts, not user intent | Yes | Called only from an authorized scheduling route | Low | No routing logic should be added here. |
| `cognitive/scheduler.py` | `poll_due_events`, `_fire_reminder`, `push_event` | Emits due/system events | Yes | Not user-intent routing; delivery is routed | Low | Keep scheduler factual; event delivery authorization belongs to the router. |
| `cognitive/plan_executor.py` | `execute`, `_execute_*` | Performs plan side effects | Yes | Yes; validates the decision at execution | Low | Keep new capability mappings synchronized with new risky step types. |
| `hebe_engine.py` | `cognitive_flow` wake/sleep branch | Applies authorized local wake state | Yes for behavior | Yes | Low | Keep resolver evidence subordinate to the router intent. |
| `hebe_engine.py` | manual pending/TTS/stream handlers in `cognitive_flow` | Mutates pending, audio, stream state and can execute stream commands | Temporarily | Yes; caller and handler both validate grants | Medium | Replace route hints with registered matchers as command families migrate; retain the hard handler guard. |
| `hebe_engine.py` | canonical app-open coordination | Traces a deliberated app plan and records its execution receipt | Yes; coordination only | Yes; execution is owned by `PlanExecutor` | Low | Keep orchestration free of direct `ActionRuntime.execute` calls. |
| `hebe_engine.py` | `legacy_flow` + `orchestrator/*` | Compatibility entry point | Entry point retained | Delegates to CognitiveRouter pipeline | Low | Remove the unused alternate implementation after external callers are ruled out. |
| `orchestrator/gates.py` | `check`, `_handle_pending_clarification` | Consumes pending replies before semantic new-request checks | Only for legacy flow | No | Critical if re-enabled | Do not reconnect to `handle_command`; adapt it to `CognitiveDecision` before reuse. |
| `hebe_engine.py` | `process_internal_event` | Twitch firewall, router, viewer-policy veto, then event plan | Yes | Yes | Medium | Preserve the firewall-before-router and offline-stream gates. |
| `stream/input_firewall.py` | `InputAuthorityFirewall.decide` | Ingress trust, bot/media filtering, allowed output/action envelope | Yes; security boundary | Must remain before router | Low | Never merge this into language intent routing. |
| `cognitive/wake_name_resolver.py` | `WakeNameResolver.resolve` | Addressing/wake-name evidence | Yes | Yes; resolver runs only inside authorized wake handler | Low | Keep resolver evidence subordinate to router intent and capability. |
| `cognitive/stream_companion_flow.py` | classifiers and `ResponseDecisionResolver` | Conversation relevance and output recommendation | Yes, advisory | CognitiveDecision should dominate user routes | Medium | Rename/document as advisory when central event routing is added. |
| `stream/policy.py` | owner/viewer semantic policies | Safety veto, viewer authority, reply permission | Yes; safety layer | May veto after routing, never grant blocked capabilities | Low | Preserve the current veto-only relationship. |
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
  Internal Twitch events receive a `CognitiveDecision` after the firewall and before policy or
  deliberation. Known-offline events stop without reply/action; live-event grants are regression-tested.
- Known bot messages are rejected by the firewall (`would_call_llm=false`); chat observation also
  refuses blocked firewall decisions.
- Ambient STT marked `ignore` or context-only cannot enter direct cognition/action in the current
  STT path. Regression coverage asserts the cognitive flow is not invoked.
- Firewall-approved owner STT addressed directly to Hebe is passed to `CognitiveRouter` without a
  pre-router semantic-intent veto. Personal-state statements such as hunger and fatigue therefore
  reach `owner_personal_state`; the same unaddressed statements remain ambient/context-only.
- Structured time/date, app, appointment, reminder, catalogue, and personal-state intents are
  selected before fallback chat. Regression coverage makes fallback invocation fail for these routes.
- Fallback chat cannot claim an appointment, reminder, app launch, memory write, or similar action
  succeeded unless execution contains a successful action-like result. Unsupported completion claims
  are blocked and logged by `FALLBACK_GUARD`.
- Game progression, item, boss, build, and mechanics questions route through `game_guidance_query`.
  Ambiguous run state produces a persisted clarification. Compatible answers route as
  `game_guidance_clarification_answer`, mutate run state only through the authorized executor step,
  and leave unrelated/new requests untouched. Generic fallback chat is not authorized to invent
  walkthrough instructions, and concrete guidance requires a same-game local RAG or web source.
- Response/persona fallbacks remain because they are failure output, not intent classifiers. Some
  are stylistically template-like; changing them is a persona refactor and is intentionally deferred.

## Obsolete artifacts reviewed but retained

- `backend/app/orchestrator/*` remains on disk for compatibility, but `HebeEngine.legacy_flow`
  delegates to `cognitive_flow`. External imports cannot be disproved, so the package remains flagged.
- `backend/ollama/Modelfile.dolphin-old` is not runtime code. Its deployment use cannot be proven
  from this repository, so it remains.
- Database migrations, schemas, debug routing logs, temporal parsers, stream gates, capability
  catalogue, and Simulation Lab were intentionally retained.

## Next bounded cleanup

1. Confirm no external caller imports the alternate orchestrator package, then remove that package.
2. Replace the remaining owner-manual route hints with registered capability matchers as each
   command family is migrated. The hints remain temporary and medium risk, while handler-side
   authorization is mandatory regardless of how a route was selected.
3. Add capability inference mappings whenever a new risky `PlanStep` kind is introduced.
