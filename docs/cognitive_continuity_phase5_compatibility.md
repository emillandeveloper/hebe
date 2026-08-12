# Cognitive Continuity v2 Phase 5 compatibility map

Phase 5 is additive. No Phase 6 deletion or destructive migration is included.

| Existing subsystem | Phase 5 classification | Current role |
|---|---|---|
| Memory extractor | `legacy-primary` | Continues legacy extraction; it cannot commit Phase 5 typed deltas. |
| Stream summaries | `legacy-primary` | Human/session recap only; never authoritative consolidation input by itself. |
| Rolling summaries | `compatibility-only` | Bounded conversational context, not durable truth. |
| Session primer | `v2-shadow` | Existing session/game preparation remains active while typed ContinuityContext is introduced. |
| Schedule observations/hypotheses | `v2-primary` | Existing tables and recurrence logic are retained and aligned with temporal status. |
| Chatter summaries | `compatibility-only` | Legacy display/context; SocialWorld remains the typed social authority. |
| `persona/hebe_identity.py` | `v2-primary` | Versioned StableHebeCore source; immutable to learned state. |
| `hebe_voice.py` | `v2-primary` | Rendering voice; LeoLanguageModel is understanding-only. |
| Behavior constraints | `v2-primary` | Policy remains above learned preferences. |
| Existing action receipts | `v2-primary` | Domain stores remain authoritative; HistoricalActionLedger is a read projection. |
| Legacy pending systems | `compatibility-only` | Conversation Continuity v2 is primary, with existing projections retained. |
| Game compatibility stores | `compatibility-only` | GameRun/GameKnowledge v2 remain authoritative. |
| Social compatibility stores | `compatibility-only` | SocialWorld/SharedCulture v2 remain authoritative. |

Deprecated candidates for Phase 6 review (not removed here): broad transcript-to-memory extraction, unrestricted rolling-summary truth, and duplicate pending-state stores.
