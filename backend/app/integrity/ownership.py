from __future__ import annotations


OWNERSHIP = [
 ("pending/clarifications","ConversationContinuityService","LegacyPendingCompatibilityAdapter"),("conversation state","ConversationContinuityService","HebeState compatibility projection"),("OpenThreads","OpenThreadRepository","none"),
 ("raw evidence/timeline","live_session_timeline","none"),("scene projections","SceneConsequenceReducer","wide runtime scene snapshots"),("memory_facts","BeliefRepository","LegacyMemoryFactAdapter"),("memory_chunks","BeliefMemoryRetrieval","retrieval cache only"),
 ("legacy MemoryExtractor","BeliefLifecycleService","MemoryExtractor deprecated writer"),("rolling summaries","live_session_rolling_summaries","archive/projection"),("stream summaries","stream_summaries","archive/projection"),("schedule systems","ScheduleLearningService","legacy stream_schedule compatibility"),
 ("game knowledge/research/guidance","GameContextResolver","GameDossier compatibility cache"),("GameProgressState","GameRunService","compatibility projection"),("GameRunState","GameRunService","runtime projection"),("GameDossier","GameKnowledgeV2Service","compatibility cache"),
 ("chatter profiles/facts/summaries","SocialWorldService","legacy social compatibility archive"),("SocialWorld","SocialWorldService","none"),("SharedCulture","SocialWorldService","none"),("persona identity/voice","StableHebeCore","none"),
 ("owner preferences","OwnerProceduralPreferences","none"),("action receipts","domain executors","none"),("action ledger","HistoricalActionLedger","generic memory action claims"),("Presence inputs","PresenceModel","none"),
 ("ContextBuilder","ContinuityContextBuilder","legacy CognitiveContextBuilder retained for non-v2 payload"),("FinalEmissionGate","FinalEmissionGate","none"),("Hebe evolving opinions","HebeSelfModel","none"),("Leo language","LeoLanguageModel","none"),
 ("consolidation","SessionConsolidator","broad summary-to-truth deprecated"),("temporal lifecycle","TemporalRelevanceService","domain TTL helpers retained as compatibility")]


def inventory(*, before: bool) -> list[dict]:
    result=[]
    for concern,owner,legacy in OWNERSHIP:
        if before:
            result.append({"concern":concern,"canonical_owner":owner,"current_writers":[owner,legacy] if legacy not in {"none","archive/projection","retrieval cache only","compatibility projection","compatibility cache"} else [owner],"current_readers":[owner],"legacy_writers":[] if legacy=="none" else [legacy],"legacy_readers":[] if legacy=="none" else [legacy],"compatibility_adapters":[] if legacy=="none" else [legacy],"shadow_paths":[legacy] if "shadow" in legacy.casefold() else [],"duplicate_state":legacy not in {"none"},"removal_candidate":legacy not in {"none"},"blocking_dependency":"runtime compatibility consumers" if legacy not in {"none"} else ""})
        else:
            result.append({"concern":concern,"canonical_owner":owner,"canonical_read_path":owner,"canonical_write_path":owner,"retained_compatibility":[] if legacy=="none" else [legacy],"reason":"One semantic owner; retained component cannot independently establish v2 truth."})
    return result
