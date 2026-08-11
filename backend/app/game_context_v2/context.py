from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Callable

from app.epistemics.models import EvidenceRef, RetrievalRequest
from app.game_context_v2.models import GameContext
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameKnowledgeService, GameRunService


class GameContextResolver:
    """Canonical Scene → Run → Knowledge → Memory → Gap → Research seam."""
    def __init__(self,repository:GameV2Repository,runs:GameRunService,knowledge:GameKnowledgeService,*,research_service:Any=None,memory_retrieval:Any=None,now_fn:Callable[[],float]=time.time)->None:
        self.repository=repository;self.runs=runs;self.knowledge=knowledge;self.research_service=research_service;self.memory_retrieval=memory_retrieval;self.now_fn=now_fn
        self.last_context={};self.context_latencies=[];self.research_calls=0;self.cache_hits=0;self.research_failures=[]

    def build(self,*,game:str,purpose:str,stream_session_id:str="",run_id:str="",subject_ref:str="",predicate:str="",question_type:str="",query_intent:str="",spoiler_ceiling:str="strict",required_confidence:float=.6,event_id:str="context",allow_research:bool=False,historical_run:bool=False,scene_assertions:tuple[dict[str,Any],...]|list[dict[str,Any]]=())->GameContext:
        started=time.perf_counter();identity=self.repository.resolve_identity(game)
        run=self.repository.get_run(run_id) if run_id else next(iter(self.repository.list_runs(game_id=identity.game_id,owner_id="leo",statuses=(("HISTORICAL","COMPLETED","ABANDONED") if historical_run else ("ACTIVE",)))),None)
        run_facts=self.runs.facts(run.id,historical=historical_run) if run else []
        current=[item for item in run_facts if item["epistemic_status"] in {"KNOWN","INFERRED","SUSPECTED"} and not item["superseded_by"]]
        hypotheses=[item for item in current if item["epistemic_status"] in {"INFERRED","SUSPECTED"}]
        selected,rejected=self.knowledge.find(identity.game_id,subject_ref=subject_ref,predicate=predicate,spoiler_ceiling=spoiler_ceiling)
        research_status="knowledge_available" if selected else "knowledge_miss"
        rag_context=[]
        if not selected and self.memory_retrieval is not None:
            result=self.memory_retrieval.retrieve(RetrievalRequest(context_kind="owner_local",purpose=purpose,subject=subject_ref,allowed_scopes=("game_run","game","global"),temporal_intent="historical" if historical_run else "current",max_results=20,provenance_required=True))
            for item in result.selected_claims:
                namespace=str(item.get("namespace") or "");scope_id=str(item.get("scope_id") or "")
                scoped_general=namespace not in {"game_run","game_knowledge"} and str(item.get("scope_kind") or "") in {"global","game"} and scope_id in {"",identity.game_id}
                predicate_matches=not predicate or str(item.get("predicate") or "")==predicate
                if scoped_general and predicate_matches:rag_context.append(item)
            if rag_context:research_status="memory_available"
        gaps=[]
        if not selected and any(item.get("rejection_reason")=="spoiler_blocked" for item in rejected):
            research_status="spoiler_blocked"
            print(f"[HEBE][GAME_RESEARCH_SKIP] reason=spoiler_blocked game={identity.game_id}",flush=True)
        elif not selected and not rag_context and allow_research and question_type and subject_ref:
            gap=self.knowledge.create_gap(game_id=identity.game_id,run_id=run.id if run else "",subject_ref=subject_ref,question_type=question_type,query_intent=query_intent or predicate,spoiler_ceiling=spoiler_ceiling,required_confidence=required_confidence,event_id=event_id);gaps=[gap.to_dict()]
            provider=getattr(self.research_service,"provider",None)
            if self.research_service is None or (provider is not None and not bool(getattr(provider,"available",True))):
                research_status="research_unavailable";print(f"[HEBE][GAME_RESEARCH_SKIP] reason=provider_unavailable game={identity.game_id}",flush=True)
            else:
                print(f"[HEBE][GAME_RESEARCH_REQUEST] gap_key={gap.normalized_gap_key} reason=typed_internal_miss",flush=True)
                try:
                    plan=self.research_service.plan_search(game_title=identity.canonical_name,game_id=identity.game_id,entity=subject_ref,question_type=question_type,expected_fact_type=predicate or question_type,owner_uncertainty=query_intent,spoiler_limit=spoiler_ceiling)
                    researched=self.research_service.research(plan,progress=None,allow_cache=True);self.research_calls+=1;fact_ids=[];rows=[]
                    for fact in researched:
                        rows.append(fact if isinstance(fact,dict) else {"claim":getattr(fact,"claim",None),"source_url":getattr(fact,"source_location",""),"excerpt":getattr(fact,"exact_supporting_excerpt_internal",""),"confidence":getattr(fact,"confidence",.75),"source_quality":getattr(fact,"source_type","unknown"),"spoiler_class":getattr(fact,"spoiler_classification","safe_general_mechanic")})
                    for index,row in enumerate(rows):
                        location=str(row.get("source_url") or row.get("url") or row.get("source_location") or "");excerpt=str(row.get("excerpt") or row.get("supporting_excerpt") or "")
                        evidence=EvidenceRef(source_event_id=f"{event_id}:research:{index}",source_record_type="game_research_result",source_record_id=location,observed_at=self.now_fn(),extractor="game_research_validator",extractor_version="v1",literal_span={"source_url":location,"excerpt":excerpt[:240]})
                        belief=self.knowledge.add_validated(game_id=identity.game_id,subject_ref=str(row.get("subject_ref") or subject_ref),predicate=str(row.get("predicate") or predicate or question_type),object_value=row.get("object",row.get("claim")),confidence=float(row.get("confidence") or .75),evidence=evidence,source_type="web",source_quality=str(row.get("source_quality") or row.get("source_type") or "secondary"),spoiler_class=str(row.get("spoiler_class") or row.get("spoiler_classification") or "safe_general_mechanic"),version_tag=str(row.get("version") or ""))
                        if belief:fact_ids.append(belief.id)
                    if fact_ids:
                        gap=replace(gap,status="resolved",updated_at=self.now_fn(),resolved_fact_ids=tuple(fact_ids));self.repository.save_gap(gap);gaps=[gap.to_dict()]
                        selected,rejected=self.knowledge.find(identity.game_id,subject_ref=subject_ref,predicate=predicate or question_type,spoiler_ceiling=spoiler_ceiling);research_status="research_completed"
                    else:research_status="research_failed_validation"
                except Exception as exc:
                    research_status="research_failed";self.research_failures.append({"gap_key":gap.normalized_gap_key,"error":type(exc).__name__})
        elif selected:
            self.cache_hits+=1;print(f"[HEBE][GAME_KNOWLEDGE_HIT] game={identity.game_id} gap_key={subject_ref}|{predicate} claim_ids={[item['id'] for item in selected]}",flush=True);print(f"[HEBE][GAME_RESEARCH_SKIP] reason=already_known game={identity.game_id}",flush=True)
        scene=tuple(dict(item) for item in scene_assertions)
        manifest=[]
        for item in [*scene,*current,*selected,*rag_context]:manifest.append({key:item.get(key) for key in ("id","namespace","scope_kind","scope_id","epistemic_status","confidence","authority_class","evidence_ids","spoiler_class","source_type")})
        advice_allowed=bool(selected) if purpose=="game_advice" else True;reaction_allowed=True
        latency=(time.perf_counter()-started)*1000;self.context_latencies.append(latency)
        context=GameContext(identity.to_dict(),scene,run.to_dict() if run else {},tuple(current),tuple(hypotheses),tuple(selected),tuple(rejected),tuple(rag_context),tuple(gaps),research_status,tuple(manifest),advice_allowed,reaction_allowed,len(json.dumps(manifest,ensure_ascii=False).encode()),latency)
        self.last_context=context.to_dict();print(f"[HEBE][GAME_CONTEXT_BUILD] run_claims={len(current)} knowledge_claims={len(selected)} research_calls={self.research_calls}",flush=True);return context

    def diagnostics(self)->dict[str,Any]:
        import math,statistics
        v=sorted(self.context_latencies)
        return {"research_calls":self.research_calls,"cache_hits":self.cache_hits,"failures":list(self.research_failures),"context_performance":{"count":len(v),"p50_ms":round(statistics.median(v),6) if v else 0.0,"p95_ms":round(v[max(0,math.ceil(len(v)*.95)-1)],6) if v else 0.0}}
