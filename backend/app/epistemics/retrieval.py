from __future__ import annotations
import json,time,statistics,math
from typing import Any
from app.epistemics.models import BeliefStatus,RetrievalRequest,RetrievalResult
from app.epistemics.repository import BeliefRepository

class MemoryRetrievalCoordinator:
    def __init__(self,repository:BeliefRepository,*,now_fn=time.time): self.repository=repository;self.now_fn=now_fn;self.last_request={};self.last_result={};self.latencies=[]
    def retrieve(self,request:RetrievalRequest)->RetrievalResult:
        start=time.perf_counter();now=self.now_fn();selected=[];rejected=[];reasons={}
        rows=self.repository.list(subject_ref=request.subject) if request.subject else self.repository.list()
        for belief in rows:
            reason=""
            if belief.sensitivity not in request.allowed_sensitivity: reason="sensitivity_violation"
            elif not self._scope_allowed(belief.scope_kind,request): reason="scope_violation"
            elif request.epistemic_statuses and belief.epistemic_status not in request.epistemic_statuses: reason="status_filtered"
            elif request.temporal_intent!="historical" and belief.epistemic_status in {BeliefStatus.HISTORICAL,BeliefStatus.SUPERSEDED,BeliefStatus.REJECTED}: reason="not_current"
            elif request.temporal_intent=="historical" and belief.epistemic_status not in {BeliefStatus.HISTORICAL,BeliefStatus.SUPERSEDED}: reason="not_historical"
            elif belief.valid_until and belief.valid_until<=now and request.temporal_intent!="historical": reason="validity_expired"
            elif belief.relevance_until and belief.relevance_until<=now and request.temporal_intent!="historical": reason="relevance_expired"
            elif request.max_age and now-belief.last_confirmed_at>request.max_age: reason="max_age_exceeded"
            elif request.provenance_required and not belief.evidence_ids: reason="provenance_missing"
            item=belief.to_dict(); item["age_seconds"]=max(0.0,now-belief.last_confirmed_at)
            if reason: item["rejection_reason"]=reason;rejected.append(item);reasons[reason]=reasons.get(reason,0)+1
            else: selected.append(item)
        rank={BeliefStatus.KNOWN.value:5,BeliefStatus.INFERRED.value:4,BeliefStatus.SUSPECTED.value:3,BeliefStatus.HISTORICAL.value:2,BeliefStatus.SUPERSEDED.value:1,BeliefStatus.REJECTED.value:0}
        authority={"owner":5,"domain_validator":4,"deterministic":3,"extractor":2,"model":1,"legacy":0}
        selected.sort(key=lambda x:(int(x["owner_confirmed"]),rank[x["epistemic_status"]],authority.get(str(x["authority_class"]),1),len(x["evidence_ids"]),x["confidence"],x["last_confirmed_at"]),reverse=True)
        selected=selected[:max(0,request.max_results)]
        for item in selected:
            item["provenance"]=[
                {key:row[key] for key in ("id","source_event_id","source_record_type","source_record_id","relation","weight","observed_at","extractor","extractor_version")}
                for row in self.repository.evidence_for(str(item["id"]))
            ]
        manifest=json.dumps(selected,ensure_ascii=False,sort_keys=True).encode()
        latency=(time.perf_counter()-start)*1000;self.latencies.append(latency)
        result=RetrievalResult(tuple(selected),tuple(rejected),reasons,len(manifest),latency)
        self.last_request={"context_kind":request.context_kind,"purpose":request.purpose,"subject":request.subject,"temporal_intent":request.temporal_intent}
        self.last_result=result.to_dict();print(f"[HEBE][BELIEF_RETRIEVE] purpose={request.purpose} scope={request.context_kind} selected={[x['id'] for x in selected]} rejected={reasons}",flush=True);return result
    @staticmethod
    def _scope_allowed(scope,request):
        if request.allowed_scopes: return scope in request.allowed_scopes
        if request.context_kind=="stream_public": return scope in {"stream_public","public","global"}
        return scope not in {"stream_public_only"}
    def performance(self):
        v=sorted(self.latencies);return {"count":len(v),"p50_ms":round(statistics.median(v),6) if v else 0.0,"p95_ms":round(v[max(0,math.ceil(len(v)*.95)-1)],6) if v else 0.0}
