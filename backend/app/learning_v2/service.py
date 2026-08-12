from __future__ import annotations

import hashlib
import json
import time
import uuid
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.epistemics.models import Belief, BeliefStatus, EvidenceRef, EvidenceRelation
from app.learning_v2.repository import LearningRepository

def _metric(values):
    ordered=sorted(values);return {"count":len(ordered),"p50_ms":round(statistics.median(ordered),6) if ordered else 0.0,"p95_ms":round(ordered[max(0,math.ceil(len(ordered)*.95)-1)],6) if ordered else 0.0}


@dataclass(frozen=True, slots=True)
class ConsolidationCandidate:
    domain: str; delta_type: str; payload: dict[str,Any]; evidence_ids: tuple[str,...]; idempotency_key: str=""

@dataclass(frozen=True, slots=True)
class ActionClaimDecision:
    allowed: bool; claim_strength: str; status: str; reason: str


class StableHebeCore:
    FORBIDDEN={"change_owner","disable_owner_authority","allow_viewer_control","remove_core_boundary","owner","authority_hierarchy","core_boundary"}
    def __init__(self) -> None:
        path=Path(__file__).resolve().parents[1]/"cognitive"/"persona"/"hebe_identity.py"
        self.path=str(path);self.version=hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    def validate(self,candidate:ConsolidationCandidate)->str:
        tokens={candidate.delta_type.casefold()}
        def collect(value):
            if isinstance(value,dict):
                for key,item in value.items():tokens.add(str(key).casefold());collect(item)
            elif isinstance(value,(list,tuple)): 
                for item in value:collect(item)
            elif isinstance(value,str):tokens.add(value.casefold())
        collect(candidate.payload)
        return "stable_core_immutable" if candidate.domain.upper() in {"STABLE_CORE","IDENTITY"} or tokens & self.FORBIDDEN else ""


class _BeliefModel:
    namespace="";allowed_predicates:set[str]=set()
    def __init__(self,repository,learning:LearningRepository,*,now_fn=time.time):self.repository=repository;self.learning=learning;self.now_fn=now_fn;self.lookup_latencies=[]
    def current(self,subject="",predicate=""):
        started=time.perf_counter();items=self.repository.list(namespace=self.namespace,subject_ref=subject,predicate=predicate);self.lookup_latencies.append((time.perf_counter()-started)*1000);return [x for x in items if x.epistemic_status in {BeliefStatus.KNOWN,BeliefStatus.INFERRED,BeliefStatus.SUSPECTED} and not x.superseded_by]
    def _write(self,*,subject,predicate,value,evidence:EvidenceRef,authority,owner_confirmed=False,retention="LONG",scope_kind="persona",scope_id="hebe"):
        if predicate not in self.allowed_predicates:raise ValueError("predicate_not_allowed")
        now=self.now_fn();active=self.current(subject,predicate)
        same=next((x for x in active if x.object_value==value),None)
        if same:
            self.repository.add_evidence(same.id,evidence,subject_key=self.repository._subject_key(same));return same,False
        belief=Belief(f"belief_{uuid.uuid4().hex}",self.namespace,scope_kind,scope_id,subject,predicate,value,BeliefStatus.KNOWN,1.0 if owner_confirmed else .78,authority,now,now,now,0,0,"",owner_confirmed,"normal",1,retention,1)
        result,created=self.repository.propose(belief,evidence)
        for old in active:self.repository.transition(old.id,status=BeliefStatus.SUPERSEDED,valid_until=now,superseded_by=result.id)
        return result,created
    def performance(self):return {f"{self.namespace}_lookup":_metric(self.lookup_latencies)}


class HebeSelfModel(_BeliefModel):
    namespace="hebe_self";allowed_predicates={"preference.game","opinion.character","opinion.mechanic","preference.interaction_style","attitude.topic"}
    def learn(self,**kwargs):
        result,created=self._write(authority="hebe_experience",owner_confirmed=False,**kwargs);print(f"[HEBE][SELF_OPINION_{'LEARN' if created else 'SUPERSEDE'}] belief_id={result.id}",flush=True);return result,created


class OwnerProceduralPreferences(_BeliefModel):
    namespace="owner_preference";allowed_predicates={"raid_ack.omit_viewer_count","raid_ack.omit_viewer_count_by_default","raid_ack.allow_explicit_exception","promo.style","social.followup.frequency","game_advice.preference","banter.intensity","phrase.use"}
    def learn(self,**kwargs):
        result,created=self._write(authority="owner",owner_confirmed=True,scope_kind="owner",scope_id="leo",**kwargs);print(f"[HEBE][OWNER_PREFERENCE_{'LEARN' if created else 'CORRECT'}] belief_id={result.id}",flush=True);return result,created
    def rendering_policy(self,context:str)->dict[str,Any]:
        values={b.predicate:b.object_value for b in self.current("leo") if b.predicate.startswith(context+".")}
        return {"context":context,"omit_viewer_count":bool(values.get("raid_ack.omit_viewer_count",values.get("raid_ack.omit_viewer_count_by_default",False))),"active_preferences":values}


class LeoLanguageModel(_BeliefModel):
    namespace="leo_language";allowed_predicates={"lexical.confirmation","lexical.rejection","style.brevity","repair.pattern","language.switch"}
    def observe(self,*,predicate,value,event_id,evidence,explicit=False):
        count=self.learning.observe(model="leo_language",subject=predicate,value=str(value),event_id=event_id,at=self.now_fn(),explicit=explicit)
        if count<2 and not explicit:return None,False
        return self._write(subject="leo",predicate=predicate,value=value,evidence=evidence,authority="owner_observation",owner_confirmed=False,scope_kind="owner",scope_id="leo")
    def interpretation_aliases(self)->dict[str,str]:
        return {str(x.object_value):"affirmative" for x in self.current("leo","lexical.confirmation")}


class HistoricalActionLedger:
    VALID={"REQUESTED","ATTEMPTED","FAILED","UNKNOWN","SUCCEEDED"}
    def __init__(self,repository:LearningRepository,*,now_fn=time.time):self.repository=repository;self.now_fn=now_fn;self.lookup_latencies=[];self.last_decision={}
    def project(self,*,source_store,source_record_id,action_type,target,status,evidence=None,requested_at=0,completed_at=0):
        status=status.upper()
        if status not in self.VALID:raise ValueError("invalid_action_status")
        row=self.repository.project_action(receipt_id=f"action_{uuid.uuid4().hex}",action_type=action_type,target=target,status=status,source_store=source_store,source_record_id=source_record_id,requested_at=requested_at or self.now_fn(),completed_at=completed_at or (self.now_fn() if status in {"FAILED","SUCCEEDED"} else 0),evidence=dict(evidence or {}));print(f"[HEBE][ACTION_LEDGER_PROJECT] action={action_type} status={status} source={source_record_id}",flush=True);return row
    def validate_claim(self,*,action_type,target=""):
        started=time.perf_counter();rows=[x for x in self.repository.rows("action_ledger",order="requested_at DESC,id DESC") if x["action_type"]==action_type and (not target or x["target"].casefold()==target.casefold())];self.lookup_latencies.append((time.perf_counter()-started)*1000)
        status=str(rows[0]["status"]) if rows else "UNKNOWN";decision=ActionClaimDecision(status=="SUCCEEDED","strong" if status=="SUCCEEDED" else "uncertain" if status=="UNKNOWN" else "none",status,"receipt_success" if status=="SUCCEEDED" else "receipt_unknown" if status=="UNKNOWN" else "receipt_not_successful");self.last_decision={"allowed":decision.allowed,"claim_strength":decision.claim_strength,"status":decision.status,"reason":decision.reason,"action_type":action_type,"target":target};print(f"[HEBE][ACTION_HISTORY_VALIDATE] action={action_type} status={status} allowed={decision.allowed}",flush=True);return decision
    def project_existing_receipts(self)->int:
        conn=self.repository.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:
            tables={str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "promotion_events" not in tables:return 0
            rows=[dict(r) for r in conn.execute("SELECT id,resolved_login,execution_status,failure_reason,created_at FROM promotion_events")]
        finally:conn.close()
        count=0
        for row in rows:
            raw=str(row.get("execution_status") or "").casefold();status="SUCCEEDED" if raw in {"sent","succeeded","success"} else "FAILED" if raw in {"failed","blocked","error"} else "ATTEMPTED"
            self.project(source_store="promotion_events",source_record_id=str(row["id"]),action_type="twitch_shoutout",target=str(row.get("resolved_login") or ""),status=status,evidence={"failure_reason":row.get("failure_reason")},requested_at=0);count+=1
        return count
    def performance(self):return {"action_ledger_lookup":_metric(self.lookup_latencies)}


class SceneConsequenceReducer:
    def __init__(self,repository:LearningRepository,preferences:OwnerProceduralPreferences,*,now_fn=time.time):self.repository=repository;self.preferences=preferences;self.now_fn=now_fn;self.last_transition={}
    def outgoing_raid(self,*,event_id,destination,receipt_status,viewer_count=0):
        if receipt_status!="SUCCEEDED":return {}
        payload={"raid_started":True,"stream_ending":True,"social_transition":True,"destination_person":destination,"farewell_opportunity":{"contribution_type":"RAID_FAREWELL","presence_gated":True,"final_emission_gated":True},"rendering_policy":self.preferences.rendering_policy("raid_ack"),"telemetry":{"viewer_count":viewer_count}}
        self.repository.save_scene(event_id=event_id,transition_type="OUTGOING_RAID",destination=destination,payload=payload,at=self.now_fn());self.last_transition=payload;print(f"[HEBE][SCENE_TRANSITION] type=outgoing_raid destination={destination}",flush=True);return payload
    def incoming_raid(self,*,event_id,source,viewer_count=0):
        payload={"raid_arrival":True,"stream_ending":False,"social_transition":False,"source_person":source,"rendering_policy":self.preferences.rendering_policy("raid_ack"),"telemetry":{"viewer_count":viewer_count}};self.repository.save_scene(event_id=event_id,transition_type="INCOMING_RAID",destination=source,payload=payload,at=self.now_fn());self.last_transition=payload;return payload


class TemporalRelevanceService:
    VERSION="temporal-v1"
    def __init__(self,connection_factory,beliefs,threads,social_repository,learning:LearningRepository,*,now_fn=time.time):self.connection_factory=connection_factory;self.beliefs=beliefs;self.threads=threads;self.social_repository=social_repository;self.learning=learning;self.now_fn=now_fn;self.last_actions=[];self.latencies=[]
    def maintain(self):
        started=time.perf_counter();now=self.now_fn();actions=[]
        # Fetch actionable-status rows without validity filtering so due rows can be transitioned and audited.
        for thread in self.threads.list_open():
            if min(x for x in (thread.valid_until,thread.relevance_until) if x)>now:continue
            from app.continuity.models import OpenThreadStatus
            self.threads.transition(thread.id,expected_version=thread.version,status=OpenThreadStatus.EXPIRED,event_id="temporal_maintenance",now=now);actions.append(self._audit(thread.id,"open_thread",thread.status.value,"EXPIRED","relevance_or_validity_elapsed"))
        for b in self.beliefs.list():
            if b.epistemic_status in {BeliefStatus.KNOWN,BeliefStatus.INFERRED,BeliefStatus.SUSPECTED} and b.relevance_until and b.relevance_until<=now and b.retention_policy not in {"AUTHORITATIVE","AUDIT","LONG"}:
                self.beliefs.transition(b.id,status=BeliefStatus.HISTORICAL);actions.append(self._audit(b.id,"belief",b.epistemic_status.value,"HISTORICAL","relevance_elapsed"))
        if self.social_repository:
            for item in self.social_repository.culture():
                if item["status"]=="ACTIVE" and now-float(item["last_reinforced_at"])>=30*86400:
                    old=item["status"];item["status"]="WEAKENING";item["confidence"]=max(.2,float(item["confidence"])-.15);self.social_repository.save_culture(item);actions.append(self._audit(item["id"],"shared_culture",old,"WEAKENING","reinforcement_stale"))
        self.last_actions=actions;self.latencies.append((time.perf_counter()-started)*1000);return actions
    def _audit(self,ref,typ,old,new,reason):
        operation="EXPIRE" if new=="EXPIRED" else "WEAKEN" if new=="WEAKENING" else "ARCHIVE";self.learning.audit_maintenance(object_ref=ref,object_type=typ,old_status=old,new_status=new,reason=reason,at=self.now_fn(),policy=self.VERSION);print(f"[HEBE][TEMPORAL_{operation}] object={ref} old={old} new={new}",flush=True);return {"object_ref":ref,"object_type":typ,"old_status":old,"new_status":new,"reason":reason}
    def performance(self):return {"temporal_maintenance":_metric(self.latencies)}


class SessionConsolidator:
    VERSION="consolidator-v1";DOMAINS={"GAME","SOCIAL","OWNER_PREFERENCE","SCHEDULE","HEBE_SELF","THREAD","SHARED_CULTURE","LEO_LANGUAGE","STABLE_CORE"}
    def __init__(self,repository:LearningRepository,core:StableHebeCore,self_model:HebeSelfModel,preferences:OwnerProceduralPreferences,language:LeoLanguageModel,*,now_fn=time.time,candidate_provider=None):self.repository=repository;self.core=core;self.self_model=self_model;self.preferences=preferences;self.language=language;self.now_fn=now_fn;self.candidate_provider=candidate_provider;self.last_result={};self.durations=[];self.validation_latencies=[];self.provider_failures=[]
    def consolidate(self,*,session_id,start_event,end_event,candidates=None,pre_state_version=""):
        started=time.perf_counter();run_id,fresh=self.repository.begin_run(session_id=session_id,start_event=start_event,end_event=end_event,pre_state_version=pre_state_version or "unknown",version=self.VERSION,now=self.now_fn())
        if not fresh:
            self.last_result={"run_id":run_id,"status":"ALREADY_COMPLETE","accepted_deltas":0,"rejected_deltas":0,"duplicate_deltas":0,"watermark":{"start":start_event,"end":end_event}};return self.last_result
        evidence_rows=self.repository.timeline_evidence(session_id=session_id);available_evidence={str(x.get("event_uid") or x.get("source_record_id") or "") for x in evidence_rows}
        if candidates is None:
            candidates=[]
            if self.candidate_provider:
                for domain in sorted(self.DOMAINS-{"STABLE_CORE"}):
                    try:candidates.extend(self.candidate_provider(session_id=session_id,domain=domain,schema_version=1,evidence=evidence_rows) or [])
                    except Exception as exc:self.provider_failures.append({"session_id":session_id,"domain":domain,"reason":type(exc).__name__});print(f"[HEBE][CONSOLIDATION_DELTA_REJECT] domain={domain} reason=candidate_provider_unavailable",flush=True)
        accepted=rejected=duplicates=0;print(f"[HEBE][CONSOLIDATION_START] run_id={run_id} session={session_id}",flush=True)
        try:
            for index,raw in enumerate(candidates):
                candidate=raw if isinstance(raw,ConsolidationCandidate) else ConsolidationCandidate(str(raw.get("domain") or ""),str(raw.get("delta_type") or ""),dict(raw.get("payload") or {}),tuple(str(x) for x in raw.get("evidence_ids") or ()),str(raw.get("idempotency_key") or ""))
                key=candidate.idempotency_key or hashlib.sha256(json.dumps([session_id,candidate.domain,candidate.delta_type,candidate.payload,candidate.evidence_ids],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                vstart=time.perf_counter();reason=self._validate(candidate)
                if not reason and available_evidence and not set(candidate.evidence_ids).issubset(available_evidence):reason="evidence_outside_session"
                self.validation_latencies.append((time.perf_counter()-vstart)*1000);committed=""
                if not reason:
                    evidence=EvidenceRef(candidate.evidence_ids[0],"live_session_timeline",candidate.evidence_ids[0],EvidenceRelation.SUPPORTS,observed_at=self.now_fn(),extractor="session_consolidator",extractor_version=self.VERSION)
                    try:
                        if candidate.domain=="OWNER_PREFERENCE":committed=self.preferences.learn(subject="leo",predicate=str(candidate.payload["predicate"]),value=candidate.payload.get("value"),evidence=evidence)[0].id
                        elif candidate.domain=="HEBE_SELF":committed=self.self_model.learn(subject=str(candidate.payload.get("subject") or "hebe"),predicate=str(candidate.payload["predicate"]),value=candidate.payload.get("value"),evidence=evidence)[0].id
                        elif candidate.domain=="LEO_LANGUAGE":
                            item,_=self.language.observe(predicate=str(candidate.payload["predicate"]),value=candidate.payload.get("value"),event_id=candidate.evidence_ids[0],evidence=evidence,explicit=bool(candidate.payload.get("explicit")));committed=item.id if item else "observation_only"
                        else:committed=str(candidate.payload.get("committed_object_ref") or "audit_only")
                    except Exception as exc:reason=str(exc)
                validator="REJECTED" if reason else "ACCEPTED"
                _,created=self.repository.record_delta(run_id=run_id,domain=candidate.domain,delta_type=candidate.delta_type,payload=candidate.payload,evidence_ids=list(candidate.evidence_ids),validator_result=validator,committed_ref=committed,idempotency_key=key,rejection_reason=reason,now=self.now_fn())
                if not created:duplicates+=1
                elif reason:rejected+=1;print(f"[HEBE][CONSOLIDATION_DELTA_REJECT] domain={candidate.domain} reason={reason}",flush=True)
                else:accepted+=1;print(f"[HEBE][CONSOLIDATION_DELTA_ACCEPT] domain={candidate.domain} ref={committed}",flush=True)
            self.repository.finish_run(run_id,status="COMPLETED",now=self.now_fn())
        except Exception:
            self.repository.finish_run(run_id,status="FAILED",now=self.now_fn());raise
        if not accepted:print(f"[HEBE][CONSOLIDATION_NO_CHANGE] run_id={run_id}",flush=True)
        self.durations.append((time.perf_counter()-started)*1000);self.last_result={"run_id":run_id,"status":"COMPLETED","accepted_deltas":accepted,"rejected_deltas":rejected,"duplicate_deltas":duplicates,"watermark":{"start":start_event,"end":end_event}};print(f"[HEBE][CONSOLIDATION_COMPLETE] run_id={run_id} accepted={accepted} rejected={rejected}",flush=True);return self.last_result
    def _validate(self,c):
        if c.domain not in self.DOMAINS:return "unknown_domain"
        if not c.evidence_ids:return "missing_evidence"
        forbidden=self.core.validate(c)
        if forbidden:return forbidden
        if c.domain=="HEBE_SELF" and c.payload.get("predicate") not in self.self_model.allowed_predicates:return "self_predicate_not_allowed"
        if c.domain=="HEBE_SELF" and c.payload.get("source_domain") not in {"hebe_expression","hebe_experience","owner_feedback_about_hebe"}:return "self_evidence_origin_required"
        if c.domain=="OWNER_PREFERENCE" and c.payload.get("predicate") not in self.preferences.allowed_predicates:return "owner_preference_not_allowed"
        if c.domain=="OWNER_PREFERENCE" and not bool(c.payload.get("explicit_owner_feedback")):return "explicit_owner_feedback_required"
        if c.domain=="LEO_LANGUAGE" and c.payload.get("predicate") not in self.language.allowed_predicates:return "language_predicate_not_allowed"
        source=str(c.payload.get("source_domain") or "")
        if c.domain=="HEBE_SELF" and source in {"leo_opinion","viewer_opinion"}:return "cross_domain_contamination"
        if c.domain=="GAME" and source=="hebe_opinion":return "cross_domain_contamination"
        if c.domain=="SOCIAL" and bool(c.payload.get("objective_truth")):return "social_hypothesis_not_objective"
        return ""
    def performance(self):return {"consolidation_duration":_metric(self.durations),"candidate_validation":_metric(self.validation_latencies)}


class ContinuityContextBuilder:
    def __init__(self,self_model,preferences,language,ledger,scene,*,now_fn=time.time):self.self_model=self_model;self.preferences=preferences;self.language=language;self.ledger=ledger;self.scene=scene;self.now_fn=now_fn;self.last_context={};self.latencies=[]
    def build(self,*,purpose,conversation=None,scene=None,open_threads=(),game=None,social=None):
        started=time.perf_counter();self_values=[x.to_dict()|{"claim_type":"self_opinion"} for x in self.self_model.current()][:5];prefs=[x.to_dict()|{"claim_type":"owner_preference"} for x in self.preferences.current("leo")][:10]
        manifest=[{"id":x["id"],"namespace":x["namespace"],"status":x["epistemic_status"],"evidence_ids":x["evidence_ids"]} for x in self_values+prefs]
        self.last_context={"purpose":purpose,"conversation":conversation or {},"scene":scene or self.scene.last_transition,"open_threads":list(open_threads)[:5],"game":game or {},"social":social or {},"self":self_values,"owner_preferences":prefs,"leo_language":{"interpretation_aliases":self.language.interpretation_aliases(),"usage":"understanding_only"},"voice_policy":"StableHebeCore + HebeVoice; never imitate Leo","action_evidence":[],"provenance_manifest":manifest,"manifest_size_bytes":len(json.dumps(manifest,ensure_ascii=False).encode())};self.latencies.append((time.perf_counter()-started)*1000);return self.last_context
    def performance(self):return {"continuity_context_build":_metric(self.latencies)}
