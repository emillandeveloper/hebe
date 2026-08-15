from __future__ import annotations

import math
import re
import statistics
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from app.epistemics.models import Belief, BeliefStatus, EvidenceRef, EvidenceRelation
from app.epistemics.service import BeliefLifecycleService
from app.game_context_v2.models import GameKnowledgeGap, GameRun, GameRunResolution, GameRunStatus
from app.game_context_v2.repository import GameV2Repository


class GameRunService:
    def __init__(self,repository:GameV2Repository,beliefs:BeliefLifecycleService,*,now_fn:Callable[[],float]=time.time)->None:
        self.repository=repository;self.beliefs=beliefs;self.now_fn=now_fn;self.last_resolution={};self.resolution_latencies=[]

    _STATE_FIELDS = {
        "platform_version", "playthrough_type", "spoiler_policy", "current_location",
        "current_character", "party_members", "last_confirmed_progress", "current_objective",
        "challenge", "known_constraints", "current_chapter", "encountered_characters",
        "encountered_bosses", "unlocked_mechanics", "progress_markers",
    }
    _GUARDED_FIELDS = {
        "current_location", "current_character", "party_members",
        "current_objective", "last_confirmed_progress",
    }
    _OWNER_PROVENANCE = {"leo_clarification", "manual_command", "owner_explicit"}

    def resolve(self,*,game:str,stream_session_id:str,source_event_id:str,owner_id:str="leo",run_kind:str="unknown",rules:dict[str,Any]|None=None,explicit_new:bool=False,explicit_continue:bool=False)->GameRunResolution:
        started=time.perf_counter();now=self.now_fn();identity=self.repository.resolve_identity(game)
        active=self.repository.list_runs(owner_id=owner_id,statuses=(GameRunStatus.ACTIVE.value,))
        for other in active:
            if other.game_id!=identity.game_id:self.repository.set_run_status(other.id,GameRunStatus.PAUSED,at=now)
        candidates=self.repository.list_runs(game_id=identity.game_id,owner_id=owner_id,statuses=(GameRunStatus.ACTIVE.value,GameRunStatus.PAUSED.value))
        decision="ambiguous";confidence=.45;reason="multiple_candidate_runs";run=None
        if explicit_new:
            for old in candidates:
                if old.status==GameRunStatus.ACTIVE:self.repository.set_run_status(old.id,GameRunStatus.PAUSED,at=now)
            run=self.repository.create_run(GameRun(f"game_run_{uuid.uuid4().hex}",identity.game_id,owner_id,run_kind or "unknown",dict(rules or {}),GameRunStatus.ACTIVE,now,now,0,1,source_event_id))
            decision="new";confidence=1.0;reason="explicit_owner_new_run"
        elif len(candidates)==1:
            run=candidates[0]
            if run.status!=GameRunStatus.ACTIVE:run=self.repository.set_run_status(run.id,GameRunStatus.ACTIVE,at=now)
            decision="resume";confidence=.98 if explicit_continue else .86;reason="explicit_continuation" if explicit_continue else "single_compatible_run"
        elif not candidates:
            run=self.repository.create_run(GameRun(f"game_run_{uuid.uuid4().hex}",identity.game_id,owner_id,run_kind or "unknown",dict(rules or {}),GameRunStatus.ACTIVE,now,now,0,1,source_event_id))
            decision="new";confidence=.92 if run_kind!="unknown" else .75;reason="no_compatible_run"
        if run is not None:
            link=self.repository.link_session(run.id,str(stream_session_id),at=now,evidence_event_id=source_event_id)
            if link["created"]:self.repository.add_event(run_id=run.id,event_type="run_started" if decision=="new" else "run_resumed",subject_ref="run",predicate="lifecycle",object_value=decision,evidence_event_id=source_event_id,observed_at=now,epistemic_status="KNOWN")
        latency=(time.perf_counter()-started)*1000
        self.resolution_latencies.append(latency)
        result=GameRunResolution(identity,run,confidence,decision,(source_event_id,),reason,latency);self.last_resolution=result.to_dict()
        print(f"[HEBE][GAME_RUN_RESOLVE] game={identity.game_id} run_id={getattr(run,'id','')} decision={decision} confidence={confidence:.3f} evidence={[source_event_id]}",flush=True)
        return result

    def state(self, run_id: str) -> dict[str, Any]:
        run=self.repository.get_run(run_id)
        if run is None:raise KeyError(run_id)
        identity=self.repository.get_identity(run.game_id)
        result={
            "run_id":run.id,"game_id":run.game_id,
            "game":identity.canonical_name if identity else run.game_id,
            "platform_version":"","playthrough_type":run.run_kind or "unknown",
            "spoiler_policy":"spoiler_safe_hints","current_location":"",
            "current_character":"","party_members":[],"last_confirmed_progress":"",
            "current_objective":"","challenge":str(run.rules.get("challenge") or ""),
            "known_constraints":list(run.rules.get("known_constraints") or []),
            "current_chapter":"","encountered_characters":[],"encountered_bosses":[],
            "unlocked_mechanics":[],"progress_markers":[],"last_updated":run.last_active_at,
            "provenance":"canonical_game_run","confidence":0.0,"status":run.status.value,
        }
        facts=self.beliefs.repository.list(
            namespace="game_run",scope_kind="game_run",scope_id=run_id,subject_ref="run_state",
        )
        current=[item for item in facts if item.epistemic_status not in {
            BeliefStatus.SUPERSEDED,BeliefStatus.HISTORICAL,BeliefStatus.REJECTED
        } and not item.superseded_by]
        for fact in reversed(current):
            if fact.predicate in self._STATE_FIELDS:
                result[fact.predicate]=fact.object_value
                result["last_updated"]=max(float(result["last_updated"]),fact.last_confirmed_at)
                result["confidence"]=max(float(result["confidence"]),fact.confidence)
                result["provenance"]=fact.authority_class
        return result

    def update_state(
        self,run_id:str,*,updates:dict[str,Any],provenance:str,confidence:float,
        evidence:EvidenceRef,
    )->dict[str,Any]:
        if self.repository.get_run(run_id) is None:raise KeyError(run_id)
        accepted:dict[str,Any]={};rejected:dict[str,str]={};belief_ids:dict[str,str]={}
        for field_name,value in updates.items():
            if field_name not in self._STATE_FIELDS:
                rejected[field_name]="unknown_run_state_field";continue
            ok,reason,cleaned=self._guard_state_value(
                field_name,value,provenance=provenance,confidence=confidence,
            )
            if not ok:
                rejected[field_name]=reason
                print(f"[HEBE][GAME_RUN_STATE_WRITE_GUARD] accepted=false field={field_name} reason={reason}",flush=True)
                continue
            current=self.beliefs.repository.active_for_identity(
                namespace="game_run",scope_kind="game_run",scope_id=run_id,
                subject_ref="run_state",predicate=field_name,
            )
            existing=current[0] if current else None
            if existing is not None and existing.object_value==cleaned:
                self.beliefs.support(existing.id,evidence=evidence)
                belief=existing;changed=False
            elif existing is not None:
                belief=self.beliefs.correct(
                    existing.id,object_value=cleaned,
                    evidence=replace(evidence,relation=EvidenceRelation.CORRECTS),authority_class="owner",
                );changed=True
            else:
                belief=self.beliefs.seed_known(
                    namespace="game_run",scope_kind="game_run",scope_id=run_id,
                    subject_ref="run_state",predicate=field_name,object_value=cleaned,
                    authority_class="owner",evidence=evidence,
                );changed=True
            if changed:
                self.repository.add_event(
                    run_id=run_id,event_type="run_state_updated",subject_ref="run_state",
                    predicate=field_name,object_value=cleaned,evidence_event_id=evidence.source_event_id,
                    belief_id=belief.id,observed_at=evidence.observed_at,
                    epistemic_status=belief.epistemic_status.value,
                )
            accepted[field_name]=cleaned;belief_ids[field_name]=belief.id
            print(f"[HEBE][GAME_RUN_STATE_WRITE_GUARD] accepted=true field={field_name} reason=accepted",flush=True)
        return {"accepted":accepted,"rejected":rejected,"belief_ids":belief_ids,"state":self.state(run_id)}

    def clear_state(self,run_id:str,*,fields:tuple[str,...]|list[str],evidence:EvidenceRef)->dict[str,Any]:
        if self.repository.get_run(run_id) is None:raise KeyError(run_id)
        cleared=[]
        for field_name in fields:
            if field_name not in self._STATE_FIELDS:continue
            current=self.beliefs.repository.active_for_identity(
                namespace="game_run",scope_kind="game_run",scope_id=run_id,
                subject_ref="run_state",predicate=field_name,
            )
            for belief in current:
                self.beliefs.repository.add_evidence(
                    belief.id,replace(evidence,relation=EvidenceRelation.CORRECTS),
                    subject_key=self.beliefs.repository._subject_key(belief),
                )
                self.beliefs.mark_historical(belief.id,valid_until=evidence.observed_at)
            if current:
                self.repository.add_event(
                    run_id=run_id,event_type="run_state_cleared",subject_ref="run_state",
                    predicate=field_name,object_value=None,evidence_event_id=evidence.source_event_id,
                    observed_at=evidence.observed_at,epistemic_status="HISTORICAL",
                );cleared.append(field_name)
        return {"cleared":cleared,"state":self.state(run_id)}

    def _guard_state_value(
        self,field_name:str,value:Any,*,provenance:str,confidence:float,
    )->tuple[bool,str,Any]:
        if field_name in self._GUARDED_FIELDS and (
            provenance not in self._OWNER_PROVENANCE or float(confidence or 0.0)<.75
        ):
            return False,"low_confidence_or_missing_provenance",value
        values=value if isinstance(value,list) else [value]
        cleaned=[]
        for item in values:
            raw=str(item or "").strip();text=" ".join(re.sub(r"[^a-z0-9 ]+"," ",raw.casefold()).split())
            if not text:return False,"empty_value",value
            if len(text)>70 or len(text.split())>6:return False,"sentence_fragment",value
            if re.search(r"\b(?:familia|madre|padre|herman[oa]s?|anime|artes? marciales?|tranquilamente|viewer|espectador)\b",text):
                return False,"real_life_or_stt_junk",value
            if field_name=="current_location" and not re.search(
                r"\b(?:palace|palacio|castle|castillo|dungeon|mazmorra|temple|templo|city|ciudad|zone|zona|level|nivel|mementos|shibuya|kamoshida|midgar|gaia|alexandria|lindblum|burmecia|karnak)\b",text,
            ):
                return False,"location_not_game_like",value
            if field_name in {"current_character","party_members"} and re.search(r"\b(?:eh|vale|pues|nada)\b",text):
                return False,"character_not_entity_like",value
            cleaned.append(raw)
        return True,"accepted",cleaned if isinstance(value,list) else cleaned[0]

    def performance(self)->dict[str,float|int]:
        values=sorted(self.resolution_latencies)
        return {"count":len(values),"p50_ms":round(statistics.median(values),6) if values else 0.0,"p95_ms":round(values[max(0,math.ceil(len(values)*.95)-1)],6) if values else 0.0}

    def pause(self,run_id:str,*,stream_session_id:str,event_id:str)->GameRun:
        now=self.now_fn();self.repository.end_session(run_id,stream_session_id,at=now);run=self.repository.set_run_status(run_id,GameRunStatus.PAUSED,at=now)
        self.repository.add_event(run_id=run_id,event_type="run_paused",subject_ref="run",predicate="lifecycle",object_value="paused",evidence_event_id=event_id,observed_at=now,epistemic_status="KNOWN");return run

    def finish(self,run_id:str,*,status:GameRunStatus=GameRunStatus.COMPLETED,event_id:str)->GameRun:
        run=self.repository.set_run_status(run_id,status,at=self.now_fn(),ended=True)
        self.repository.add_event(run_id=run_id,event_type="run_completed" if status==GameRunStatus.COMPLETED else "run_ended",subject_ref="run",predicate="lifecycle",object_value=status.value,evidence_event_id=event_id,observed_at=self.now_fn(),epistemic_status="KNOWN");return run

    def record_fact(self,run_id:str,*,subject_ref:str,predicate:str,object_value:Any,evidence:EvidenceRef,event_type:str="notable_run_event",owner_confirmed:bool=False,confidence:float=.7,entailment_valid:bool=True)->Belief|None:
        run=self.repository.get_run(run_id)
        if run is None:raise KeyError(run_id)
        if not entailment_valid or not evidence.literal_span:
            self.beliefs.last_transition={"operation":"reject","reason":"unsupported_run_inference","subject_ref":subject_ref,"predicate":predicate};print(f"[HEBE][BELIEF_REJECT] reason=unsupported_run_inference namespace=game_run subject={subject_ref} predicate={predicate}",flush=True);return None
        common=dict(namespace="game_run",scope_kind="game_run",scope_id=run_id,subject_ref=subject_ref,predicate=predicate,object_value=object_value,authority_class="owner" if owner_confirmed else "extractor",evidence=evidence)
        belief=self.beliefs.seed_known(**common) if owner_confirmed else self.beliefs.propose(**common,confidence=confidence,status=BeliefStatus.INFERRED)
        if belief:
            self.repository.add_event(run_id=run_id,event_type=event_type,subject_ref=subject_ref,predicate=predicate,object_value=object_value,evidence_event_id=evidence.source_event_id,belief_id=belief.id,observed_at=evidence.observed_at,epistemic_status=belief.epistemic_status.value)
            print(f"[HEBE][GAME_RUN_FACT] run_id={run_id} predicate={predicate} status={belief.epistemic_status.value} evidence={evidence.source_event_id}",flush=True)
        return belief

    def correct_fact(self,belief_id:str,*,object_value:Any,evidence:EvidenceRef)->Belief:
        old=self.beliefs.repository.get(belief_id)
        if old is None or old.namespace!="game_run" or old.scope_kind!="game_run":raise KeyError(belief_id)
        new=self.beliefs.correct(old.id,object_value=object_value,evidence=replace(evidence,relation=EvidenceRelation.CORRECTS),authority_class="owner")
        self.repository.add_event(run_id=old.scope_id,event_type="owner_correction",subject_ref=old.subject_ref,predicate=old.predicate,object_value=object_value,evidence_event_id=evidence.source_event_id,belief_id=new.id,observed_at=evidence.observed_at,epistemic_status=new.epistemic_status.value)
        print(f"[HEBE][GAME_RUN_CORRECT] old={old.id} new={new.id} owner_confirmed=true",flush=True);return new

    def facts(self,run_id:str,*,historical:bool=False)->list[dict[str,Any]]:
        started=time.perf_counter();rows=self.beliefs.repository.list(namespace="game_run",scope_kind="game_run",scope_id=run_id)
        result=[row.to_dict() for row in rows if (historical or row.epistemic_status not in {BeliefStatus.SUPERSEDED,BeliefStatus.HISTORICAL,BeliefStatus.REJECTED}) and (historical or not row.superseded_by)]
        self.repository.lookup_latencies["run_fact"].append((time.perf_counter()-started)*1000);return result


class GameKnowledgeService:
    def __init__(self,repository:GameV2Repository,beliefs:BeliefLifecycleService,*,now_fn:Callable[[],float]=time.time)->None:
        self.repository=repository;self.beliefs=beliefs;self.now_fn=now_fn;self.validation_log=[]

    def add_validated(self,*,game_id:str,subject_ref:str,predicate:str,object_value:Any,confidence:float,evidence:EvidenceRef,source_type:str,source_quality:str,spoiler_class:str="safe_general_mechanic",version_tag:str="")->Belief|None:
        if not evidence.source_event_id or not evidence.source_record_id or source_type=="web" and (not evidence.literal_span or not evidence.literal_span.get("source_url")):
            self.validation_log.append({"result":"rejected","reason":"citation_or_support_missing","source_event_id":evidence.source_event_id});print(f"[HEBE][GAME_RESEARCH_VALIDATE] claim={evidence.source_event_id} result=rejected reason=citation_or_support_missing",flush=True);return None
        belief=self.beliefs.propose(namespace="game_knowledge",scope_kind="game",scope_id=game_id,subject_ref=subject_ref,predicate=predicate,object_value=object_value,confidence=confidence,authority_class="research_validator" if source_type=="web" else "domain_validator",evidence=evidence,status=BeliefStatus.INFERRED)
        if belief:
            fact_id=f"game_fact_{uuid.uuid4().hex}";self.repository.add_knowledge_link(fact_id=fact_id,game_id=game_id,belief_id=belief.id,source_type=source_type,source_quality=source_quality,spoiler_class=spoiler_class,version_tag=version_tag,created_at=self.now_fn())
            self.validation_log.append({"claim_id":belief.id,"fact_id":fact_id,"result":"accepted","reason":"validated_provenance"});print(f"[HEBE][GAME_RESEARCH_VALIDATE] claim={belief.id} result=accepted reason=validated_provenance",flush=True)
        return belief

    def find(self,game_id:str,*,subject_ref:str="",predicate:str="",spoiler_ceiling:str="safe_general_mechanic")->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
        links=self.repository.knowledge(game_id);selected=[];rejected=[]
        allowed={"safe_general_mechanic","safe_current_progress"} if spoiler_ceiling in {"safe","safe_general_mechanic","strict"} else None
        for link in links:
            belief=self.beliefs.repository.get(link["belief_id"])
            if belief is None or belief.epistemic_status in {BeliefStatus.SUPERSEDED,BeliefStatus.REJECTED}:continue
            item={**belief.to_dict(),**{key:link[key] for key in ("id","source_type","source_quality","spoiler_class","version_tag")}}
            if subject_ref and belief.subject_ref!=subject_ref:item["rejection_reason"]="subject_mismatch";rejected.append(item);continue
            if predicate and belief.predicate!=predicate:item["rejection_reason"]="predicate_mismatch";rejected.append(item);continue
            if allowed is not None and link["spoiler_class"] not in allowed:item["rejection_reason"]="spoiler_blocked";rejected.append(item);continue
            selected.append(item)
        selected.sort(key=lambda item:(item["confidence"],item["last_confirmed_at"]),reverse=True)
        return selected,rejected

    def create_gap(self,*,game_id:str,run_id:str,subject_ref:str,question_type:str,query_intent:str,spoiler_ceiling:str,required_confidence:float,event_id:str)->GameKnowledgeGap:
        key="|".join((game_id,self.repository.normalize(subject_ref),self.repository.normalize(question_type),self.repository.normalize(query_intent),spoiler_ceiling));existing=self.repository.gap(key)
        if existing:return existing
        now=self.now_fn();gap=GameKnowledgeGap(f"game_gap_{uuid.uuid4().hex}",game_id,run_id,subject_ref,question_type,query_intent,spoiler_ceiling,required_confidence,event_id,key,"open",now,now)
        self.repository.save_gap(gap);print(f"[HEBE][GAME_KNOWLEDGE_GAP] game={game_id} gap_key={key} type={question_type}",flush=True);return gap
