from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from app.epistemics.models import Belief, BeliefStatus, EvidenceRef, EvidenceRelation
from app.epistemics.repository import BeliefRepository, InvalidBeliefTransition


class BeliefLifecycleService:
    """Deterministic admission and lifecycle policy; model output is proposal only."""
    def __init__(self, repository: BeliefRepository, *, now_fn: Callable[[],float]=time.time) -> None:
        self.repository=repository; self.now_fn=now_fn; self.write_latencies=[]; self.last_transition={}; self.rejections=[]

    def propose(self, *, namespace:str,scope_kind:str,scope_id:str,subject_ref:str,predicate:str,object_value:Any,
                confidence:float,authority_class:str,evidence:EvidenceRef,status:BeliefStatus=BeliefStatus.INFERRED,
                sensitivity:str="normal",valid_until:float=0,relevance_until:float=0,owner_confirmed:bool=False)->Belief|None:
        start=time.perf_counter(); now=self.now_fn()
        reason=self._validate(status=status,authority=authority_class,evidence=evidence,owner_confirmed=owner_confirmed)
        if reason:
            self._reject(reason,namespace,subject_ref,predicate); return None
        # Only the correction/confirmation API can create owner-confirmed KNOWN beliefs.
        if status==BeliefStatus.KNOWN or owner_confirmed:
            self._reject("invalid_transition",namespace,subject_ref,predicate); return None
        belief=Belief(id=f"belief_{uuid.uuid4().hex}",namespace=namespace,scope_kind=scope_kind,scope_id=scope_id,
            subject_ref=subject_ref,predicate=predicate,object_value=object_value,epistemic_status=status,
            confidence=max(0,min(1,float(confidence))),authority_class=authority_class,created_at=now,last_confirmed_at=now,
            valid_from=now,valid_until=float(valid_until or 0),relevance_until=float(relevance_until or 0),sensitivity=sensitivity)
        current=self.repository.active_for_identity(namespace=namespace,scope_kind=scope_kind,scope_id=scope_id,subject_ref=subject_ref,predicate=predicate)
        stronger=next((b for b in current if b.owner_confirmed and b.epistemic_status==BeliefStatus.KNOWN),None)
        if stronger and stronger.object_value!=object_value:
            self.repository.add_evidence(
                stronger.id, replace(evidence, relation=EvidenceRelation.CONTRADICTS),
                subject_key=self.repository._subject_key(stronger),
            )
            self._reject("authority_conflict",namespace,subject_ref,predicate)
            self.last_transition={"operation":"contradict","active_belief_id":stronger.id,"decision":"retained_owner_truth"}
            return None
        result,created=self.repository.propose(belief,evidence)
        self._finish(start); self.last_transition={"operation":"propose","belief_id":result.id,"created":created}
        print(f"[HEBE][BELIEF_PROPOSE] belief_id={result.id} namespace={namespace} status={status.value} confidence={belief.confidence:.3f} evidence_ids={list(result.evidence_ids)}",flush=True)
        return result

    def correct(self, old_belief_id:str, *, object_value:Any,evidence:EvidenceRef,authority_class:str="owner")->Belief:
        start=time.perf_counter(); old=self.repository.get(old_belief_id)
        if old is None: raise KeyError(old_belief_id)
        if authority_class!="owner" or evidence.relation!=EvidenceRelation.CORRECTS or not evidence.source_event_id:
            raise InvalidBeliefTransition("authorized_owner_correction_required")
        now=self.now_fn(); new=Belief(id=f"belief_{uuid.uuid4().hex}",namespace=old.namespace,scope_kind=old.scope_kind,scope_id=old.scope_id,
            subject_ref=old.subject_ref,predicate=old.predicate,object_value=object_value,epistemic_status=BeliefStatus.KNOWN,
            confidence=1.0,authority_class="owner",created_at=now,last_confirmed_at=now,valid_from=now,valid_until=0,
            relevance_until=0,owner_confirmed=True,sensitivity=old.sensitivity)
        old2,new2,created=self.repository.correct(old.id,new,evidence); self._finish(start)
        self.last_transition={"operation":"correct","old_belief_id":old2.id,"new_belief_id":new2.id,"created":created}
        print(f"[HEBE][BELIEF_CORRECT] old_belief={old2.id} new_belief={new2.id} authority=owner decision=supersede",flush=True)
        return new2

    def seed_known(self, **kwargs)->Belief:
        """Trusted fixture/domain-validation seam, never callable from model output."""
        evidence=kwargs.pop("evidence"); now=self.now_fn()
        if kwargs.get("authority_class")!="owner" or evidence.relation not in {EvidenceRelation.SUPPORTS,EvidenceRelation.CORRECTS}: raise InvalidBeliefTransition("owner_evidence_required")
        b=Belief(id=f"belief_{uuid.uuid4().hex}",epistemic_status=BeliefStatus.KNOWN,confidence=1.0,created_at=now,last_confirmed_at=now,valid_from=now,valid_until=0,relevance_until=0,owner_confirmed=True,sensitivity="normal",**kwargs)
        return self.repository.propose(b,evidence)[0]

    def support(self, belief_id:str, *, evidence:EvidenceRef)->Belief:
        belief=self._require(belief_id)
        if evidence.relation!=EvidenceRelation.SUPPORTS: raise InvalidBeliefTransition("support_evidence_required")
        self.repository.add_evidence(belief.id,evidence,subject_key=self.repository._subject_key(belief))
        result=self.repository.transition(belief.id,last_confirmed_at=self.now_fn())
        self.last_transition={"operation":"support","belief_id":belief.id}
        print(f"[HEBE][BELIEF_SUPPORT] belief_id={belief.id} evidence={evidence.source_event_id}",flush=True)
        return result

    def contradict(self, belief_id:str, *, evidence:EvidenceRef)->Belief:
        belief=self._require(belief_id)
        if evidence.relation!=EvidenceRelation.CONTRADICTS: raise InvalidBeliefTransition("contradiction_evidence_required")
        self.repository.add_evidence(belief.id,evidence,subject_key=self.repository._subject_key(belief))
        self.last_transition={"operation":"contradict","belief_id":belief.id,"decision":"record_only"}
        return self._require(belief.id)

    def confirm(self, belief_id:str, *, evidence:EvidenceRef, authority_class:str="owner")->Belief:
        belief=self._require(belief_id)
        if authority_class!="owner" or evidence.relation!=EvidenceRelation.SUPPORTS: raise InvalidBeliefTransition("owner_confirmation_required")
        self.repository.add_evidence(belief.id,evidence,subject_key=self.repository._subject_key(belief))
        result=self.repository.transition(belief.id,status=BeliefStatus.KNOWN,owner_confirmed=True,authority_class="owner",confidence=1.0,last_confirmed_at=self.now_fn())
        self.last_transition={"operation":"confirm","belief_id":belief.id}
        return result

    def supersede(self, belief_id:str, *, superseded_by:str, at:float|None=None)->Belief:
        self._require(superseded_by)
        result=self.repository.transition(belief_id,status=BeliefStatus.SUPERSEDED,superseded_by=superseded_by,valid_until=float(at if at is not None else self.now_fn()))
        self.last_transition={"operation":"supersede","belief_id":belief_id,"superseded_by":superseded_by}
        print(f"[HEBE][BELIEF_SUPERSEDE] old_belief={belief_id} new_belief={superseded_by}",flush=True)
        return result

    def mark_historical(self, belief_id:str, *, valid_until:float|None=None)->Belief:
        result=self.repository.transition(belief_id,status=BeliefStatus.HISTORICAL,valid_until=float(valid_until if valid_until is not None else self.now_fn()))
        self.last_transition={"operation":"mark_historical","belief_id":belief_id}
        return result

    def expire_validity(self, belief_id:str, *, at:float|None=None)->Belief:
        result=self.repository.transition(belief_id,valid_until=float(at if at is not None else self.now_fn()))
        self.last_transition={"operation":"expire_validity","belief_id":belief_id}
        return result

    def archive_relevance(self, belief_id:str, *, at:float|None=None)->Belief:
        result=self.repository.transition(belief_id,relevance_until=float(at if at is not None else self.now_fn()))
        self.last_transition={"operation":"archive_relevance","belief_id":belief_id}
        return result

    def _require(self, belief_id:str)->Belief:
        belief=self.repository.get(belief_id)
        if belief is None: raise KeyError(belief_id)
        return belief

    def _validate(self,*,status,authority,evidence,owner_confirmed):
        if not evidence.source_event_id or not evidence.source_record_type or not evidence.source_record_id: return "no_evidence"
        if authority in {"extractor","model"} and not evidence.literal_span: return "no_literal_span"
        if status==BeliefStatus.KNOWN and not (authority=="owner" and owner_confirmed): return "invalid_transition"
        return ""
    def _reject(self,reason,namespace,subject,predicate):
        item={"reason":reason,"namespace":namespace,"subject_ref":subject,"predicate":predicate}; self.rejections.append(item); self.last_transition={"operation":"reject",**item}; print(f"[HEBE][BELIEF_REJECT] reason={reason} namespace={namespace} subject={subject} predicate={predicate}",flush=True)
    def _finish(self,start): self.write_latencies.append((time.perf_counter()-start)*1000)
    def performance(self):
        v=sorted(self.write_latencies)
        return {"count":len(v),"p50_ms":round(statistics.median(v),6) if v else 0.0,"p95_ms":round(v[max(0,__import__('math').ceil(len(v)*.95)-1)],6) if v else 0.0}
