from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
import uuid
from typing import Any, Callable

from app.epistemics.models import Belief, BeliefStatus, EvidenceRef, EvidenceRelation


ACTIVE = {BeliefStatus.KNOWN.value, BeliefStatus.INFERRED.value, BeliefStatus.SUSPECTED.value}


class InvalidBeliefTransition(RuntimeError): pass


class BeliefRepository:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory
        self.lookup_latencies: list[float] = []
        self.evidence_lookup_latencies: list[float] = []
        self.write_latencies: list[float] = []

    def get(self, belief_id: str) -> Belief | None:
        started=time.perf_counter()
        conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try:
            row=conn.execute("SELECT * FROM beliefs WHERE id=?",(belief_id,)).fetchone()
            return self._from_row(conn,row) if row else None
        finally: conn.close(); self.lookup_latencies.append((time.perf_counter()-started)*1000)

    def list(self, *, namespace: str="", scope_kind: str="", scope_id: str="", subject_ref: str="", predicate: str="") -> list[Belief]:
        started=time.perf_counter()
        conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try:
            where=["1=1"]; params=[]
            for key,value in (("namespace",namespace),("scope_kind",scope_kind),("scope_id",scope_id),("subject_ref",subject_ref),("predicate",predicate)):
                if value: where.append(f"{key}=?"); params.append(value)
            rows=conn.execute("SELECT * FROM beliefs WHERE "+" AND ".join(where)+" ORDER BY created_at DESC,id",params).fetchall()
            return [self._from_row(conn,row) for row in rows]
        finally: conn.close(); self.lookup_latencies.append((time.perf_counter()-started)*1000)

    def active_for_identity(self, *, namespace: str, scope_kind: str, scope_id: str, subject_ref: str, predicate: str) -> list[Belief]:
        return [b for b in self.list(namespace=namespace,scope_kind=scope_kind,scope_id=scope_id,subject_ref=subject_ref,predicate=predicate) if b.epistemic_status.value in ACTIVE and not b.superseded_by]

    def propose(self, belief: Belief, evidence: EvidenceRef) -> tuple[Belief,bool]:
        started=time.perf_counter()
        conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            duplicate=conn.execute("SELECT belief_id FROM belief_evidence WHERE source_event_id=? AND relation=? AND subject_key=?",(evidence.source_event_id,evidence.relation.value,self._subject_key(belief))).fetchone()
            if duplicate:
                conn.rollback(); existing=self.get(str(duplicate["belief_id"])); return existing,False
            self._insert_belief(conn,belief); self._insert_evidence(conn,belief.id,evidence,belief)
            conn.commit(); return self.get(belief.id),True
        except Exception: conn.rollback(); raise
        finally: conn.close(); self.write_latencies.append((time.perf_counter()-started)*1000)

    def correct(self, old_id: str, new_belief: Belief, evidence: EvidenceRef) -> tuple[Belief,Belief,bool]:
        if evidence.relation != EvidenceRelation.CORRECTS or new_belief.authority_class != "owner" or not new_belief.owner_confirmed or new_belief.epistemic_status != BeliefStatus.KNOWN:
            raise InvalidBeliefTransition("owner_correction_contract_required")
        started=time.perf_counter(); conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            oldrow=conn.execute("SELECT * FROM beliefs WHERE id=?",(old_id,)).fetchone()
            if not oldrow: raise KeyError(old_id)
            old=self._from_row(conn,oldrow)
            if (old.namespace,old.scope_kind,old.scope_id,old.subject_ref,old.predicate)!=(new_belief.namespace,new_belief.scope_kind,new_belief.scope_id,new_belief.subject_ref,new_belief.predicate):
                raise InvalidBeliefTransition("ambiguous_correction_target")
            duplicate=conn.execute("SELECT belief_id FROM belief_evidence WHERE source_event_id=? AND relation='CORRECTS' AND subject_key=?",(evidence.source_event_id,self._subject_key(new_belief))).fetchone()
            if duplicate:
                conn.rollback(); return old,self.get(str(duplicate["belief_id"])),False
            self._insert_belief(conn,new_belief); self._insert_evidence(conn,new_belief.id,evidence,new_belief)
            conn.execute("UPDATE beliefs SET epistemic_status='SUPERSEDED',superseded_by=?,valid_until=?,version=version+1 WHERE id=?",(new_belief.id,evidence.observed_at,old_id))
            conn.commit(); return self.get(old_id),self.get(new_belief.id),True
        except Exception: conn.rollback(); raise
        finally: conn.close(); self.write_latencies.append((time.perf_counter()-started)*1000)

    def add_evidence(self, belief_id: str, evidence: EvidenceRef, *, subject_key: str) -> bool:
        started=time.perf_counter(); conn=self.connection_factory()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur=conn.execute("""INSERT OR IGNORE INTO belief_evidence(id,belief_id,source_event_id,source_record_type,source_record_id,relation,weight,observed_at,extractor,extractor_version,literal_span_json,subject_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"evidence_{uuid.uuid4().hex}",belief_id,evidence.source_event_id,evidence.source_record_type,evidence.source_record_id,evidence.relation.value,evidence.weight,evidence.observed_at,evidence.extractor,evidence.extractor_version,json.dumps(evidence.literal_span,ensure_ascii=False),subject_key))
            conn.commit(); return cur.rowcount==1
        except Exception: conn.rollback(); raise
        finally: conn.close(); self.write_latencies.append((time.perf_counter()-started)*1000)

    def transition(
        self,
        belief_id: str,
        *,
        status: BeliefStatus | None = None,
        owner_confirmed: bool | None = None,
        authority_class: str | None = None,
        confidence: float | None = None,
        last_confirmed_at: float | None = None,
        valid_until: float | None = None,
        relevance_until: float | None = None,
        superseded_by: str | None = None,
    ) -> Belief:
        """Apply a validated lifecycle transition without deleting evidence/history."""
        started=time.perf_counter(); conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM beliefs WHERE id=?",(belief_id,)).fetchone()
            if row is None: raise KeyError(belief_id)
            updates: list[str]=["version=version+1"]; values: list[Any]=[]
            for column,value in (
                ("epistemic_status",status.value if status else None),
                ("owner_confirmed",int(owner_confirmed) if owner_confirmed is not None else None),
                ("authority_class",authority_class),("confidence",confidence),
                ("last_confirmed_at",last_confirmed_at),("valid_until",valid_until),
                ("relevance_until",relevance_until),("superseded_by",superseded_by),
            ):
                if value is not None: updates.append(f"{column}=?"); values.append(value)
            values.append(belief_id)
            conn.execute("UPDATE beliefs SET "+",".join(updates)+" WHERE id=?",values)
            conn.commit()
        except Exception: conn.rollback(); raise
        finally: conn.close(); self.write_latencies.append((time.perf_counter()-started)*1000)
        result=self.get(belief_id)
        if result is None: raise KeyError(belief_id)
        return result

    def evidence_for(self, belief_id: str) -> list[dict[str,Any]]:
        started=time.perf_counter()
        conn=self.connection_factory(); conn.row_factory=sqlite3.Row
        try: return [dict(r) for r in conn.execute("SELECT * FROM belief_evidence WHERE belief_id=? ORDER BY observed_at,id",(belief_id,))]
        finally: conn.close(); self.evidence_lookup_latencies.append((time.perf_counter()-started)*1000)

    def performance(self) -> dict[str, dict[str, float | int]]:
        return {
            "belief_lookup": self._percentiles(self.lookup_latencies),
            "evidence_lookup": self._percentiles(self.evidence_lookup_latencies),
            "sqlite_write": self._percentiles(self.write_latencies),
        }

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float | int]:
        ordered=sorted(values)
        return {
            "count":len(ordered),
            "p50_ms":round(statistics.median(ordered),6) if ordered else 0.0,
            "p95_ms":round(ordered[max(0,math.ceil(len(ordered)*.95)-1)],6) if ordered else 0.0,
        }

    @staticmethod
    def _subject_key(b: Belief)->str: return "|".join((b.namespace,b.scope_kind,b.scope_id,b.subject_ref,b.predicate))

    def _insert_belief(self,conn,b):
        conn.execute("""INSERT INTO beliefs(id,namespace,scope_kind,scope_id,subject_ref,predicate,object_json,epistemic_status,confidence,authority_class,created_at,last_confirmed_at,valid_from,valid_until,relevance_until,superseded_by,owner_confirmed,sensitivity,schema_version,retention_policy,version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (b.id,b.namespace,b.scope_kind,b.scope_id,b.subject_ref,b.predicate,json.dumps(b.object_value,ensure_ascii=False),b.epistemic_status.value,b.confidence,b.authority_class,b.created_at,b.last_confirmed_at,b.valid_from,b.valid_until,b.relevance_until,b.superseded_by,int(b.owner_confirmed),b.sensitivity,b.schema_version,b.retention_policy,b.version))

    def _insert_evidence(self,conn,belief_id,e,b):
        conn.execute("""INSERT INTO belief_evidence(id,belief_id,source_event_id,source_record_type,source_record_id,relation,weight,observed_at,extractor,extractor_version,literal_span_json,subject_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"evidence_{uuid.uuid4().hex}",belief_id,e.source_event_id,e.source_record_type,e.source_record_id,e.relation.value,e.weight,e.observed_at,e.extractor,e.extractor_version,json.dumps(e.literal_span,ensure_ascii=False),self._subject_key(b)))

    def _from_row(self,conn,row):
        ids=tuple(str(r[0]) for r in conn.execute("SELECT id FROM belief_evidence WHERE belief_id=? ORDER BY observed_at,id",(row["id"],)).fetchall())
        return Belief(id=row["id"],namespace=row["namespace"],scope_kind=row["scope_kind"],scope_id=row["scope_id"],subject_ref=row["subject_ref"],predicate=row["predicate"],object_value=json.loads(row["object_json"]),epistemic_status=BeliefStatus(row["epistemic_status"]),confidence=float(row["confidence"]),authority_class=row["authority_class"],created_at=float(row["created_at"]),last_confirmed_at=float(row["last_confirmed_at"]),valid_from=float(row["valid_from"]),valid_until=float(row["valid_until"]),relevance_until=float(row["relevance_until"]),superseded_by=row["superseded_by"],owner_confirmed=bool(row["owner_confirmed"]),sensitivity=row["sensitivity"],schema_version=int(row["schema_version"]),retention_policy=row["retention_policy"],version=int(row["version"]),evidence_ids=ids)
