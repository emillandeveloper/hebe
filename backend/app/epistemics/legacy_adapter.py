from __future__ import annotations
import json,time
from typing import Any,Callable
from app.epistemics.models import BeliefStatus,EvidenceRef
from app.epistemics.service import BeliefLifecycleService

class LegacyMemoryFactAdapter:
    """Single compatibility seam. Legacy rows are shadow proposals, never assumed KNOWN."""
    def __init__(self,service:BeliefLifecycleService,connection_factory:Callable): self.service=service;self.connection_factory=connection_factory;self.telemetry={"legacy_to_v2":[],"v2_to_legacy":[],"shadow_diffs":[],"backfill":{"safe":0,"compatibility_only":0,"ambiguous":0,"invalid_stale":0}}
    def shadow_project(self,fact_id:int):
        conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try: row=conn.execute("SELECT * FROM memory_facts WHERE id=?",(fact_id,)).fetchone()
        finally: conn.close()
        if not row:return None
        payload=json.loads(row["payload_json"] or "{}"); now=self.service.now_fn()
        result=self.service.propose(namespace=f"legacy.{row['kind']}",scope_kind="owner_local",scope_id="leo",subject_ref=str(row["subject"] or "legacy"),predicate=str(payload.get("predicate") or "legacy_fact"),object_value=payload.get("value",payload),confidence=float(row["confidence"]),authority_class="legacy",status=BeliefStatus.SUSPECTED,evidence=EvidenceRef(source_event_id=f"memory_fact:{fact_id}",source_record_type="memory_facts",source_record_id=str(fact_id),observed_at=now,extractor="legacy_adapter"))
        item={"memory_fact_id":fact_id,"belief_id":getattr(result,"id",""),"classification":"COMPATIBILITY_ONLY"}
        self.telemetry["legacy_to_v2"].append(item);self.telemetry["backfill"]["compatibility_only"]+=1;return result

    def project_to_legacy(self,belief_id:str)->int|None:
        """Optional old-reader projection; all v2-to-legacy writes pass through this seam."""
        belief=self.service.repository.get(belief_id)
        if belief is None:return None
        conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:
            existing=conn.execute("SELECT id FROM memory_facts WHERE belief_id=?",(belief.id,)).fetchone()
            if existing:return int(existing["id"])
            now=str(time.time())
            cur=conn.execute(
                """INSERT INTO memory_facts(kind,subject,payload_json,source_text,confidence,created_at,updated_at,active,belief_id,epistemic_status)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (belief.namespace,belief.subject_ref,json.dumps({"predicate":belief.predicate,"value":belief.object_value},ensure_ascii=False),"",belief.confidence,now,now,int(belief.epistemic_status.value in {"KNOWN","INFERRED","SUSPECTED"}),belief.id,belief.epistemic_status.value),
            )
            conn.commit();fact_id=int(cur.lastrowid)
        finally:conn.close()
        self.telemetry["v2_to_legacy"].append({"belief_id":belief.id,"memory_fact_id":fact_id});return fact_id
