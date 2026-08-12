from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
import uuid
from typing import Any, Callable


class LearningRepository:
    """Persistence for Phase 5 audits and projections, never a competing truth store."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory
        self.latencies: dict[str, list[float]] = {"lookup": [], "write": [], "context": []}

    def rows(self, table: str, *, order: str = "") -> list[dict[str, Any]]:
        started = time.perf_counter(); conn = self.connection_factory(); conn.row_factory = sqlite3.Row
        allowed = {"consolidation_runs", "consolidation_deltas", "action_ledger", "temporal_maintenance_audit", "learning_observations", "scene_transitions"}
        if table not in allowed: raise ValueError("unsupported learning table")
        try:
            result = [dict(row) for row in conn.execute(f"SELECT * FROM {table}" + (f" ORDER BY {order}" if order else ""))]
            for item in result:
                for key in tuple(item):
                    if key.endswith("_json"):
                        item[key[:-5]] = json.loads(item.pop(key) or "{}")
            return result
        finally: conn.close(); self.latencies["lookup"].append((time.perf_counter()-started)*1000)

    def begin_run(self, *, session_id: str, start_event: str, end_event: str, pre_state_version: str, version: str, now: float) -> tuple[str, bool]:
        key = f"{session_id}|{start_event}|{end_event}|{version}"
        conn=self.connection_factory();conn.row_factory=sqlite3.Row;started=time.perf_counter()
        try:
            row=conn.execute("SELECT id,status FROM consolidation_runs WHERE idempotency_key=?",(key,)).fetchone()
            if row:
                if str(row["status"])=="FAILED":
                    conn.execute("UPDATE consolidation_runs SET status='RUNNING',started_at=?,completed_at=0 WHERE id=?",(now,row["id"]));conn.commit();return str(row["id"]),True
                return str(row["id"]),False
            run_id=f"consolidation_{uuid.uuid4().hex}"
            conn.execute("INSERT INTO consolidation_runs(id,session_id,input_start_event,input_end_event,pre_state_version,consolidator_version,status,started_at,completed_at,idempotency_key) VALUES(?,?,?,?,?,?,?, ?,0,?)",(run_id,session_id,start_event,end_event,pre_state_version,version,"RUNNING",now,key));conn.commit();return run_id,True
        finally:conn.close();self.latencies["write"].append((time.perf_counter()-started)*1000)

    def record_delta(self, *, run_id: str, domain: str, delta_type: str, payload: dict[str,Any], evidence_ids: list[str], validator_result: str, committed_ref: str, idempotency_key: str, rejection_reason: str, now: float) -> tuple[str,bool]:
        conn=self.connection_factory();started=time.perf_counter()
        try:
            delta_id=f"delta_{uuid.uuid4().hex}"
            cur=conn.execute("INSERT OR IGNORE INTO consolidation_deltas(id,consolidation_run_id,domain,delta_type,payload_json,evidence_ids_json,validator_result,committed_object_ref,idempotency_key,rejection_reason,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(delta_id,run_id,domain,delta_type,json.dumps(payload,ensure_ascii=False,sort_keys=True),json.dumps(evidence_ids),validator_result,committed_ref,idempotency_key,rejection_reason,now));conn.commit()
            if cur.rowcount:return delta_id,True
            row=conn.execute("SELECT id FROM consolidation_deltas WHERE idempotency_key=?",(idempotency_key,)).fetchone();return str(row[0]),False
        finally:conn.close();self.latencies["write"].append((time.perf_counter()-started)*1000)

    def finish_run(self, run_id: str, *, status: str, now: float) -> None:
        conn=self.connection_factory()
        try:conn.execute("UPDATE consolidation_runs SET status=?,completed_at=? WHERE id=?",(status,now,run_id));conn.commit()
        finally:conn.close()

    def observe(self, *, model: str, subject: str, value: str, event_id: str, at: float, explicit: bool=False) -> int:
        conn=self.connection_factory()
        try:
            conn.execute("INSERT OR IGNORE INTO learning_observations(id,model,subject,value,event_id,observed_at,explicit) VALUES(?,?,?,?,?,?,?)",(f"obs_{uuid.uuid4().hex}",model,subject,value,event_id,at,int(explicit)));conn.commit()
            return int(conn.execute("SELECT COUNT(*) FROM learning_observations WHERE model=? AND subject=? AND value=?",(model,subject,value)).fetchone()[0])
        finally:conn.close()

    def project_action(self, *, receipt_id: str, action_type: str, target: str, status: str, source_store: str, source_record_id: str, requested_at: float, completed_at: float, evidence: dict[str,Any]) -> dict[str,Any]:
        started=time.perf_counter();conn=self.connection_factory();conn.row_factory=sqlite3.Row
        try:
            conn.execute("INSERT INTO action_ledger(id,action_type,target,status,source_store,source_record_id,requested_at,completed_at,evidence_json,schema_version) VALUES(?,?,?,?,?,?,?,?,?,1) ON CONFLICT(source_store,source_record_id) DO UPDATE SET status=excluded.status,completed_at=excluded.completed_at,evidence_json=excluded.evidence_json",(receipt_id,action_type,target,status,source_store,source_record_id,requested_at,completed_at,json.dumps(evidence,ensure_ascii=False)));conn.commit()
            row=conn.execute("SELECT * FROM action_ledger WHERE source_store=? AND source_record_id=?",(source_store,source_record_id)).fetchone();return dict(row)
        finally:conn.close();self.latencies["write"].append((time.perf_counter()-started)*1000)

    def audit_maintenance(self, *, object_ref: str, object_type: str, old_status: str, new_status: str, reason: str, at: float, policy: str) -> None:
        conn=self.connection_factory()
        try:conn.execute("INSERT INTO temporal_maintenance_audit(id,object_ref,object_type,old_status,new_status,reason,changed_at,policy_version) VALUES(?,?,?,?,?,?,?,?)",(f"maintenance_{uuid.uuid4().hex}",object_ref,object_type,old_status,new_status,reason,at,policy));conn.commit()
        finally:conn.close()

    def save_scene(self, *, event_id: str, transition_type: str, destination: str, payload: dict[str,Any], at: float) -> dict[str,Any]:
        conn=self.connection_factory();conn.row_factory=sqlite3.Row
        try:
            conn.execute("INSERT OR IGNORE INTO scene_transitions(id,source_event_id,transition_type,destination_ref,payload_json,created_at) VALUES(?,?,?,?,?,?)",(f"scene_{uuid.uuid4().hex}",event_id,transition_type,destination,json.dumps(payload,ensure_ascii=False),at));conn.commit();row=conn.execute("SELECT * FROM scene_transitions WHERE source_event_id=? AND transition_type=?",(event_id,transition_type)).fetchone();return dict(row)
        finally:conn.close()

    def performance(self) -> dict[str,dict[str,float|int]]:
        result={}
        for key,values in self.latencies.items():
            ordered=sorted(values);result[key]={"count":len(ordered),"p50_ms":round(statistics.median(ordered),6) if ordered else 0.0,"p95_ms":round(ordered[max(0,math.ceil(len(ordered)*.95)-1)],6) if ordered else 0.0}
        return result

    def timeline_evidence(self, *, session_id: str, limit: int = 250) -> list[dict[str,Any]]:
        """Bounded canonical evidence projection; ordering uses the timeline's stable row order."""
        conn=self.connection_factory();conn.row_factory=sqlite3.Row
        try:
            tables={str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "live_session_timeline" not in tables:return []
            columns={str(r[1]) for r in conn.execute("PRAGMA table_info(live_session_timeline)")}
            wanted=[x for x in ("event_uid","event_type","source","speaker","normalized_text","context_kind","source_record_type","source_record_id","authority","valid_from","valid_until","schema_version") if x in columns]
            where=" WHERE session_id=?" if "session_id" in columns and session_id not in {"","closed_stream"} else ""
            params=(session_id,) if where else ()
            rows=conn.execute(f"SELECT {','.join(wanted)} FROM live_session_timeline{where} ORDER BY rowid DESC LIMIT ?",(*params,int(limit))).fetchall()
            return [dict(r) for r in reversed(rows)]
        finally:conn.close()
