from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .models import CLASSIFICATIONS
from .scanner import IntegrityScanner


class HygienePlanner:
    """Deterministic classifier. Apply mode records/isolate-safe changes only."""

    def __init__(self, db_path: str | Path, *, now: float | None = None) -> None:
        self.db_path=Path(db_path).resolve();self.now=float(now if now is not None else time.time())

    def _connect(self, *, readonly: bool) -> sqlite3.Connection:
        if readonly:
            conn=sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro",uri=True)
        else:conn=sqlite3.connect(self.db_path)
        conn.row_factory=sqlite3.Row;return conn

    def plan(self) -> dict:
        records=[]
        with closing(self._connect(readonly=True)) as conn:
            tables={str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "memory_facts" in tables:
                columns={str(r[1]) for r in conn.execute("PRAGMA table_info(memory_facts)")}
                for row in conn.execute("SELECT * FROM memory_facts ORDER BY id"):
                    active=bool(row["active"]); belief_id=str(row["belief_id"] or "") if "belief_id" in columns else ""
                    source=str(row["source_text"] or "").strip() if "source_text" in columns else ""
                    if belief_id: classification,reason="ARCHIVE","v2 belief projection; retain compatibility history but not independent truth"
                    elif not active:classification,reason="KEEP","inactive legacy history"
                    elif source:classification,reason="NEEDS_REVIEW","active legacy semantic row has source text but no structured v2 provenance"
                    else:classification,reason="INVALIDATE","active legacy row lacks reconstructable provenance"
                    records.append(self._record("memory_facts",row["id"],classification,reason,source_present=bool(source)))
            if "memory_chunks" in tables:
                columns={str(r[1]) for r in conn.execute("PRAGMA table_info(memory_chunks)")}
                for row in conn.execute("SELECT id,kind,active"+(",belief_id" if "belief_id" in columns else "")+" FROM memory_chunks ORDER BY id"):
                    belief_id=str(row["belief_id"] or "") if "belief_id" in columns else ""
                    if not row["active"]:classification,reason="KEEP","inactive retrieval history"
                    elif belief_id:classification,reason="KEEP","retrieval cache linked to canonical structured belief"
                    elif str(row["kind"]).casefold() in {"fact","belief","knowledge"}:classification,reason="NEEDS_REVIEW","semantic-looking vector row has no canonical owner"
                    else:classification,reason="KEEP","retrieval-only cache; never canonical truth"
                    records.append(self._record("memory_chunks",row["id"],classification,reason,belief_linked=bool(belief_id)))
            if "beliefs" in tables:
                evidence=set(str(r[0]) for r in conn.execute("SELECT DISTINCT belief_id FROM belief_evidence")) if "belief_evidence" in tables else set()
                for row in conn.execute("SELECT id,epistemic_status,superseded_by,valid_until,relevance_until FROM beliefs ORDER BY id"):
                    status=str(row["epistemic_status"])
                    if status in {"SUPERSEDED","HISTORICAL","REJECTED"}:classification,reason="KEEP","v2 history and audit trail"
                    elif str(row["id"]) not in evidence:classification,reason="INVALIDATE","active belief lacks evidence"
                    elif row["valid_until"] and float(row["valid_until"])<=self.now:classification,reason="ARCHIVE","validity elapsed"
                    else:classification,reason="KEEP","active v2 belief with evidence"
                    records.append(self._record("beliefs",row["id"],classification,reason))
            if "conversations" in tables:
                for row in conn.execute("SELECT id,status,expires_at FROM conversations ORDER BY id"):
                    expired=float(row["expires_at"])<=self.now
                    classification="ARCHIVE" if expired and str(row["status"]) in {"WAITING_ON_LEO","WAITING_ON_HEBE","ACTIVE"} else "KEEP"
                    reason="expired immediate state must not remain actionable" if classification=="ARCHIVE" else "valid transient/audit state"
                    records.append(self._record("conversations",row["id"],classification,reason))
            if "social_episodes" in tables:
                for row in conn.execute("SELECT id,retention_until,relevance_until,sensitivity,retrieval_scope FROM social_episodes ORDER BY id"):
                    if str(row["sensitivity"]) in {"private","sensitive"} and str(row["retrieval_scope"])=="stream_public":classification,reason="INVALIDATE","sensitive social record violates public retrieval boundary"
                    elif float(row["retention_until"])<=self.now:classification,reason="ARCHIVE","retention elapsed; preserve until explicit privacy purge policy"
                    else:classification,reason="KEEP","social episode remains within retention policy"
                    records.append(self._record("social_episodes",row["id"],classification,reason))
            if "schedule_hypotheses" in tables:
                for row in conn.execute("SELECT rowid,* FROM schedule_hypotheses ORDER BY rowid"):
                    records.append(self._record("schedule_hypotheses",row[0],"NEEDS_REVIEW","legacy schedule hypothesis cannot outrank current Twitch metadata automatically"))
        counts=Counter(r["classification"] for r in records)
        return {"schema_version":1,"planner":"phase6-hygiene-v1","mode":"dry-run","db":str(self.db_path),"db_sha256":IntegrityScanner.fingerprint(self.db_path),"generated_at_epoch":self.now,"classification_counts":{key:int(counts.get(key,0)) for key in CLASSIFICATIONS},"destructive_changes":0,"records":records}

    def apply_safe(self, plan: dict | None = None) -> dict:
        plan=plan or self.plan();run_id="hygiene_"+hashlib.sha256((plan["db_sha256"]+str(self.now)).encode()).hexdigest()[:20]
        before=IntegrityScanner.fingerprint(self.db_path);applied=[]
        with closing(self._connect(readonly=False)) as conn:
            required={"cognitive_migration_audit","cognitive_hygiene_runs"}
            tables={str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not required.issubset(tables):raise RuntimeError("Phase 6 migrations must run before hygiene apply")
            now_iso=datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT OR IGNORE INTO cognitive_hygiene_runs(run_id,db_fingerprint_before,mode,started_at) VALUES(?,?,?,?)",(run_id,before,"safe-apply",now_iso))
            for item in plan["records"]:
                classification=item["classification"]
                # Only deterministic isolation is applied. No DELETE/MERGE/MIGRATE is automatic here.
                if classification=="INVALIDATE" and item["store"]=="memory_facts":
                    conn.execute("UPDATE memory_facts SET active=0 WHERE CAST(id AS TEXT)=? AND active=1",(item["record_id"],));applied.append(item)
                elif classification=="ARCHIVE" and item["store"]=="conversations":
                    conn.execute("UPDATE conversations SET status='ARCHIVED',closure_reason='phase6_expired_hygiene',version=version+1 WHERE id=? AND status IN ('WAITING_ON_LEO','WAITING_ON_HEBE','ACTIVE')",(item["record_id"],));applied.append(item)
                else:continue
                conn.execute("""INSERT OR IGNORE INTO cognitive_migration_audit
                    (id,run_id,operation,source_store,source_record_id,target_store,target_record_id,classification,reason,provenance_json,applied_at)
                    VALUES(?,?,?,?,?,'','',?,?,?,?)""",(f"audit_{uuid.uuid4().hex}",run_id,"isolate",item["store"],item["record_id"],classification,item["reason"],json.dumps(item["provenance"],sort_keys=True),now_iso))
            counts=Counter(x["classification"] for x in applied)
            conn.execute("UPDATE cognitive_hygiene_runs SET completed_at=?,classification_counts_json=?,destructive_changes=0 WHERE run_id=?",(now_iso,json.dumps(dict(counts),sort_keys=True),run_id));conn.commit()
        return {**plan,"mode":"safe-apply","run_id":run_id,"applied_count":len(applied),"applied_counts":dict(counts),"destructive_changes":0,"db_sha256_before":before,"db_sha256_after":IntegrityScanner.fingerprint(self.db_path)}

    @staticmethod
    def _record(store, record_id, classification, reason, **provenance):
        return {"store":store,"record_id":str(record_id),"classification":classification,"reason":reason,"provenance":provenance}

    @staticmethod
    def markdown(plan: dict) -> str:
        lines=["# Cognitive Data Hygiene Plan","",f"- Mode: **{plan['mode']}**",f"- Destructive changes: **{plan['destructive_changes']}**","","## Classification counts","","| Classification | Count |","|---|---:|"]
        lines += [f"| {key} | {plan['classification_counts'].get(key,0)} |" for key in CLASSIFICATIONS]
        lines += ["","## Semantic or destructive classifications",""]
        selected=[r for r in plan["records"] if r["classification"] not in {"KEEP"}]
        lines += [f"- `{r['classification']}` `{r['store']}:{r['record_id']}` — {r['reason']}" for r in selected] or ["None."]
        return "\n".join(lines)+"\n"
