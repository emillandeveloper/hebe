from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Iterable

from .models import Finding, SEVERITIES


ACTIVE_BELIEF = ("KNOWN", "INFERRED", "SUSPECTED")


class IntegrityScanner:
    """Read-only cross-domain invariant scanner for a SQLite cognitive store."""

    def __init__(self, db_path: str | Path, *, now: float | None = None) -> None:
        self.db_path = Path(db_path).resolve()
        self.now = float(now if now is not None else time.time())
        self.findings: list[Finding] = []

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> set[str]:
        return {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def _add(self, check: str, category: str, severity: str, message: str, rows: Iterable = (), *, blocking: bool = False) -> None:
        ids = tuple(str(r[0] if not isinstance(r, sqlite3.Row) else r[0]) for r in rows)
        self.findings.append(Finding(check, category, severity, message, max(1, len(ids)), ids[:25], blocking))

    def _query(self, conn: sqlite3.Connection, tables: set[str], required: Iterable[str], sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        if not set(required).issubset(tables):
            return []
        return list(conn.execute(sql, params))

    def scan(self) -> dict:
        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)
        self.findings = []
        with closing(self._connect()) as conn:
            tables = self._tables(conn)
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                self._add("sqlite.integrity", "database", "ERROR", integrity, blocking=True)
            fk = list(conn.execute("PRAGMA foreign_key_check"))
            if fk:
                self._add("sqlite.foreign_keys", "database", "ERROR", "Foreign-key violations", (f"{r[0]}:{r[1]}" for r in fk), blocking=True)
            self._beliefs(conn, tables)
            self._actions(conn, tables)
            self._continuity(conn, tables)
            self._game(conn, tables)
            self._social(conn, tables)
            self._schedule(conn, tables)
            self._identity(conn, tables)
            self._consolidation(conn, tables)
        counts = Counter(item.severity for item in self.findings)
        blocking = sum(1 for item in self.findings if item.blocking)
        categories = sorted({item.category for item in self.findings})
        return {
            "schema_version": 1,
            "scanner": "phase6-integrity-v1",
            "db": str(self.db_path),
            "db_sha256": self.fingerprint(self.db_path),
            "scanned_at_epoch": self.now,
            "status": "FAIL" if blocking else "PASS",
            "blocking_error_count": blocking,
            "counts": {key: int(counts.get(key, 0)) for key in SEVERITIES},
            "categories_with_findings": categories,
            "findings": [item.to_dict() for item in self.findings],
        }

    def _beliefs(self, conn, tables) -> None:
        if "beliefs" not in tables:
            self._add("belief.schema", "beliefs", "INFO", "Belief v2 table not installed")
            return
        rows = self._query(conn, tables, ("belief_evidence",), """
            SELECT b.id FROM beliefs b LEFT JOIN belief_evidence e ON e.belief_id=b.id
            WHERE b.epistemic_status IN ('KNOWN','INFERRED','SUSPECTED')
            GROUP BY b.id HAVING COUNT(e.id)=0""")
        if rows: self._add("belief.provenance", "beliefs", "ERROR", "Active reusable beliefs without evidence", rows, blocking=True)
        rows = self._query(conn, tables, ("belief_evidence",), """
            SELECT b.id FROM beliefs b LEFT JOIN belief_evidence e ON e.belief_id=b.id
            WHERE b.owner_confirmed=1 GROUP BY b.id
            HAVING b.authority_class<>'owner' OR COUNT(e.id)=0""")
        if rows: self._add("belief.owner_authority", "beliefs", "ERROR", "Owner-confirmed beliefs lack owner authority/evidence", rows, blocking=True)
        rows = list(conn.execute("SELECT id FROM beliefs WHERE superseded_by=id AND superseded_by<>''"))
        if rows: self._add("belief.self_supersession", "beliefs", "ERROR", "Beliefs supersede themselves", rows, blocking=True)
        rows = list(conn.execute("SELECT b.id FROM beliefs b LEFT JOIN beliefs s ON s.id=b.superseded_by WHERE b.superseded_by<>'' AND s.id IS NULL"))
        if rows: self._add("belief.missing_successor", "beliefs", "ERROR", "Supersession successor is missing", rows, blocking=True)
        graph = {str(r[0]): str(r[1]) for r in conn.execute("SELECT id,superseded_by FROM beliefs WHERE superseded_by<>''")}
        cyclic = []
        for start in graph:
            seen=set(); node=start
            while node in graph:
                if node in seen: cyclic.append(start); break
                seen.add(node); node=graph[node]
        if cyclic: self._add("belief.supersession_cycle", "beliefs", "ERROR", "Cyclic supersession chains", cyclic, blocking=True)
        rows = list(conn.execute("""
            SELECT min(id) FROM beliefs WHERE epistemic_status IN ('KNOWN','INFERRED','SUSPECTED') AND superseded_by=''
            GROUP BY namespace,scope_kind,scope_id,subject_ref,predicate HAVING count(*)>1"""))
        if rows: self._add("belief.duplicate_current", "beliefs", "ERROR", "Duplicate active truth in exclusive identity", rows, blocking=True)
        rows = list(conn.execute("SELECT id FROM beliefs WHERE epistemic_status IN ('KNOWN','INFERRED','SUSPECTED') AND valid_until>0 AND valid_until<=?", (self.now,)))
        if rows: self._add("belief.expired_current", "beliefs", "ERROR", "Expired beliefs remain current", rows, blocking=True)
        rows = list(conn.execute("SELECT id FROM beliefs WHERE sensitivity IN ('private','sensitive','secret') AND scope_kind IN ('stream_public','public')"))
        if rows: self._add("belief.scope_privacy", "beliefs", "ERROR", "Sensitive beliefs use public scope", rows, blocking=True)
        if "memory_chunks" in tables:
            columns={str(r[1]) for r in conn.execute("PRAGMA table_info(memory_chunks)")}
            if "belief_id" in columns:
                rows=list(conn.execute("SELECT id FROM memory_chunks WHERE active=1 AND coalesce(belief_id,'')='' AND lower(kind) IN ('fact','belief','knowledge')"))
                if rows: self._add("belief.vector_truth", "beliefs", "ERROR", "Semantic vector rows lack a structured owner", rows, blocking=True)

    def _actions(self, conn, tables) -> None:
        if "viewer_promotion_profiles" in tables:
            rows=list(conn.execute("SELECT twitch_user_id FROM viewer_promotion_profiles WHERE active=1 AND auto_promo_mode<>'disabled' AND (owner_locked<>1 OR created_by NOT IN ('owner','owner_command'))"))
            if rows:self._add("action.promotion_authority", "actions", "ERROR", "Auto-promotion profile lacks explicit owner delegation", rows, blocking=True)
        if "action_ledger" not in tables: return
        allowed_receipts={"promotion_events"}
        bad=[]
        for row in conn.execute("SELECT id,source_store,source_record_id FROM action_ledger WHERE status='SUCCEEDED'"):
            store=str(row[1]); record=str(row[2])
            if store not in allowed_receipts or store not in tables:
                bad.append((row[0],)); continue
            found=conn.execute(f'SELECT execution_status FROM "{store}" WHERE CAST(id AS TEXT)=? LIMIT 1',(record,)).fetchone()
            if not found or str(found[0] or "").casefold() not in {"sent","succeeded","success"}:bad.append((row[0],))
        if bad: self._add("action.receipt_backing", "actions", "ERROR", "Successful ledger entries lack an authoritative receipt", bad, blocking=True)
        rows=list(conn.execute("SELECT min(id) FROM action_ledger WHERE status='SUCCEEDED' GROUP BY source_store,source_record_id HAVING count(*)>1"))
        if rows: self._add("action.duplicate_success", "actions", "ERROR", "Duplicate successful receipt projections", rows, blocking=True)

    def _continuity(self, conn, tables) -> None:
        if "conversations" in tables:
            rows=list(conn.execute("SELECT id FROM conversations WHERE status IN ('WAITING_ON_LEO','WAITING_ON_HEBE','ACTIVE') AND expires_at<=?",(self.now,)))
            if rows:self._add("continuity.expired_actionable", "continuity", "ERROR", "Expired conversations remain actionable", rows, blocking=True)
            rows=list(conn.execute("SELECT id FROM conversations WHERE status IN ('ARCHIVED','INTERRUPTED','EXPIRED','CLOSED') AND attention_state='FOREGROUND'"))
            if rows:self._add("continuity.closed_foreground", "continuity", "WARNING", "Closed conversations remain foreground", rows)
        if "open_threads" in tables:
            rows=list(conn.execute("SELECT id FROM open_threads WHERE status='OPEN' AND (valid_until<=? OR relevance_until<=?)",(self.now,self.now)))
            if rows:self._add("thread.expired_open", "continuity", "ERROR", "Expired OpenThreads remain active", rows, blocking=True)
            rows=list(conn.execute("SELECT id FROM open_threads WHERE status='RESOLVED' AND (resolved_at<=0 OR resolution_event_id='')"))
            if rows:self._add("thread.resolution_metadata", "continuity", "ERROR", "Resolved threads lack resolution metadata", rows, blocking=True)
            if "people" in tables:
                people={str(r[0]) for r in conn.execute("SELECT person_id FROM people")};missing=[]
                for row in conn.execute("SELECT id,participant_ids_json FROM open_threads WHERE scope_kind IN ('person','social','stream_public')"):
                    try:refs=json.loads(row[1] or "[]")
                    except Exception:refs=[];missing.append(row[0])
                    if any(str(ref) not in people and str(ref) not in {"leo","hebe"} for ref in refs):missing.append(row[0])
                if missing:self._add("thread.orphan_participant", "continuity", "ERROR", "OpenThreads reference missing people", missing, blocking=True)

    def _game(self, conn, tables) -> None:
        if "game_runs" in tables:
            rows=list(conn.execute("SELECT min(id) FROM game_runs WHERE status='ACTIVE' GROUP BY game_id,owner_id HAVING count(*)>1"))
            if rows:self._add("game.multiple_active_runs", "game", "ERROR", "Multiple canonical active runs", rows, blocking=True)
        if {"game_run_sessions","game_runs"}.issubset(tables):
            rows=list(conn.execute("SELECT s.id FROM game_run_sessions s LEFT JOIN game_runs r ON r.id=s.game_run_id WHERE r.id IS NULL"))
            if rows:self._add("game.orphan_run_session", "game", "ERROR", "Run-session links reference missing runs", rows, blocking=True)
        if {"game_knowledge_facts","beliefs"}.issubset(tables):
            rows=list(conn.execute("SELECT k.id FROM game_knowledge_facts k JOIN beliefs b ON b.id=k.belief_id WHERE b.namespace='game_run' OR b.scope_kind='game_run'"))
            if rows:self._add("game.run_fact_in_knowledge", "game", "ERROR", "GameRun facts leaked into GameKnowledge", rows, blocking=True)
            rows=list(conn.execute("SELECT k.id FROM game_knowledge_facts k LEFT JOIN beliefs b ON b.id=k.belief_id WHERE b.id IS NULL"))
            if rows:self._add("game.orphan_knowledge", "game", "ERROR", "GameKnowledge references missing beliefs", rows, blocking=True)

    def _social(self, conn, tables) -> None:
        if "person_identities" in tables:
            rows=list(conn.execute("SELECT min(id) FROM person_identities WHERE platform='twitch' AND platform_user_id<>'' GROUP BY platform_user_id HAVING count(distinct person_id)>1"))
            if rows:self._add("social.duplicate_stable_identity", "social", "ERROR", "Stable Twitch ID maps to multiple people", rows, blocking=True)
        if {"shared_culture_items","people"}.issubset(tables):
            people={str(r[0]) for r in conn.execute("SELECT person_id FROM people")}; missing=[]
            for row in conn.execute("SELECT id,participant_ids_json FROM shared_culture_items"):
                try: refs=json.loads(row[1] or "[]")
                except Exception: refs=[]; missing.append(row[0])
                if any(str(ref) not in people for ref in refs):missing.append(row[0])
            if missing:self._add("social.culture_participants", "social", "ERROR", "SharedCulture participants are missing", missing, blocking=True)
            rows=list(conn.execute("SELECT id FROM shared_culture_items WHERE status='RETIRED' AND cooldown_until>?",(self.now,)))
            if rows:self._add("social.retired_cooldown", "social", "WARNING", "Retired culture retains selection cooldown state", rows)
        if "social_episodes" in tables:
            rows=list(conn.execute("SELECT id FROM social_episodes WHERE sensitivity IN ('private','sensitive') AND retrieval_scope='stream_public'"))
            if rows:self._add("social.sensitive_public", "social", "ERROR", "Sensitive social episodes are public", rows, blocking=True)

    def _schedule(self, conn, tables) -> None:
        if "schedule_hypotheses" in tables:
            columns={str(r[1]) for r in conn.execute("PRAGMA table_info(schedule_hypotheses)")}
            status="status" if "status" in columns else "lifecycle_status" if "lifecycle_status" in columns else ""
            if status:
                identity=("weekday","time_window","content_key","stream_format") if {"weekday","time_window","content_key","stream_format"}.issubset(columns) else ("weekday","start_time","end_time") if {"weekday","start_time","end_time"}.issubset(columns) else ()
                rows=list(conn.execute(f"SELECT min(rowid) FROM schedule_hypotheses WHERE upper({status}) IN ('ACTIVE','CURRENT') GROUP BY {','.join(identity)} HAVING count(*)>1")) if identity else []
                if rows:self._add("schedule.duplicate_current", "schedule", "NEEDS_REVIEW", "Equivalent current schedule hypotheses need review", rows)

    def _identity(self, conn, tables) -> None:
        rows=[]
        if "beliefs" in tables:
            rows=list(conn.execute("SELECT id FROM beliefs WHERE namespace IN ('stable_core','identity') AND epistemic_status IN ('KNOWN','INFERRED','SUSPECTED')"))
        if rows:self._add("identity.stable_core_mutation", "identity", "ERROR", "Mutable beliefs attempt to own StableHebeCore", rows, blocking=True)

    def _consolidation(self, conn, tables) -> None:
        if "consolidation_deltas" in tables:
            rows=list(conn.execute("SELECT min(id) FROM consolidation_deltas GROUP BY idempotency_key HAVING count(*)>1"))
            if rows:self._add("consolidation.duplicate_delta", "consolidation", "ERROR", "Duplicate consolidation idempotency keys", rows, blocking=True)
            rows=list(conn.execute("SELECT id FROM consolidation_deltas WHERE validator_result<>'ACCEPTED' AND committed_object_ref<>''"))
            if rows:self._add("consolidation.rejected_committed", "consolidation", "ERROR", "Rejected deltas have committed semantic references", rows, blocking=True)
        if {"consolidation_deltas","consolidation_runs"}.issubset(tables):
            rows=list(conn.execute("SELECT d.id FROM consolidation_deltas d LEFT JOIN consolidation_runs r ON r.id=d.consolidation_run_id WHERE r.id IS NULL"))
            if rows:self._add("consolidation.orphan_delta", "consolidation", "ERROR", "Consolidation deltas reference missing runs", rows, blocking=True)

    @staticmethod
    def fingerprint(path: str | Path) -> str:
        digest=hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def markdown(report: dict) -> str:
        lines=["# Cognitive Integrity Report","",f"- Status: **{report['status']}**",f"- Blocking errors: {report['blocking_error_count']}",f"- Database SHA-256: `{report['db_sha256']}`","","## Findings",""]
        if not report["findings"]:lines.append("No invariant violations detected.")
        for item in report["findings"]:
            lines.append(f"- **{item['severity']}** `{item['check_id']}` — {item['message']} ({item['count']})")
        return "\n".join(lines)+"\n"
