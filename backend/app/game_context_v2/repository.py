from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.game_context_v2.models import GameIdentity, GameKnowledgeGap, GameRun, GameRunStatus


class GameV2Repository:
    def __init__(self, connection_factory: Callable) -> None:
        self.connection_factory=connection_factory
        self.lookup_latencies: dict[str,list[float]]={"run":[],"run_fact":[],"knowledge":[],"research_gap":[]}
        self.write_latencies: list[float]=[]
        raw=json.loads((Path(__file__).with_name("game_identities.json")).read_text(encoding="utf-8"))
        self.catalog=tuple(GameIdentity(game_id=row["game_id"],canonical_name=row["canonical_name"],aliases=tuple(row.get("aliases") or ()),platform_ids=dict(row.get("platform_ids") or {}),series=str(row.get("series") or "")) for row in raw["identities"])

    @staticmethod
    def normalize(value:str)->str:
        import re,unicodedata
        text=unicodedata.normalize("NFKD",str(value or "")).encode("ascii","ignore").decode().casefold()
        return "_".join(re.findall(r"[a-z0-9]+",text)) or "unknown_game"

    def resolve_identity(self,value:str)->GameIdentity:
        key=self.normalize(value)
        for item in self.catalog:
            if key in {self.normalize(item.game_id),self.normalize(item.canonical_name),*(self.normalize(alias) for alias in item.aliases)}:
                self.ensure_identity(item);return item
        clean=" ".join(str(value or "Unknown Game").split())
        item=GameIdentity(key,clean,(clean,));self.ensure_identity(item);return item

    def ensure_identity(self,item:GameIdentity)->None:
        started=time.perf_counter();conn=self.connection_factory()
        try:
            conn.execute("""INSERT INTO game_identities(game_id,canonical_name,aliases_json,platform_ids_json,series,schema_version)
                VALUES(?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET canonical_name=excluded.canonical_name,aliases_json=excluded.aliases_json""",
                (item.game_id,item.canonical_name,json.dumps(item.aliases,ensure_ascii=False),json.dumps(item.platform_ids,ensure_ascii=False),item.series,item.schema_version));conn.commit()
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)

    def create_run(self,run:GameRun)->GameRun:
        started=time.perf_counter();conn=self.connection_factory()
        try:
            conn.execute("""INSERT INTO game_runs(id,game_id,owner_id,run_kind,rules_json,status,started_at,last_active_at,ended_at,current_checkpoint_version,created_from_event_id,schema_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(run.id,run.game_id,run.owner_id,run.run_kind,json.dumps(run.rules,ensure_ascii=False),run.status.value,run.started_at,run.last_active_at,run.ended_at,run.current_checkpoint_version,run.created_from_event_id,run.schema_version));conn.commit();return run
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)

    def get_run(self,run_id:str)->GameRun|None:
        started=time.perf_counter();conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:row=conn.execute("SELECT * FROM game_runs WHERE id=?",(run_id,)).fetchone();return self._run(row) if row else None
        finally:conn.close();self.lookup_latencies["run"].append((time.perf_counter()-started)*1000)

    def list_runs(self,*,game_id:str="",owner_id:str="",statuses:tuple[str,...]=())->list[GameRun]:
        started=time.perf_counter();conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:
            where=["1=1"];args=[]
            if game_id:where.append("game_id=?");args.append(game_id)
            if owner_id:where.append("owner_id=?");args.append(owner_id)
            if statuses:where.append("status IN ("+",".join("?" for _ in statuses)+")");args.extend(statuses)
            return [self._run(row) for row in conn.execute("SELECT * FROM game_runs WHERE "+" AND ".join(where)+" ORDER BY last_active_at DESC,id",args)]
        finally:conn.close();self.lookup_latencies["run"].append((time.perf_counter()-started)*1000)

    def set_run_status(self,run_id:str,status:GameRunStatus,*,at:float,ended:bool=False)->GameRun:
        started=time.perf_counter();conn=self.connection_factory()
        try:
            conn.execute("UPDATE game_runs SET status=?,last_active_at=?,ended_at=CASE WHEN ? THEN ? ELSE ended_at END,current_checkpoint_version=current_checkpoint_version+1 WHERE id=?",(status.value,at,int(ended),at,run_id));conn.commit()
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)
        result=self.get_run(run_id)
        if result is None:raise KeyError(run_id)
        return result

    def link_session(self,run_id:str,stream_session_id:str,*,at:float,evidence_event_id:str)->dict[str,Any]:
        started=time.perf_counter();link_id=f"run_session_{uuid.uuid5(uuid.NAMESPACE_URL,run_id+'|'+stream_session_id).hex}"
        conn=self.connection_factory()
        try:
            cur=conn.execute("""INSERT OR IGNORE INTO game_run_sessions(id,game_run_id,stream_session_id,started_at,ended_at,evidence_event_id,source,schema_version)
                VALUES(?,?,?,?,0,?,'canonical_timeline',1)""",(link_id,run_id,str(stream_session_id),at,evidence_event_id));conn.commit()
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)
        return {"id":link_id,"game_run_id":run_id,"stream_session_id":str(stream_session_id),"started_at":at,"evidence_event_id":evidence_event_id,"created":cur.rowcount==1}

    def end_session(self,run_id:str,stream_session_id:str,*,at:float)->None:
        conn=self.connection_factory()
        try:conn.execute("UPDATE game_run_sessions SET ended_at=? WHERE game_run_id=? AND stream_session_id=?",(at,run_id,str(stream_session_id)));conn.commit()
        finally:conn.close()

    def session_links(self,run_id:str)->list[dict[str,Any]]:
        conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:return [dict(row) for row in conn.execute("SELECT * FROM game_run_sessions WHERE game_run_id=? ORDER BY started_at,id",(run_id,))]
        finally:conn.close()

    def add_event(self,*,run_id:str,event_type:str,subject_ref:str,predicate:str,object_value:Any,evidence_event_id:str,belief_id:str="",observed_at:float,epistemic_status:str)->dict[str,Any]:
        started=time.perf_counter();event_id=f"game_event_{uuid.uuid4().hex}";conn=self.connection_factory()
        try:
            conn.execute("""INSERT INTO game_run_events(id,game_run_id,event_type,subject_ref,predicate,object_json,evidence_event_id,belief_id,observed_at,epistemic_status,schema_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,1)""",(event_id,run_id,event_type,subject_ref,predicate,json.dumps(object_value,ensure_ascii=False),evidence_event_id,belief_id,observed_at,epistemic_status));conn.commit()
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)
        return {"id":event_id,"game_run_id":run_id,"event_type":event_type,"subject_ref":subject_ref,"predicate":predicate,"object":object_value,"evidence_event_id":evidence_event_id,"belief_id":belief_id,"observed_at":observed_at,"epistemic_status":epistemic_status}

    def events(self,run_id:str="")->list[dict[str,Any]]:
        conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:
            rows=conn.execute("SELECT * FROM game_run_events"+(" WHERE game_run_id=?" if run_id else "")+" ORDER BY observed_at,id",(run_id,) if run_id else ()).fetchall();result=[]
            for row in rows:item=dict(row);item["object"]=json.loads(item.pop("object_json"));result.append(item)
            return result
        finally:conn.close()

    def add_knowledge_link(self,*,fact_id:str,game_id:str,belief_id:str,source_type:str,source_quality:str,spoiler_class:str,dossier_link:str="",version_tag:str="",created_at:float)->None:
        started=time.perf_counter();conn=self.connection_factory()
        try:
            conn.execute("""INSERT OR IGNORE INTO game_knowledge_facts(id,game_id,belief_id,source_type,source_quality,spoiler_class,dossier_link,version_tag,created_at,schema_version)
                VALUES(?,?,?,?,?,?,?,?,?,1)""",(fact_id,game_id,belief_id,source_type,source_quality,spoiler_class,dossier_link,version_tag,created_at));conn.commit()
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)

    def knowledge(self,game_id:str)->list[dict[str,Any]]:
        started=time.perf_counter();conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:return [dict(row) for row in conn.execute("SELECT * FROM game_knowledge_facts WHERE game_id=? ORDER BY created_at DESC,id",(game_id,))]
        finally:conn.close();self.lookup_latencies["knowledge"].append((time.perf_counter()-started)*1000)

    def save_gap(self,gap:GameKnowledgeGap)->GameKnowledgeGap:
        started=time.perf_counter();conn=self.connection_factory()
        try:
            conn.execute("""INSERT INTO game_knowledge_v2_gaps(id,game_id,run_id,subject_ref,question_type,query_intent,spoiler_ceiling,required_confidence,created_from_event_id,normalized_gap_key,status,created_at,updated_at,resolved_fact_ids_json,schema_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(normalized_gap_key) DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at,resolved_fact_ids_json=excluded.resolved_fact_ids_json""",
                (gap.id,gap.game_id,gap.run_id,gap.subject_ref,gap.question_type,gap.query_intent,gap.spoiler_ceiling,gap.required_confidence,gap.created_from_event_id,gap.normalized_gap_key,gap.status,gap.created_at,gap.updated_at,json.dumps(gap.resolved_fact_ids),gap.schema_version));conn.commit();return gap
        finally:conn.close();self.write_latencies.append((time.perf_counter()-started)*1000)

    def gap(self,key:str)->GameKnowledgeGap|None:
        started=time.perf_counter();conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:row=conn.execute("SELECT * FROM game_knowledge_v2_gaps WHERE normalized_gap_key=?",(key,)).fetchone();return self._gap(row) if row else None
        finally:conn.close();self.lookup_latencies["research_gap"].append((time.perf_counter()-started)*1000)

    def gaps(self,game_id:str="")->list[dict[str,Any]]:
        conn=self.connection_factory();conn.row_factory=__import__('sqlite3').Row
        try:return [self._gap(row).to_dict() for row in conn.execute("SELECT * FROM game_knowledge_v2_gaps"+(" WHERE game_id=?" if game_id else "")+" ORDER BY created_at",(game_id,) if game_id else ())]
        finally:conn.close()

    def performance(self)->dict[str,Any]:
        return {**{key:self._pct(values) for key,values in self.lookup_latencies.items()},"db_write":self._pct(self.write_latencies)}

    @staticmethod
    def _pct(values:list[float])->dict[str,float|int]:
        v=sorted(values);return {"count":len(v),"p50_ms":round(statistics.median(v),6) if v else 0.0,"p95_ms":round(v[max(0,math.ceil(len(v)*.95)-1)],6) if v else 0.0}

    @staticmethod
    def _run(row)->GameRun:
        return GameRun(row["id"],row["game_id"],row["owner_id"],row["run_kind"],json.loads(row["rules_json"] or "{}"),GameRunStatus(row["status"]),float(row["started_at"]),float(row["last_active_at"]),float(row["ended_at"]),int(row["current_checkpoint_version"]),row["created_from_event_id"],int(row["schema_version"]))

    @staticmethod
    def _gap(row)->GameKnowledgeGap:
        return GameKnowledgeGap(row["id"],row["game_id"],row["run_id"],row["subject_ref"],row["question_type"],row["query_intent"],row["spoiler_ceiling"],float(row["required_confidence"]),row["created_from_event_id"],row["normalized_gap_key"],row["status"],float(row["created_at"]),float(row["updated_at"]),tuple(json.loads(row["resolved_fact_ids_json"] or "[]")),int(row["schema_version"]))
