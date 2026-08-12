from __future__ import annotations
import json,math,sqlite3,statistics,time,uuid
from typing import Any,Callable
from app.social_world_v2.models import CultureStatus,Person,PersonIdentity,SocialEpisode

class SocialWorldRepository:
    def __init__(self,connection_factory:Callable[[],sqlite3.Connection]):self.connection_factory=connection_factory;self.latencies={"identity":[],"episode_write":[],"thread_lookup":[],"culture_select":[],"context":[]};self.writes=[]
    def _rows(self,sql,args=()):
        c=self.connection_factory();c.row_factory=sqlite3.Row
        try:return [dict(r) for r in c.execute(sql,args)]
        finally:c.close()
    def resolve_person(self,*,platform:str,platform_user_id:str,login:str,display_name:str,now:float,source:str)->tuple[Person,PersonIdentity,str]:
        start=time.perf_counter();platform=platform.casefold();login=str(login or "").strip();uid=str(platform_user_id or "").strip();c=self.connection_factory();c.row_factory=sqlite3.Row
        try:
            row=c.execute("SELECT * FROM person_identities WHERE platform=? AND platform_user_id=?",(platform,uid)).fetchone() if uid else None
            if row is None and not uid:
                matches=c.execute("SELECT * FROM person_identities WHERE platform=? AND lower(login)=lower(?)",(platform,login)).fetchall()
                row=matches[0] if len(matches)==1 else None
            decision="existing" if row else "new"
            if row:
                aliases=list(json.loads(row["aliases_json"] or "[]"));old=str(row["login"] or "")
                if old and old.casefold()!=login.casefold() and old not in aliases:aliases.append(old)
                if login and login not in aliases:aliases.append(login)
                c.execute("UPDATE person_identities SET login=?,display_name=?,aliases_json=?,last_seen_at=?,confidence=1.0,source=? WHERE id=?",(login,display_name or login,json.dumps(aliases,ensure_ascii=False),now,source,row["id"]));person_id=row["person_id"];identity_id=row["id"]
                c.execute("UPDATE people SET last_seen_at=? WHERE person_id=?",(now,person_id))
            else:
                person_id=f"person_{uuid.uuid4().hex}";identity_id=f"identity_{uuid.uuid4().hex}";aliases=[login] if login else []
                c.execute("INSERT INTO people(person_id,created_at,last_seen_at,scope,schema_version) VALUES(?,?,?,?,1)",(person_id,now,now,"stream_public"))
                c.execute("INSERT INTO person_identities(id,person_id,platform,platform_user_id,login,display_name,aliases_json,first_seen_at,last_seen_at,confidence,source,schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",(identity_id,person_id,platform,uid,login,display_name or login,json.dumps(aliases,ensure_ascii=False),now,now,1.0 if uid else .75,source))
            c.commit();pr=c.execute("SELECT * FROM people WHERE person_id=?",(person_id,)).fetchone();ir=c.execute("SELECT * FROM person_identities WHERE id=?",(identity_id,)).fetchone()
            return Person(pr["person_id"],pr["created_at"],pr["last_seen_at"],pr["scope"],pr["schema_version"]),self._identity(ir),decision
        finally:c.close();self.latencies["identity"].append((time.perf_counter()-start)*1000)
    def record_session(self,person_id,session_id,at):
        c=self.connection_factory()
        try:c.execute("INSERT OR IGNORE INTO person_sessions(person_id,stream_session_id,first_seen_at,last_seen_at) VALUES(?,?,?,?)",(person_id,session_id,at,at));c.execute("UPDATE person_sessions SET last_seen_at=? WHERE person_id=? AND stream_session_id=?",(at,person_id,session_id));c.commit()
        finally:c.close()
    def save_episode(self,e:SocialEpisode):
        start=time.perf_counter();c=self.connection_factory()
        try:c.execute("INSERT INTO social_episodes(id,episode_type,participant_ids_json,origin_event_id,related_event_ids_json,summary,tone_observations_json,created_at,relevance_until,retention_until,sensitivity,retention_class,retrieval_scope,salience_reason,schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(e.id,e.episode_type,json.dumps(e.participant_ids),e.origin_event_id,json.dumps(e.related_event_ids),e.summary,json.dumps(e.tone_observations),e.created_at,e.relevance_until,e.retention_until,e.sensitivity,e.retention_class,e.retrieval_scope,e.salience_reason,e.schema_version));c.commit();return e
        finally:c.close();v=(time.perf_counter()-start)*1000;self.latencies["episode_write"].append(v);self.writes.append(v)
    def episodes(self,person_id=""):
        rows=self._rows("SELECT * FROM social_episodes"+(" WHERE participant_ids_json LIKE ?" if person_id else "")+" ORDER BY created_at DESC",(f'%"{person_id}"%',) if person_id else ())
        for r in rows:
            for key in ("participant_ids","related_event_ids","tone_observations"):r[key]=json.loads(r.pop(key+"_json") or "[]")
        return rows
    def people(self):return self._rows("SELECT * FROM people ORDER BY created_at,person_id")
    def identities(self):
        rows=self._rows("SELECT * FROM person_identities ORDER BY first_seen_at,id")
        for r in rows:r["aliases"]=json.loads(r.pop("aliases_json") or "[]")
        return rows
    def person(self,person_id):
        rows=self._rows("SELECT * FROM people WHERE person_id=?",(person_id,));return rows[0] if rows else {}
    def familiarity(self,person_id):
        c=self.connection_factory()
        try:sessions=c.execute("SELECT COUNT(*) FROM person_sessions WHERE person_id=?",(person_id,)).fetchone()[0];episodes=c.execute("SELECT COUNT(*) FROM social_episodes WHERE participant_ids_json LIKE ?",(f'%"{person_id}"%',)).fetchone()[0]
        finally:c.close()
        score=min(1.0,.18*min(sessions,4)+.08*min(episodes,5));return {"distinct_sessions":sessions,"meaningful_episodes":episodes,"score":round(score,3),"band":"regular" if sessions>=3 else "familiar" if sessions>=2 else "new"}
    def save_culture(self,item):
        c=self.connection_factory()
        try:c.execute("""INSERT INTO shared_culture_items(id,label,meaning,origin_episode_id,participant_ids_json,scope,tone,status,confidence,created_at,last_reinforced_at,last_used_at,reuse_count,cooldown_until,schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1) ON CONFLICT(id) DO UPDATE SET status=excluded.status,confidence=excluded.confidence,last_reinforced_at=excluded.last_reinforced_at,last_used_at=excluded.last_used_at,reuse_count=excluded.reuse_count,cooldown_until=excluded.cooldown_until""",(item["id"],item["label"],item["meaning"],item["origin_episode_id"],json.dumps(item["participant_ids"]),item["scope"],item["tone"],item["status"],item["confidence"],item["created_at"],item["last_reinforced_at"],item["last_used_at"],item["reuse_count"],item["cooldown_until"]));c.commit();return item
        finally:c.close()
    def culture(self,item_id=""):
        rows=self._rows("SELECT * FROM shared_culture_items"+(" WHERE id=?" if item_id else "")+" ORDER BY created_at,id",(item_id,) if item_id else ())
        for r in rows:r["participant_ids"]=json.loads(r.pop("participant_ids_json") or "[]")
        return rows[0] if item_id and rows else ({} if item_id else rows)
    def add_culture_evidence(self,item_id,event_id,episode_id,reaction,polarity,weight,at,authority):
        c=self.connection_factory()
        try:c.execute("INSERT OR IGNORE INTO shared_culture_evidence(id,culture_item_id,event_id,episode_id,reaction,polarity,weight,observed_at,authority) VALUES(?,?,?,?,?,?,?,?,?)",(f"culture_evidence_{uuid.uuid4().hex}",item_id,event_id,episode_id,reaction,polarity,weight,at,authority));c.commit()
        finally:c.close()
    def culture_evidence(self,item_id=""):return self._rows("SELECT * FROM shared_culture_evidence"+(" WHERE culture_item_id=?" if item_id else "")+" ORDER BY observed_at,id",(item_id,) if item_id else ())
    def performance(self):return {k:self._pct(v) for k,v in self.latencies.items()}|{"db_write":self._pct(self.writes)}
    @staticmethod
    def _pct(v):
        x=sorted(v);return {"count":len(x),"p50_ms":round(statistics.median(x),6) if x else 0.0,"p95_ms":round(x[max(0,math.ceil(len(x)*.95)-1)],6) if x else 0.0}
    @staticmethod
    def _identity(r):return PersonIdentity(r["id"],r["person_id"],r["platform"],r["platform_user_id"],r["login"],r["display_name"],tuple(json.loads(r["aliases_json"] or "[]")),r["first_seen_at"],r["last_seen_at"],r["confidence"],r["source"],r["schema_version"])
