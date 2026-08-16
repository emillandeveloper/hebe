from __future__ import annotations

import json
import math
import sqlite3
import statistics
import time
import uuid
from typing import Any, Callable

from app.social_world_v2.models import CultureStatus, Person, PersonIdentity, SocialEpisode


class SocialIdentityConflict(RuntimeError):
    def __init__(self, reason: str, candidate_person_ids: tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.candidate_person_ids = candidate_person_ids


class SocialWorldRepository:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory
        self.latencies = {
            "identity": [],
            "episode_write": [],
            "summary_write": [],
            "thread_lookup": [],
            "culture_select": [],
            "context": [],
        }
        self.writes: list[float] = []

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(sql, args)]
        finally:
            conn.close()

    def resolve_person(
        self,
        *,
        platform: str,
        platform_user_id: str,
        login: str,
        display_name: str,
        now: float,
        source: str,
    ) -> tuple[Person, PersonIdentity, str]:
        started = time.perf_counter()
        platform = str(platform or "").casefold()
        login = _login(login)
        stable_id = str(platform_user_id or "").strip()
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            row: sqlite3.Row | None = None
            if stable_id:
                row = conn.execute(
                    "SELECT * FROM person_identities WHERE platform=? AND platform_user_id=?",
                    (platform, stable_id),
                ).fetchone()
                decision = "existing_verified" if row else "new_verified"
            else:
                candidates = self._login_candidates(conn, platform, login)
                if len({str(candidate["person_id"]) for candidate in candidates}) > 1:
                    raise SocialIdentityConflict(
                        "ambiguous_login_without_stable_id",
                        tuple(sorted({str(candidate["person_id"]) for candidate in candidates})),
                    )
                row = candidates[0] if candidates else None
                decision = "existing_unverified" if row else "new_unverified"

            if row is not None:
                person_id = str(row["person_id"])
                identity_id = str(row["id"])
                aliases = list(dict.fromkeys(
                    _login(value) for value in _json_list(row["aliases_json"]) if _login(value)
                ))
                old_login = _login(row["login"])
                if old_login and old_login not in aliases:
                    aliases.append(old_login)
                if login and login not in aliases:
                    aliases.append(login)
                next_login = login or old_login
                next_display = str(display_name or row["display_name"] or next_login)
                next_confidence = 1.0 if str(row["platform_user_id"] or "") else float(row["confidence"] or 0.6)
                conn.execute(
                    """UPDATE person_identities SET login=?,display_name=?,aliases_json=?,
                       last_seen_at=?,confidence=?,source=? WHERE id=?""",
                    (
                        next_login,
                        next_display,
                        json.dumps(aliases, ensure_ascii=False),
                        now,
                        next_confidence,
                        source,
                        identity_id,
                    ),
                )
                conn.execute("UPDATE people SET last_seen_at=max(last_seen_at,?) WHERE person_id=?", (now, person_id))
            else:
                person_id = f"person_{uuid.uuid4().hex}"
                identity_id = f"identity_{uuid.uuid4().hex}"
                aliases = [login] if login else []
                conn.execute(
                    "INSERT INTO people(person_id,created_at,last_seen_at,scope,schema_version) VALUES(?,?,?,'stream_public',1)",
                    (person_id, now, now),
                )
                conn.execute(
                    """INSERT INTO person_identities(
                       id,person_id,platform,platform_user_id,login,display_name,aliases_json,
                       first_seen_at,last_seen_at,confidence,source,schema_version)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,1)""",
                    (
                        identity_id,
                        person_id,
                        platform,
                        stable_id,
                        login,
                        str(display_name or login),
                        json.dumps(aliases, ensure_ascii=False),
                        now,
                        now,
                        1.0 if stable_id else 0.6,
                        source,
                    ),
                )
            conn.commit()
            person_row = conn.execute("SELECT * FROM people WHERE person_id=?", (person_id,)).fetchone()
            identity_row = conn.execute("SELECT * FROM person_identities WHERE id=?", (identity_id,)).fetchone()
            return self._person(person_row), self._identity(identity_row), decision
        finally:
            conn.close()
            self.latencies["identity"].append((time.perf_counter() - started) * 1000)

    def find_identity(
        self,
        *,
        platform: str = "twitch",
        platform_user_id: str = "",
        login: str = "",
    ) -> PersonIdentity | None:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            stable_id = str(platform_user_id or "").strip()
            if stable_id:
                row = conn.execute(
                    "SELECT * FROM person_identities WHERE platform=? AND platform_user_id=?",
                    (str(platform or "").casefold(), stable_id),
                ).fetchone()
                return self._identity(row) if row else None
            candidates = self._login_candidates(conn, str(platform or "").casefold(), _login(login))
            person_ids = {str(row["person_id"]) for row in candidates}
            if len(person_ids) > 1:
                raise SocialIdentityConflict("ambiguous_login_without_stable_id", tuple(sorted(person_ids)))
            return self._identity(candidates[0]) if candidates else None
        finally:
            conn.close()

    @staticmethod
    def _login_candidates(conn: sqlite3.Connection, platform: str, login: str) -> list[sqlite3.Row]:
        if not login:
            return []
        result: list[sqlite3.Row] = []
        for row in conn.execute("SELECT * FROM person_identities WHERE platform=?", (platform,)).fetchall():
            aliases = {_login(value) for value in _json_list(row["aliases_json"])}
            if _login(row["login"]) == login or login in aliases:
                result.append(row)
        return result

    def record_session(self, person_id: str, session_id: str, at: float) -> None:
        conn = self.connection_factory()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO person_sessions(person_id,stream_session_id,first_seen_at,last_seen_at) VALUES(?,?,?,?)",
                (person_id, str(session_id), at, at),
            )
            conn.execute(
                "UPDATE person_sessions SET last_seen_at=max(last_seen_at,?) WHERE person_id=? AND stream_session_id=?",
                (at, person_id, str(session_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def record_presence(
        self,
        *,
        observation_id: str,
        person_id: str,
        stream_session_id: str,
        observed_at: float,
        source: str,
        message_seen: bool,
        direct_interaction: bool,
    ) -> bool:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO social_presence_events(
                   id,person_id,stream_session_id,observed_at,source,message_count,
                   direct_interaction_count,schema_version) VALUES(?,?,?,?,?,?,?,1)""",
                (
                    observation_id,
                    person_id,
                    str(stream_session_id or ""),
                    observed_at,
                    str(source or "observation"),
                    int(message_seen),
                    int(direct_interaction),
                ),
            )
            if cursor.rowcount != 1:
                conn.commit()
                return False
            session_id = str(stream_session_id or "")
            row = conn.execute(
                "SELECT * FROM person_sessions WHERE person_id=? AND stream_session_id=?",
                (person_id, session_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO person_sessions(
                       person_id,stream_session_id,first_seen_at,last_seen_at,first_message_at,
                       last_message_at,last_direct_interaction_at,message_count,
                       direct_interaction_count,presence_sources_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        person_id,
                        session_id,
                        observed_at,
                        observed_at,
                        observed_at if message_seen else 0.0,
                        observed_at if message_seen else 0.0,
                        observed_at if direct_interaction else 0.0,
                        int(message_seen),
                        int(direct_interaction),
                        json.dumps([str(source or "observation")]),
                    ),
                )
            else:
                sources = set(_json_list(row["presence_sources_json"]))
                sources.add(str(source or "observation"))
                conn.execute(
                    """UPDATE person_sessions SET last_seen_at=max(last_seen_at,?),
                       first_message_at=CASE WHEN ?>0 AND first_message_at=0 THEN ? ELSE first_message_at END,
                       last_message_at=CASE WHEN ?>0 THEN max(last_message_at,?) ELSE last_message_at END,
                       last_direct_interaction_at=CASE WHEN ?>0 THEN max(last_direct_interaction_at,?) ELSE last_direct_interaction_at END,
                       message_count=message_count+?,direct_interaction_count=direct_interaction_count+?,
                       presence_sources_json=? WHERE person_id=? AND stream_session_id=?""",
                    (
                        observed_at,
                        int(message_seen),
                        observed_at,
                        int(message_seen),
                        observed_at,
                        int(direct_interaction),
                        observed_at,
                        int(message_seen),
                        int(direct_interaction),
                        json.dumps(sorted(sources), ensure_ascii=False),
                        person_id,
                        session_id,
                    ),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def save_summary(
        self,
        *,
        summary_id: str,
        person_id: str,
        stream_session_id: str,
        source_type: str,
        source_record_id: str,
        summary_text: str,
        topics: list[str],
        message_count: int,
        direct_interaction_count: int,
        created_at: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        conn = self.connection_factory()
        try:
            conn.execute(
                """INSERT INTO social_summaries(
                   id,person_id,stream_session_id,source_type,source_record_id,summary_text,
                   topics_json,message_count,direct_interaction_count,created_at,schema_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,1)
                   ON CONFLICT(source_type,source_record_id) DO UPDATE SET
                   person_id=excluded.person_id,summary_text=excluded.summary_text,
                   topics_json=excluded.topics_json,message_count=excluded.message_count,
                   direct_interaction_count=excluded.direct_interaction_count,
                   created_at=excluded.created_at""",
                (
                    summary_id,
                    person_id,
                    str(stream_session_id),
                    source_type,
                    source_record_id,
                    str(summary_text or "")[:500],
                    json.dumps(list(dict.fromkeys(topics)), ensure_ascii=False),
                    max(0, int(message_count)),
                    max(0, int(direct_interaction_count)),
                    float(created_at),
                ),
            )
            conn.commit()
        finally:
            conn.close()
            latency = (time.perf_counter() - started) * 1000
            self.latencies["summary_write"].append(latency)
            self.writes.append(latency)
        return {
            "id": summary_id,
            "person_id": person_id,
            "stream_session_id": str(stream_session_id),
            "source_type": source_type,
            "source_record_id": source_record_id,
            "summary_text": str(summary_text or "")[:500],
            "topics": list(dict.fromkeys(topics)),
            "message_count": max(0, int(message_count)),
            "direct_interaction_count": max(0, int(direct_interaction_count)),
            "created_at": float(created_at),
        }

    def summaries(self, person_id: str = "") -> list[dict[str, Any]]:
        conn = self.connection_factory()
        try:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='social_summaries'"
            ).fetchone() is None:
                return []
        finally:
            conn.close()
        rows = self._rows(
            "SELECT * FROM social_summaries" + (" WHERE person_id=?" if person_id else "") + " ORDER BY created_at DESC,id",
            (person_id,) if person_id else (),
        )
        for row in rows:
            row["topics"] = _json_list(row.pop("topics_json"))
        return rows

    def latest_summary(self, person_id: str) -> dict[str, Any]:
        rows = self.summaries(person_id)
        return rows[0] if rows else {}

    def profile_stats(self, person_id: str) -> dict[str, Any]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS streams_seen_count,COALESCE(SUM(message_count),0) AS total_messages,
                   COALESCE(SUM(direct_interaction_count),0) AS total_direct_interactions,
                   COALESCE(MAX(last_seen_at),0) AS last_seen_at,
                   COALESCE(MAX(last_message_at),0) AS last_message_at,
                   COALESCE(MAX(last_direct_interaction_at),0) AS last_direct_interaction_at
                   FROM person_sessions WHERE person_id=?""",
                (person_id,),
            ).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()

    def save_episode(self, episode: SocialEpisode) -> SocialEpisode:
        started = time.perf_counter()
        conn = self.connection_factory()
        try:
            conn.execute(
                """INSERT INTO social_episodes(
                   id,episode_type,participant_ids_json,origin_event_id,related_event_ids_json,
                   summary,tone_observations_json,created_at,relevance_until,retention_until,
                   sensitivity,retention_class,retrieval_scope,salience_reason,schema_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode.id,
                    episode.episode_type,
                    json.dumps(episode.participant_ids),
                    episode.origin_event_id,
                    json.dumps(episode.related_event_ids),
                    episode.summary,
                    json.dumps(episode.tone_observations),
                    episode.created_at,
                    episode.relevance_until,
                    episode.retention_until,
                    episode.sensitivity,
                    episode.retention_class,
                    episode.retrieval_scope,
                    episode.salience_reason,
                    episode.schema_version,
                ),
            )
            conn.commit()
            return episode
        finally:
            conn.close()
            latency = (time.perf_counter() - started) * 1000
            self.latencies["episode_write"].append(latency)
            self.writes.append(latency)

    def episodes(self, person_id: str = "") -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM social_episodes" + (" WHERE participant_ids_json LIKE ?" if person_id else "") + " ORDER BY created_at DESC",
            (f'%"{person_id}"%',) if person_id else (),
        )
        for row in rows:
            for key in ("participant_ids", "related_event_ids", "tone_observations"):
                row[key] = json.loads(row.pop(key + "_json") or "[]")
        return rows

    def people(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM people ORDER BY created_at,person_id")

    def identities(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM person_identities ORDER BY first_seen_at,id")
        for row in rows:
            row["aliases"] = json.loads(row.pop("aliases_json") or "[]")
        return rows

    def person(self, person_id: str) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM people WHERE person_id=?", (person_id,))
        return rows[0] if rows else {}

    def familiarity(self, person_id: str) -> dict[str, Any]:
        conn = self.connection_factory()
        try:
            sessions = conn.execute("SELECT COUNT(*) FROM person_sessions WHERE person_id=?", (person_id,)).fetchone()[0]
            episodes = conn.execute(
                "SELECT COUNT(*) FROM social_episodes WHERE participant_ids_json LIKE ?",
                (f'%"{person_id}"%',),
            ).fetchone()[0]
        finally:
            conn.close()
        score = min(1.0, 0.18 * min(sessions, 4) + 0.08 * min(episodes, 5))
        return {
            "distinct_sessions": sessions,
            "meaningful_episodes": episodes,
            "score": round(score, 3),
            "band": "regular" if sessions >= 3 else "familiar" if sessions >= 2 else "new",
        }

    def save_culture(self, item: dict[str, Any]) -> dict[str, Any]:
        conn = self.connection_factory()
        try:
            conn.execute(
                """INSERT INTO shared_culture_items(
                   id,label,meaning,origin_episode_id,participant_ids_json,scope,tone,status,
                   confidence,created_at,last_reinforced_at,last_used_at,reuse_count,cooldown_until,schema_version)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status,confidence=excluded.confidence,
                   last_reinforced_at=excluded.last_reinforced_at,last_used_at=excluded.last_used_at,
                   reuse_count=excluded.reuse_count,cooldown_until=excluded.cooldown_until""",
                (
                    item["id"], item["label"], item["meaning"], item["origin_episode_id"],
                    json.dumps(item["participant_ids"]), item["scope"], item["tone"], item["status"],
                    item["confidence"], item["created_at"], item["last_reinforced_at"],
                    item["last_used_at"], item["reuse_count"], item["cooldown_until"],
                ),
            )
            conn.commit()
            return item
        finally:
            conn.close()

    def culture(self, item_id: str = "") -> dict[str, Any] | list[dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM shared_culture_items" + (" WHERE id=?" if item_id else "") + " ORDER BY created_at,id",
            (item_id,) if item_id else (),
        )
        for row in rows:
            row["participant_ids"] = json.loads(row.pop("participant_ids_json") or "[]")
        return rows[0] if item_id and rows else ({} if item_id else rows)

    def add_culture_evidence(
        self,
        item_id: str,
        event_id: str,
        episode_id: str,
        reaction: str,
        polarity: str,
        weight: float,
        at: float,
        authority: str,
    ) -> None:
        conn = self.connection_factory()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO shared_culture_evidence(
                   id,culture_item_id,event_id,episode_id,reaction,polarity,weight,observed_at,authority)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (f"culture_evidence_{uuid.uuid4().hex}", item_id, event_id, episode_id, reaction, polarity, weight, at, authority),
            )
            conn.commit()
        finally:
            conn.close()

    def culture_evidence(self, item_id: str = "") -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM shared_culture_evidence" + (" WHERE culture_item_id=?" if item_id else "") + " ORDER BY observed_at,id",
            (item_id,) if item_id else (),
        )

    def performance(self) -> dict[str, Any]:
        return {key: self._pct(values) for key, values in self.latencies.items()} | {"db_write": self._pct(self.writes)}

    @staticmethod
    def _pct(values: list[float]) -> dict[str, float | int]:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "p50_ms": round(statistics.median(ordered), 6) if ordered else 0.0,
            "p95_ms": round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 6) if ordered else 0.0,
        }

    @staticmethod
    def _person(row: sqlite3.Row) -> Person:
        return Person(row["person_id"], row["created_at"], row["last_seen_at"], row["scope"], row["schema_version"])

    @staticmethod
    def _identity(row: sqlite3.Row) -> PersonIdentity:
        return PersonIdentity(
            row["id"], row["person_id"], row["platform"], row["platform_user_id"],
            row["login"], row["display_name"], tuple(_json_list(row["aliases_json"])),
            row["first_seen_at"], row["last_seen_at"], row["confidence"], row["source"], row["schema_version"],
        )


def _login(value: Any) -> str:
    return str(value or "").strip().casefold().lstrip("@")


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


__all__ = ["SocialIdentityConflict", "SocialWorldRepository"]
