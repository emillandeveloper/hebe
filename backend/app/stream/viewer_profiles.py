from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
import sqlite3
import time
import unicodedata
from typing import Any, Callable

from app.services import db_sqlite


GENDERS = {"masculine", "feminine", "neutral", "unknown"}
SOURCE_RANK = {"heuristic": 1, "temporary_context": 2, "repeated_direct_evidence": 3, "owner_confirmed": 4, "self_declared": 5, "manual": 6}


def _norm(value: str) -> str:
    text = "".join(ch for ch in unicodedata.normalize("NFKD", str(value or "").casefold()) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", text).split())


@dataclass(slots=True)
class ViewerLinguisticProfile:
    twitch_user_id: str
    login: str
    display_name: str = ""
    preferred_grammatical_gender: str = "unknown"
    pronouns: dict[str, str] = field(default_factory=dict)
    preferred_address_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source_type: str = "heuristic"
    source_event_id: str = ""
    evidence_summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confirmed_at: float = 0.0
    expires_at: float = 0.0
    owner_locked: bool = False
    user_locked: bool = False
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ViewerLinguisticProfileStore:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection] | None = None) -> None:
        self.connection_factory = connection_factory or db_sqlite.get_db_connection
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS viewer_linguistic_profiles (
                twitch_user_id TEXT PRIMARY KEY, login TEXT NOT NULL, display_name TEXT,
                preferred_grammatical_gender TEXT NOT NULL DEFAULT 'unknown', pronouns_json TEXT,
                preferred_address_terms_json TEXT, confidence REAL NOT NULL DEFAULT 0,
                source_type TEXT NOT NULL DEFAULT 'heuristic', source_event_id TEXT, evidence_summary TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL, confirmed_at REAL NOT NULL DEFAULT 0,
                expires_at REAL NOT NULL DEFAULT 0, owner_locked INTEGER NOT NULL DEFAULT 0,
                user_locked INTEGER NOT NULL DEFAULT 0, conflict INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_viewer_linguistic_login ON viewer_linguistic_profiles(login)")
        conn.commit(); conn.close()

    def get(self, *, twitch_user_id: str = "", login: str = "") -> ViewerLinguisticProfile:
        conn = self._connect()
        row = None
        if twitch_user_id:
            row = conn.execute("SELECT * FROM viewer_linguistic_profiles WHERE twitch_user_id=?", (str(twitch_user_id),)).fetchone()
        if row is None and login:
            row = conn.execute("SELECT * FROM viewer_linguistic_profiles WHERE lower(login)=lower(?)", (str(login).lstrip("@"),)).fetchone()
        conn.close()
        if row is None:
            return ViewerLinguisticProfile(str(twitch_user_id or f"login:{_norm(login)}"), str(login or "").lstrip("@"))
        return self._from_row(row)

    def list_profiles(self) -> list[dict[str, Any]]:
        conn = self._connect(); rows = conn.execute("SELECT * FROM viewer_linguistic_profiles ORDER BY updated_at DESC").fetchall(); conn.close()
        return [self._from_row(row).to_dict() for row in rows]

    def save(self, profile: ViewerLinguisticProfile) -> ViewerLinguisticProfile:
        if profile.preferred_grammatical_gender not in GENDERS:
            raise ValueError("invalid grammatical gender")
        existing = self.get(twitch_user_id=profile.twitch_user_id, login=profile.login)
        if existing.login and existing.twitch_user_id == profile.twitch_user_id:
            profile.created_at = existing.created_at
        profile.updated_at = time.time()
        conn = self._connect()
        conn.execute("""
            INSERT INTO viewer_linguistic_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(twitch_user_id) DO UPDATE SET login=excluded.login, display_name=excluded.display_name,
            preferred_grammatical_gender=excluded.preferred_grammatical_gender, pronouns_json=excluded.pronouns_json,
            preferred_address_terms_json=excluded.preferred_address_terms_json, confidence=excluded.confidence,
            source_type=excluded.source_type, source_event_id=excluded.source_event_id, evidence_summary=excluded.evidence_summary,
            updated_at=excluded.updated_at, confirmed_at=excluded.confirmed_at, expires_at=excluded.expires_at,
            owner_locked=excluded.owner_locked, user_locked=excluded.user_locked, conflict=excluded.conflict
        """, (
            profile.twitch_user_id, profile.login, profile.display_name, profile.preferred_grammatical_gender,
            json.dumps(profile.pronouns, ensure_ascii=False), json.dumps(profile.preferred_address_terms, ensure_ascii=False),
            profile.confidence, profile.source_type, profile.source_event_id, profile.evidence_summary,
            profile.created_at, profile.updated_at, profile.confirmed_at, profile.expires_at,
            int(profile.owner_locked), int(profile.user_locked), int(profile.conflict),
        ))
        conn.commit(); conn.close()
        return profile

    def apply_evidence(self, *, twitch_user_id: str, login: str, display_name: str = "", candidate_gender: str,
                       confidence: float, source_type: str, source_event_id: str = "", evidence_summary: str = "") -> tuple[ViewerLinguisticProfile, str]:
        if candidate_gender not in GENDERS:
            raise ValueError("invalid candidate gender")
        current = self.get(twitch_user_id=twitch_user_id, login=login)
        print(f"[HEBE][VIEWER_IDENTITY_EVIDENCE] viewer={login} candidate={candidate_gender} confidence={confidence:.2f} source={source_type}", flush=True)
        confirmed = current.owner_locked or current.user_locked or current.source_type in {"owner_confirmed", "self_declared", "manual"}
        incoming_rank = SOURCE_RANK.get(source_type, 0); current_rank = SOURCE_RANK.get(current.source_type, 0)
        if confirmed and candidate_gender != current.preferred_grammatical_gender and incoming_rank < current_rank:
            current.conflict = True; current.updated_at = time.time(); self.save(current)
            print(f"[HEBE][VIEWER_IDENTITY_PROFILE] action=conflict reason=confirmed_profile_preserved", flush=True)
            return current, "conflict"
        if source_type in {"heuristic", "temporary_context"} and confidence < .8:
            print("[HEBE][VIEWER_IDENTITY_PROFILE] action=ignore reason=insufficient_evidence", flush=True)
            return current, "ignore"
        current.twitch_user_id = str(twitch_user_id or current.twitch_user_id)
        current.login = str(login or current.login).lstrip("@"); current.display_name = display_name or current.display_name
        current.preferred_grammatical_gender = candidate_gender; current.confidence = float(confidence)
        current.source_type = source_type; current.source_event_id = source_event_id; current.evidence_summary = evidence_summary
        current.conflict = False
        if source_type == "owner_confirmed": current.owner_locked = True; current.confirmed_at = time.time()
        if source_type == "self_declared": current.user_locked = True; current.confirmed_at = time.time()
        if source_type in {"heuristic", "temporary_context"}: current.expires_at = time.time() + 86400
        self.save(current)
        action = "create" if not confirmed and current.created_at == current.updated_at else "update"
        print(f"[HEBE][VIEWER_IDENTITY_PROFILE] action={action} reason=accepted_evidence", flush=True)
        return current, action

    def clear(self, *, twitch_user_id: str = "", login: str = "") -> bool:
        conn = self._connect()
        if twitch_user_id: cur = conn.execute("DELETE FROM viewer_linguistic_profiles WHERE twitch_user_id=?", (twitch_user_id,))
        else: cur = conn.execute("DELETE FROM viewer_linguistic_profiles WHERE lower(login)=lower(?)", (login.lstrip("@"),))
        conn.commit(); changed = cur.rowcount > 0; conn.close(); return changed

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ViewerLinguisticProfile:
        return ViewerLinguisticProfile(
            twitch_user_id=row["twitch_user_id"], login=row["login"], display_name=row["display_name"] or "",
            preferred_grammatical_gender=row["preferred_grammatical_gender"], pronouns=json.loads(row["pronouns_json"] or "{}"),
            preferred_address_terms=json.loads(row["preferred_address_terms_json"] or "[]"), confidence=float(row["confidence"] or 0),
            source_type=row["source_type"], source_event_id=row["source_event_id"] or "", evidence_summary=row["evidence_summary"] or "",
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]), confirmed_at=float(row["confirmed_at"] or 0),
            expires_at=float(row["expires_at"] or 0), owner_locked=bool(row["owner_locked"]), user_locked=bool(row["user_locked"]), conflict=bool(row["conflict"]),
        )


class GrammaticalAgreementGuard:
    PAIRS = {
        "tranquilo": "tranquila", "listo": "lista", "guapo": "guapa", "campeón": "campeona",
        "campeon": "campeona", "novio": "novia", "tonto": "tonta", "estúpido": "estúpida", "estupido": "estupida",
    }

    def evaluate(self, text: str, *, viewer: str, profile: ViewerLinguisticProfile, refers_to_hebe: bool = False) -> dict[str, Any]:
        original = str(text or "")
        if refers_to_hebe:
            return self._result(original, viewer, profile, [], "allow")
        gender = "neutral" if profile.conflict else profile.preferred_grammatical_gender
        repaired = original; violations: list[str] = []
        for masculine, feminine in self.PAIRS.items():
            if gender == "feminine" and re.search(rf"\b{re.escape(masculine)}\b", repaired, re.I):
                repaired = re.sub(rf"\b{re.escape(masculine)}\b", feminine, repaired, flags=re.I); violations.append(f"{masculine}_to_feminine")
            elif gender == "masculine" and re.search(rf"\b{re.escape(feminine)}\b", repaired, re.I):
                repaired = re.sub(rf"\b{re.escape(feminine)}\b", masculine, repaired, flags=re.I); violations.append(f"{feminine}_to_masculine")
            elif gender in {"unknown", "neutral"}:
                pattern = rf"\b(?:{re.escape(masculine)}|{re.escape(feminine)})\b"
                if re.search(pattern, repaired, re.I):
                    repaired = re.sub(pattern, "", repaired, flags=re.I); violations.append("gendered_form_neutralized")
        repaired = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", repaired)).strip()
        action = "allow" if not violations else "neutralize" if gender in {"neutral", "unknown"} else "repair"
        return self._result(repaired, viewer, profile, violations, action)

    @staticmethod
    def _result(text: str, viewer: str, profile: ViewerLinguisticProfile, violations: list[str], action: str) -> dict[str, Any]:
        print(f"[HEBE][GRAMMATICAL_AGREEMENT_GUARD] viewer={viewer} profile={profile.preferred_grammatical_gender} violations={violations!r} action={action}", flush=True)
        return {"passed": True, "text": text, "viewer": viewer, "profile": profile.preferred_grammatical_gender, "violations": violations, "action": action}


@dataclass(slots=True)
class ViewerProfileCommand:
    detected: bool
    viewer_text: str = ""
    gender: str = ""
    action: str = ""
    reason: str = ""


class ViewerProfileCommandParser:
    def parse(self, text: str) -> ViewerProfileCommand:
        normalized = _norm(text)
        clear = re.search(r"(?:olvida|borra|limpia)\s+(?:los\s+)?(?:pronombres|perfil|genero|forma)\s+(?:de\s+)?([a-z0-9_]+)", normalized)
        if clear: return ViewerProfileCommand(True, clear.group(1), action="clear", reason="explicit_clear")
        patterns = (
            ("feminine", r"(?:recuerda\s+que\s+)?([a-z0-9_]+)\s+(?:es\s+una\s+mujer|es\s+mujer)|(?:trata|habla|dirigete)\s+a\s+([a-z0-9_]+)\s+en\s+femenino"),
            ("masculine", r"(?:recuerda\s+que\s+)?([a-z0-9_]+)\s+(?:es\s+un\s+hombre|es\s+hombre)|(?:trata|habla|dirigete)\s+a\s+([a-z0-9_]+)\s+en\s+masculino"),
            ("neutral", r"(?:usa|utiliza)\s+(?:lenguaje\s+)?neutral\s+(?:con|para)\s+([a-z0-9_]+)"),
        )
        for gender, pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                viewer = next(group for group in match.groups() if group)
                return ViewerProfileCommand(True, viewer, gender, "set", "explicit_owner_preference")
        return ViewerProfileCommand(False, reason="not_profile_command")
