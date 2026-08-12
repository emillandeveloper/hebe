from __future__ import annotations
from dataclasses import asdict,dataclass
from enum import StrEnum
from typing import Any

class CultureStatus(StrEnum):
    CANDIDATE="CANDIDATE";ACTIVE="ACTIVE";WEAKENING="WEAKENING";RETIRED="RETIRED";REJECTED="REJECTED"

@dataclass(frozen=True,slots=True)
class Person:
    person_id:str;created_at:float;last_seen_at:float;scope:str="stream_public";schema_version:int=1
    def to_dict(self):return asdict(self)

@dataclass(frozen=True,slots=True)
class PersonIdentity:
    id:str;person_id:str;platform:str;platform_user_id:str;login:str;display_name:str;aliases:tuple[str,...];first_seen_at:float;last_seen_at:float;confidence:float;source:str;schema_version:int=1
    def to_dict(self):v=asdict(self);v["aliases"]=list(self.aliases);return v

@dataclass(frozen=True,slots=True)
class SocialEpisode:
    id:str;episode_type:str;participant_ids:tuple[str,...];origin_event_id:str;related_event_ids:tuple[str,...];summary:str;tone_observations:tuple[str,...];created_at:float;relevance_until:float;retention_until:float;sensitivity:str;retention_class:str;retrieval_scope:str;salience_reason:str;schema_version:int=1
    def to_dict(self):v=asdict(self);v["participant_ids"]=list(self.participant_ids);v["related_event_ids"]=list(self.related_event_ids);v["tone_observations"]=list(self.tone_observations);return v

@dataclass(frozen=True,slots=True)
class SocialOpportunity:
    id:str;category:str;person_id:str;evidence_ids:tuple[str,...];relevance:float;confidence:float;urgency:str;sensitivity:str;suggested_speech_act:str;scene_suitable:bool;emission_allowed:bool=False
    def to_dict(self):v=asdict(self);v["evidence_ids"]=list(self.evidence_ids);return v

@dataclass(frozen=True,slots=True)
class SocialContext:
    person:dict[str,Any];familiarity:dict[str,Any];recent_episodes:tuple[dict[str,Any],...];active_threads:tuple[dict[str,Any],...];relevant_hypotheses:tuple[dict[str,Any],...];shared_culture_candidates:tuple[dict[str,Any],...];domain_authority_refs:tuple[dict[str,Any],...];selected:tuple[dict[str,Any],...];rejected:tuple[dict[str,Any],...];reasons:dict[str,int];provenance_manifest:tuple[dict[str,Any],...];manifest_size_bytes:int;latency_ms:float
    def to_dict(self):return asdict(self)
