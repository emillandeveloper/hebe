from __future__ import annotations
from typing import Any


class LegacyGameCompatibilityAdapter:
    """Single v2↔legacy projection seam; no legacy object is promoted to truth here."""
    def __init__(self)->None:
        self.telemetry={"legacy_progress":[],"dossier":[],"legacy_run":[],"shadow_diffs":[],"backfill":{"validated":0,"compatibility_only":0,"ambiguous":0,"stale":0}}

    def observe_progress(self,progress:Any,*,run_id:str)->dict:
        item={"run_id":run_id,"classification":"COMPATIBILITY_ONLY","game_id":str(getattr(progress,"game_id","") or ""),"stream_session_id":str(getattr(progress,"stream_session_id","") or "")};self.telemetry["legacy_progress"].append(item);self.telemetry["backfill"]["compatibility_only"]+=1;return item

    def observe_dossier(self,dossier:Any)->dict:
        item={"game_id":str(getattr(dossier,"game_id","") or ""),"classification":"COMPATIBILITY_ONLY","dossier_version":int(getattr(dossier,"dossier_version",0) or 0)};self.telemetry["dossier"].append(item);self.telemetry["backfill"]["compatibility_only"]+=1;return item
