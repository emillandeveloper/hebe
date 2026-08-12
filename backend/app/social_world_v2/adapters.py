from __future__ import annotations
from typing import Any

class LegacySocialCompatibilityAdapter:
    """Centralized read/shadow seam; legacy material is never promoted to truth implicitly."""
    def __init__(self):self.telemetry={"chatter_presence":[],"chatter_profiles":[],"chatter_facts":[],"stream_chatter_summaries":[],"viewer_profiles":[],"social_events":[],"promotion_profiles":[],"backfill":{"explicit_observation":0,"safe_episode":0,"inferred_compatibility_only":0,"ambiguous":0,"sensitive":0,"stale":0},"shadow_diffs":[]}
    def observe(self,source:str,value:Any,classification:str="INFERRED_COMPATIBILITY_ONLY"):
        item={"source":source,"classification":classification,"id":str(getattr(value,"id","") or "")};self.telemetry.setdefault(source,[]).append(item);self.telemetry["backfill"][classification.casefold()]+=1;return item
