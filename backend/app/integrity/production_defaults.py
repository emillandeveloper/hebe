from __future__ import annotations

import os


CANONICAL_FLAGS = (
    "HEBE_CONSOLIDATION_V2", "HEBE_CONSOLIDATION_COMMITS_V2", "HEBE_HEBE_SELF_V2",
    "HEBE_OWNER_PREFERENCES_V2", "HEBE_LEO_LANGUAGE_V2", "HEBE_TEMPORAL_RELEVANCE_V2",
    "HEBE_SCHEDULE_LEARNING_V2", "HEBE_SCENE_CONSEQUENCE_V2", "HEBE_HISTORICAL_ACTION_LEDGER_V2",
)


def enabled(name: str, *, environ: dict[str, str] | None = None) -> bool:
    env=os.environ if environ is None else environ
    master=str(env.get("HEBE_COGNITIVE_V2_ENABLED","true")).strip().casefold() in {"1","true","yes","on"}
    default="true" if master else "false"
    return str(env.get(name,default)).strip().casefold() in {"1","true","yes","on"}


def production_defaults(*, environ: dict[str, str] | None = None) -> dict[str, bool]:
    env=os.environ if environ is None else environ
    return {name:enabled(name,environ=env) for name in CANONICAL_FLAGS}
