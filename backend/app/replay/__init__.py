"""Deterministic cognitive replay and verification infrastructure.

Exports are lazy so production modules may import the migration runner without
loading the replay runner (which itself constructs :class:`HebeEngine`).
"""

from typing import Any

__all__ = ["CognitiveReplayRunner", "CognitiveReplayScenario", "ScenarioRunResult"]


def __getattr__(name: str) -> Any:
    if name == "CognitiveReplayScenario":
        from app.replay.scenario import CognitiveReplayScenario
        return CognitiveReplayScenario
    if name in {"CognitiveReplayRunner", "ScenarioRunResult"}:
        from app.replay.cognitive import CognitiveReplayRunner, ScenarioRunResult
        return {"CognitiveReplayRunner": CognitiveReplayRunner, "ScenarioRunResult": ScenarioRunResult}[name]
    raise AttributeError(name)
