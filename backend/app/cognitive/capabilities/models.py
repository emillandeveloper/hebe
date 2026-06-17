from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CapabilityBacklog:
    priority: str = "P3"
    effort: str = "M"
    unblocked: bool = True
    blocked_by: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    recommended_next: bool = False
    todo_owner: str = "Hebe"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CapabilityBacklog":
        raw = data or {}
        return cls(
            priority=str(raw.get("priority") or "P3"),
            effort=str(raw.get("effort") or "M"),
            unblocked=bool(raw.get("unblocked", True)),
            blocked_by=[str(item) for item in raw.get("blocked_by") or []],
            next_actions=[str(item) for item in raw.get("next_actions") or []],
            acceptance_criteria=[str(item) for item in raw.get("acceptance_criteria") or []],
            recommended_next=bool(raw.get("recommended_next", False)),
            todo_owner=str(raw.get("todo_owner") or "Hebe"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Capability:
    id: str
    category: str
    name: str
    description: str
    status: str
    enabled: bool
    risk_level: str
    requires_confirmation: bool
    available_in_modes: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    implemented_by: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    notes: str = ""
    examples_semantic: list[str] = field(default_factory=list)
    output_policy: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    spoiler_policy: dict[str, Any] = field(default_factory=dict)
    goal_types: list[str] = field(default_factory=list)
    backlog: CapabilityBacklog = field(default_factory=CapabilityBacklog)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Capability":
        return cls(
            id=str(data["id"]),
            category=str(data.get("category") or "general"),
            name=str(data.get("name") or data["id"]),
            description=str(data.get("description") or ""),
            status=str(data.get("status") or "planned"),
            enabled=bool(data.get("enabled", False)),
            risk_level=str(data.get("risk_level") or "low"),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
            available_in_modes=[str(item) for item in data.get("available_in_modes") or []],
            input_schema=dict(data.get("input_schema") or {}),
            output_schema=dict(data.get("output_schema") or {}),
            dependencies=[str(item) for item in data.get("dependencies") or []],
            implemented_by=[str(item) for item in data.get("implemented_by") or []],
            tests=[str(item) for item in data.get("tests") or []],
            notes=str(data.get("notes") or ""),
            examples_semantic=[str(item) for item in data.get("examples_semantic") or []],
            output_policy=dict(data.get("output_policy") or {}),
            memory_policy=dict(data.get("memory_policy") or {}),
            spoiler_policy=dict(data.get("spoiler_policy") or {}),
            goal_types=[str(item) for item in data.get("goal_types") or []],
            backlog=CapabilityBacklog.from_dict(data.get("backlog")),
        )

    @property
    def executable(self) -> bool:
        return self.status == "implemented" and self.enabled

    @property
    def planned_only(self) -> bool:
        return self.status == "planned"

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["executable"] = self.executable
        return data
