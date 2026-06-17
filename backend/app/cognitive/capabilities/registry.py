from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.cognitive.capabilities.models import Capability


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
_STATUS_TODO_RANK = {"partial": 0, "planned": 1, "implemented": 2, "deprecated": 3}
_EFFORT_RANK = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4}


class CapabilityRegistry:
    def __init__(self, capabilities: list[Capability]):
        self._capabilities = capabilities
        self._by_id = {capability.id: capability for capability in capabilities}

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "CapabilityRegistry":
        catalog_path = Path(path) if path else Path(__file__).with_name("capability_catalog.yaml")
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        items = raw.get("capabilities") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError("capability catalog must contain a capabilities list")
        return cls([Capability.from_dict(item) for item in items])

    @classmethod
    def default(cls) -> "CapabilityRegistry":
        return _default_registry()

    def list_all_capabilities(self) -> list[Capability]:
        return list(self._capabilities)

    def list_enabled_capabilities(self) -> list[Capability]:
        return [capability for capability in self._capabilities if capability.enabled]

    def list_executable_capabilities(self) -> list[Capability]:
        return [capability for capability in self._capabilities if capability.executable]

    def get_capability(self, capability_id: str) -> Capability | None:
        return self._by_id.get(capability_id)

    def find_capabilities_by_category(self, category: str) -> list[Capability]:
        return [
            capability
            for capability in self._capabilities
            if capability.category == category
        ]

    def find_capabilities_for_goal(self, goal_type: str) -> list[Capability]:
        return [
            capability
            for capability in self._capabilities
            if goal_type in capability.goal_types
        ]

    def validate_capability_inputs(self, capability_id: str, inputs: dict[str, Any] | None) -> dict[str, Any]:
        capability = self._require_capability(capability_id)
        input_schema = capability.input_schema or {}
        required = input_schema.get("required") or []
        missing = [
            name
            for name in required
            if name not in (inputs or {}) or (inputs or {}).get(name) in (None, "")
        ]
        return {
            "ok": not missing,
            "capability_id": capability_id,
            "missing": missing,
        }

    def check_capability_available(self, capability_id: str, current_mode: str | None = None) -> dict[str, Any]:
        capability = self._require_capability(capability_id)
        mode = current_mode or "unknown"
        mode_allowed = not capability.available_in_modes or mode in capability.available_in_modes or "any" in capability.available_in_modes
        reasons: list[str] = []
        if capability.status != "implemented":
            reasons.append(f"status={capability.status}")
        if not capability.enabled:
            reasons.append("disabled")
        if not mode_allowed:
            reasons.append(f"mode={mode}")
        return {
            "available": capability.executable and mode_allowed,
            "capability_id": capability_id,
            "status": capability.status,
            "enabled": capability.enabled,
            "current_mode": mode,
            "reasons": reasons,
        }

    def check_capability_risk(self, capability_id: str) -> dict[str, Any]:
        capability = self._require_capability(capability_id)
        return {
            "capability_id": capability_id,
            "risk_level": capability.risk_level,
            "requires_confirmation": capability.requires_confirmation,
        }

    def list_planned_not_implemented(self) -> list[Capability]:
        return self._sort_todo_items([
            capability
            for capability in self._capabilities
            if capability.status == "planned"
        ])

    def list_high_priority_unblocked(self) -> list[Capability]:
        return self._sort_todo_items([
            capability
            for capability in self._capabilities
            if capability.status in {"planned", "partial"}
            and capability.backlog.priority in {"P0", "P1"}
            and capability.backlog.unblocked
            and not capability.backlog.blocked_by
        ])

    def next_recommended_todo(self) -> Capability | None:
        result = self.next_recommended_todo_with_reason()
        capability = result.get("capability")
        return capability if isinstance(capability, Capability) else None

    def next_recommended_todo_with_reason(self) -> dict[str, Any]:
        candidates = [
            capability
            for capability in self._capabilities
            if capability.status in {"planned", "partial"}
            and capability.backlog.unblocked
            and not capability.backlog.blocked_by
        ]
        if not candidates:
            return {"capability": None, "reason": "no_unblocked_partial_or_planned_capabilities"}
        recommended = [capability for capability in candidates if capability.backlog.recommended_next]
        selected = self._sort_todo_items(recommended or candidates)[0]
        if selected.backlog.recommended_next:
            reason = "backlog.recommended_next=true"
        else:
            reason = (
                "highest_priority_unblocked "
                f"priority={selected.backlog.priority} "
                f"status={selected.status} "
                f"effort={selected.backlog.effort}"
            )
        return {"capability": selected, "reason": reason}

    def list_implemented_disabled(self) -> list[Capability]:
        return self._sort_todo_items([
            capability
            for capability in self._capabilities
            if capability.status == "implemented" and not capability.enabled
        ])

    def list_partial_needing_completion(self) -> list[Capability]:
        return self._sort_todo_items([
            capability
            for capability in self._capabilities
            if capability.status == "partial"
        ])

    def backlog_summary(self) -> dict[str, Any]:
        next_todo = self.next_recommended_todo()
        planned = self.list_planned_not_implemented()
        high_priority = self.list_high_priority_unblocked()
        disabled = self.list_implemented_disabled()
        partial = self.list_partial_needing_completion()
        return {
            "counts": {
                "all": len(self._capabilities),
                "implemented": len([c for c in self._capabilities if c.status == "implemented"]),
                "partial": len([c for c in self._capabilities if c.status == "partial"]),
                "planned": len([c for c in self._capabilities if c.status == "planned"]),
                "enabled": len(self.list_enabled_capabilities()),
                "disabled": len([c for c in self._capabilities if not c.enabled]),
                "executable": len(self.list_executable_capabilities()),
                "planned_not_implemented": len(planned),
                "high_priority_unblocked": len(high_priority),
                "implemented_disabled": len(disabled),
                "partial_needing_completion": len(partial),
            },
            "planned_not_implemented": [capability.to_dict() for capability in planned],
            "high_priority_unblocked": [capability.to_dict() for capability in high_priority],
            "next_recommended_todo": next_todo.to_dict() if next_todo else None,
            "implemented_disabled": [capability.to_dict() for capability in disabled],
            "partial_needing_completion": [capability.to_dict() for capability in partial],
        }

    def answer_backlog_query(self, query_type: str) -> dict[str, Any]:
        if query_type == "planned_not_implemented":
            items = self.list_planned_not_implemented()
            return self._query_response(query_type, items)
        if query_type == "high_priority_unblocked":
            items = self.list_high_priority_unblocked()
            return self._query_response(query_type, items)
        if query_type == "next_todo":
            item = self.next_recommended_todo()
            return {
                "query_type": query_type,
                "items": [item.to_dict()] if item else [],
                "next_recommended_todo": item.to_dict() if item else None,
            }
        if query_type == "implemented_disabled":
            items = self.list_implemented_disabled()
            return self._query_response(query_type, items)
        if query_type == "partial_needs_completion":
            items = self.list_partial_needing_completion()
            return self._query_response(query_type, items)
        return {
            "query_type": query_type,
            "summary": self.backlog_summary(),
            "items": [],
        }

    def _query_response(self, query_type: str, items: list[Capability]) -> dict[str, Any]:
        return {
            "query_type": query_type,
            "items": [capability.to_dict() for capability in items],
            "count": len(items),
        }

    def _require_capability(self, capability_id: str) -> Capability:
        capability = self.get_capability(capability_id)
        if capability is None:
            raise KeyError(f"unknown capability: {capability_id}")
        return capability

    def _sort_todo_items(self, capabilities: list[Capability]) -> list[Capability]:
        return sorted(
            capabilities,
            key=lambda capability: (
                0 if capability.backlog.recommended_next else 1,
                _PRIORITY_RANK.get(capability.backlog.priority, 99),
                _STATUS_TODO_RANK.get(capability.status, 99),
                _EFFORT_RANK.get(capability.backlog.effort, 99),
                capability.category,
                capability.id,
            ),
        )


@lru_cache(maxsize=1)
def _default_registry() -> CapabilityRegistry:
    return CapabilityRegistry.from_yaml()
