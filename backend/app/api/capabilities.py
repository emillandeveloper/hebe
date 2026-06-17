from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.cognitive.capabilities import CapabilityRegistry
from app.cognitive.capabilities.models import Capability


router = APIRouter(prefix="/capabilities", tags=["capabilities"])


def capability_todo_payload(capability: Capability, reason: str | None = None) -> dict[str, Any]:
    backlog = capability.backlog
    payload = {
        "id": capability.id,
        "category": capability.category,
        "name": capability.name,
        "description": capability.description,
        "status": capability.status,
        "enabled": capability.enabled,
        "priority": backlog.priority,
        "effort": backlog.effort,
        "risk_level": capability.risk_level,
        "requires_confirmation": capability.requires_confirmation,
        "dependencies": capability.dependencies,
        "blocked_by": backlog.blocked_by,
        "next_actions": backlog.next_actions,
        "acceptance_criteria": backlog.acceptance_criteria,
        "implemented_by": capability.implemented_by,
        "recommended_next": backlog.recommended_next,
        "unblocked": backlog.unblocked,
        "todo_owner": backlog.todo_owner,
    }
    if reason:
        payload["reason"] = reason
    return payload


def capability_list_payload(
    *,
    status: str | None = None,
    category: str | None = None,
    executable: bool | None = None,
) -> dict[str, Any]:
    registry = CapabilityRegistry.default()
    capabilities = registry.list_all_capabilities()
    if status:
        capabilities = [capability for capability in capabilities if capability.status == status]
    if category:
        capabilities = [capability for capability in capabilities if capability.category == category]
    if executable is not None:
        capabilities = [capability for capability in capabilities if capability.executable is executable]
    return {
        "count": len(capabilities),
        "capabilities": [capability.to_dict() for capability in capabilities],
    }


def capability_summary_payload() -> dict[str, Any]:
    registry = CapabilityRegistry.default()
    summary = registry.backlog_summary()
    return {
        "counts": summary["counts"],
        "next_recommended_todo": summary["next_recommended_todo"],
    }


def capability_backlog_payload() -> dict[str, Any]:
    return CapabilityRegistry.default().backlog_summary()


def next_capability_payload() -> dict[str, Any]:
    result = CapabilityRegistry.default().next_recommended_todo_with_reason()
    capability = result.get("capability")
    reason = str(result.get("reason") or "")
    if capability is None:
        print("[HEBE][CAPABILITY_BACKLOG] next id=none reason=no_unblocked_items", flush=True)
        return {"next_recommended_todo": None, "reason": reason}
    print(f"[HEBE][CAPABILITY_BACKLOG] next id={capability.id} reason={reason}", flush=True)
    return {
        "next_recommended_todo": capability_todo_payload(capability, reason=reason),
        "reason": reason,
    }


def planned_capabilities_payload() -> dict[str, Any]:
    items = CapabilityRegistry.default().list_planned_not_implemented()
    return {
        "count": len(items),
        "items": [capability_todo_payload(capability) for capability in items],
    }


def partial_capabilities_payload() -> dict[str, Any]:
    items = CapabilityRegistry.default().list_partial_needing_completion()
    return {
        "count": len(items),
        "items": [capability_todo_payload(capability) for capability in items],
    }


def implemented_disabled_capabilities_payload() -> dict[str, Any]:
    items = CapabilityRegistry.default().list_implemented_disabled()
    return {
        "count": len(items),
        "items": [capability_todo_payload(capability) for capability in items],
    }


def capability_detail_payload(capability_id: str) -> dict[str, Any]:
    capability = CapabilityRegistry.default().get_capability(capability_id)
    if capability is None:
        raise HTTPException(status_code=404, detail="Capability not found")
    return capability.to_dict()


@router.get("")
def list_capabilities(
    status: str | None = Query(None),
    category: str | None = Query(None),
    executable: bool | None = Query(None),
):
    return capability_list_payload(status=status, category=category, executable=executable)


@router.get("/summary")
def get_capability_summary():
    return capability_summary_payload()


@router.get("/backlog")
def get_capability_backlog():
    return capability_backlog_payload()


@router.get("/backlog/next")
def get_next_capability_todo():
    return next_capability_payload()


@router.get("/backlog/planned")
def get_planned_capability_backlog():
    return planned_capabilities_payload()


@router.get("/backlog/partial")
def get_partial_capability_backlog():
    return partial_capabilities_payload()


@router.get("/backlog/implemented-disabled")
def get_implemented_disabled_capability_backlog():
    return implemented_disabled_capabilities_payload()


@router.get("/{capability_id}")
def get_capability(capability_id: str):
    return capability_detail_payload(capability_id)
