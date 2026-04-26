from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/debug", tags=["debug"])

@router.post("/push-event")
async def push_event(request: Request, body: dict):
    """
    Inyecta un InternalEvent manual en el scheduler.
    Body: {"event_type": "twitch_sub", "payload": {...}}
    """
    event_type = body.get("event_type")
    payload = body.get("payload") or {}

    if not event_type:
        raise HTTPException(400, "missing event_type")

    # Acceso al engine — adapta a tu DI real (probablemente request.app.state.adapter)
    adapter = request.app.state.adapter
    engine = getattr(adapter, "_engine", None)

    if engine is None or not adapter.running:
        raise HTTPException(503, "engine not running")

    event = engine.scheduler.push_event(event_type, payload)
    return {"ok": True, "event_type": event.event_type, "created_at": event.created_at}