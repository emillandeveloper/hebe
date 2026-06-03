from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services import db_sqlite
from app.services.stt_whisper import list_audio_devices, test_audio_input_device


router = APIRouter(prefix="/audio", tags=["audio"])

SETTING_DEVICE_ID = "stt.input_device_id"
SETTING_DEVICE_NAME = "stt.input_device_name"
SETTING_DEVICE_HOST_API = "stt.input_device_host_api"
SETTING_DEVICE_SAMPLE_RATE = "stt.input_device_sample_rate"
SETTING_DEVICE_CHANNELS = "stt.input_device_channels"
SETTING_DEVICE_SIGNATURE = "stt.input_device_signature"


class InputDeviceSelection(BaseModel):
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    host_api: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    signature: Optional[str] = None


class InputDeviceTestRequest(BaseModel):
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    host_api: Optional[str] = None
    seconds: float = 4.0


def _persisted_selection() -> dict:
    device_id = db_sqlite.get_setting(SETTING_DEVICE_ID, os.getenv("HEBE_STT_INPUT_DEVICE", "") or "")
    device_name = db_sqlite.get_setting(SETTING_DEVICE_NAME, os.getenv("HEBE_STT_INPUT_DEVICE_NAME", "") or "")
    host_api = db_sqlite.get_setting(SETTING_DEVICE_HOST_API, os.getenv("HEBE_STT_INPUT_DEVICE_HOST_API", "") or "")
    sample_rate = db_sqlite.get_setting(SETTING_DEVICE_SAMPLE_RATE, os.getenv("HEBE_STT_INPUT_DEVICE_SAMPLE_RATE", "") or "")
    channels = db_sqlite.get_setting(SETTING_DEVICE_CHANNELS, os.getenv("HEBE_STT_INPUT_DEVICE_CHANNELS", "") or "")
    signature = db_sqlite.get_setting(SETTING_DEVICE_SIGNATURE, os.getenv("HEBE_STT_INPUT_DEVICE_SIGNATURE", "") or "")
    return {
        "device_id": device_id or "",
        "device_name": device_name or "",
        "host_api": host_api or "",
        "sample_rate": int(sample_rate) if str(sample_rate or "").isdigit() else None,
        "channels": int(channels) if str(channels or "").isdigit() else None,
        "signature": signature or "",
    }


@router.get("/input-devices")
def audio_input_devices():
    try:
        devices = list_audio_devices()
        return {"devices": devices}
    except Exception as exc:
        print(f"[HEBE][STT][ERROR] list input devices failed: {exc!r}", flush=True)
        raise HTTPException(status_code=500, detail=f"Audio device list failed: {type(exc).__name__}: {exc}")


@router.get("/input-device")
def get_audio_input_device(request: Request):
    selected = _persisted_selection()
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None)
    runtime = getattr(engine, "runtime", None)
    stt = getattr(runtime, "stt", None)
    if stt is not None and hasattr(stt, "get_selected_input_device"):
        current = stt.get_selected_input_device()
        if current.get("device_id") or current.get("device_name"):
            selected.update(current)
    return selected


@router.post("/input-device")
async def set_audio_input_device(selection: InputDeviceSelection, request: Request):
    device_id = str(selection.device_id or "").strip()
    device_name = str(selection.device_name or "").strip()
    host_api = str(selection.host_api or "").strip()
    sample_rate = int(selection.sample_rate or 0) if selection.sample_rate else None
    channels = int(selection.channels or 0) if selection.channels else None
    signature = str(selection.signature or "").strip()

    db_sqlite.set_setting(SETTING_DEVICE_ID, device_id)
    db_sqlite.set_setting(SETTING_DEVICE_NAME, device_name)
    db_sqlite.set_setting(SETTING_DEVICE_HOST_API, host_api)
    db_sqlite.set_setting(SETTING_DEVICE_SAMPLE_RATE, str(sample_rate or ""))
    db_sqlite.set_setting(SETTING_DEVICE_CHANNELS, str(channels or ""))
    db_sqlite.set_setting(SETTING_DEVICE_SIGNATURE, signature)

    applied = False
    error = None
    adapter = getattr(request.app.state, "adapter", None)
    if adapter is not None and hasattr(adapter, "set_audio_input_device"):
        try:
            applied = bool(await adapter.set_audio_input_device(
                device_id=device_id,
                device_name=device_name,
                host_api=host_api,
                sample_rate=sample_rate,
                channels=channels,
                signature=signature,
            ))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[HEBE][STT][ERROR] apply selected input failed: {error}", flush=True)

    print(
        f"[HEBE][STT][DEVICE] selected id={device_id or '(default)'} name={device_name or '(default)'} "
        f"host_api={host_api or '(unknown)'} applied={applied}",
        flush=True,
    )
    return {
        "ok": error is None,
        "applied": applied,
        "error": error,
        "device_id": device_id,
        "device_name": device_name,
        "host_api": host_api,
        "sample_rate": sample_rate,
        "channels": channels,
        "signature": signature,
    }


@router.post("/input-device/test")
def test_selected_audio_input_device(selection: InputDeviceTestRequest):
    try:
        return test_audio_input_device(
            device_id=selection.device_id,
            device_name=selection.device_name,
            host_api=selection.host_api,
            seconds=selection.seconds,
        )
    except Exception as exc:
        print(f"[HEBE][STT][ERROR] raw mic test failed: {exc!r}", flush=True)
        raise HTTPException(status_code=500, detail=f"Mic test failed: {type(exc).__name__}: {exc}")
