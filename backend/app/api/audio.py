from __future__ import annotations

import os
import re
import time
import wave
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services import db_sqlite
from app.services.stt_whisper import WhisperModel, list_audio_devices, test_audio_input_device


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


class STTBenchmarkClip(BaseModel):
    path: str
    corrected_text: Optional[str] = None
    expected_wake: Optional[bool] = None
    expected_intent: Optional[str] = None
    expected_target: Optional[str] = None


class STTBenchmarkRequest(BaseModel):
    clips: list[STTBenchmarkClip]
    profile_ids: list[str] = Field(default_factory=lambda: ["A", "B", "C"])


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


@router.get("/stt-health")
def stt_health(request: Request):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None)
    stt = getattr(getattr(engine, "runtime", None), "stt", None)
    if stt is None:
        return {"available": False, "engine_status": "unavailable"}
    return {"available": True, **stt.health_snapshot()}


@router.get("/stt-benchmark/profiles")
def stt_benchmark_profiles():
    """Manual comparison contract. A caller supplies the same saved clips to each profile."""
    return {
        "automatic_winner": False,
        "profiles": [
            {"id": "A", "model": "small", "device": "cpu", "compute_type": "int8"},
            {"id": "B", "model": "medium", "device": "cuda", "compute_type": "int8_float16"},
            {"id": "C", "model": "medium", "device": "cuda", "compute_type": "float16"},
        ],
        "metrics": [
            "exact_command_intent_accuracy", "wake_detection", "application_target_accuracy",
            "promotion_target_accuracy", "word_error_rate", "latency", "real_time_factor",
            "gpu_memory", "rejection_rate", "clarification_rate",
        ],
        "required_clip_kinds": [
            "direct_obs_command", "promotion_super_damu", "spanish_speech", "english_speech",
            "mixed_spanish_english_game", "partial_command", "speech_with_game_audio",
        ],
    }


def _benchmark_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


def _word_error_rate(reference: str, hypothesis: str) -> float:
    left, right = _benchmark_tokens(reference), _benchmark_tokens(hypothesis)
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1] / len(left)


@router.post("/stt-benchmark/run")
def run_stt_benchmark(request: STTBenchmarkRequest):
    """Explicit manual benchmark; it never changes the configured winner/profile."""
    profiles = {
        "A": ("small", "cpu", "int8"),
        "B": ("medium", "cuda", "int8_float16"),
        "C": ("medium", "cuda", "float16"),
    }
    resolved_clips = []
    for clip in request.clips:
        path = Path(clip.path).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Diagnostic clip not found: {path}")
        resolved_clips.append((clip, path))
    results = []
    for profile_id in request.profile_ids:
        if profile_id not in profiles:
            raise HTTPException(status_code=400, detail=f"Unknown benchmark profile: {profile_id}")
        model_name, device, compute = profiles[profile_id]
        load_started = time.perf_counter()
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as exc:
            results.append({"profile_id": profile_id, "model": model_name, "device": device, "compute_type": compute, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        profile_result = {"profile_id": profile_id, "model": model_name, "device": device, "compute_type": compute, "status": "ready", "load_seconds": time.perf_counter() - load_started, "clips": []}
        for clip, path in resolved_clips:
            with wave.open(str(path), "rb") as wav:
                audio_seconds = wav.getnframes() / max(1, wav.getframerate())
            started = time.perf_counter()
            segments, info = model.transcribe(str(path), task="transcribe", vad_filter=True)
            text = "".join(segment.text for segment in segments).strip()
            latency = time.perf_counter() - started
            tokens = set(_benchmark_tokens(text))
            wake = bool(tokens & {"hebe", "ebe", "eve", "heve", "jebe"})
            inferred_intent = (
                "open_application" if tokens & {"abre", "abrir", "open", "launch"} and len(tokens & {"obs", "twitch"}) > 0
                else "promotion" if tokens & {"promo", "shoutout"}
                else ""
            )
            target_ok = None if not clip.expected_target else clip.expected_target.casefold() in " ".join(tokens)
            wake_ok = None if clip.expected_wake is None else wake == clip.expected_wake
            profile_result["clips"].append({
                "path": str(path), "text": text, "detected_language": getattr(info, "language", None),
                "latency_seconds": latency, "real_time_factor": latency / audio_seconds if audio_seconds else 0.0,
                "word_error_rate": _word_error_rate(clip.corrected_text, text) if clip.corrected_text is not None else None,
                "wake_detected": wake, "wake_correct": wake_ok, "target_correct": target_ok,
                "expected_intent": clip.expected_intent, "inferred_intent": inferred_intent,
                "intent_correct": None if not clip.expected_intent else inferred_intent == clip.expected_intent,
            })
        latencies = [item["latency_seconds"] for item in profile_result["clips"]]
        profile_result["average_latency_seconds"] = sum(latencies) / len(latencies) if latencies else 0.0
        scored_intents = [item["intent_correct"] for item in profile_result["clips"] if item["intent_correct"] is not None]
        scored_wakes = [item["wake_correct"] for item in profile_result["clips"] if item["wake_correct"] is not None]
        scored_targets = [item["target_correct"] for item in profile_result["clips"] if item["target_correct"] is not None]
        profile_result["exact_command_intent_accuracy"] = sum(scored_intents) / len(scored_intents) if scored_intents else None
        profile_result["wake_detection_accuracy"] = sum(scored_wakes) / len(scored_wakes) if scored_wakes else None
        profile_result["target_accuracy"] = sum(scored_targets) / len(scored_targets) if scored_targets else None
        profile_result["rejection_rate"] = sum(not item["text"] for item in profile_result["clips"]) / len(profile_result["clips"]) if profile_result["clips"] else 0.0
        profile_result["clarification_rate"] = sum(bool(item["expected_intent"] and not item["intent_correct"] and item["text"]) for item in profile_result["clips"]) / len(profile_result["clips"]) if profile_result["clips"] else 0.0
        profile_result["gpu_memory_bytes"] = None
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                profile_result["gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        results.append(profile_result)
        del model
    return {"automatic_winner": False, "profiles": results}
