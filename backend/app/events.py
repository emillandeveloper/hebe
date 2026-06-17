from pydantic import BaseModel
from typing import Any, Literal


class Event(BaseModel):
  # Str para que el backend pueda emitir nuevos eventos sin romperse (tool.*, etc.)
  type: str
  data: Any
  ts: float
  event_id: str | None = None


class ClientMsg(BaseModel):
  type: Literal["client.message", "client.command"]
  data: dict
