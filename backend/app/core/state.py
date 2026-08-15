from dataclasses import dataclass, field
from typing import Optional, Any
from app.stream.state import StreamSessionState
from app.cognitive.game_guidance import GameRunState

@dataclass
class HebeState:
    mode: str = "active"   # active | sleep | stream | focused
    hebe_sleeping: bool = False
    is_running: bool = False
    is_processing: bool = False
    tts_enabled: bool = False

    last_input_text: Optional[str] = None
    last_input_source: Optional[str] = None
    last_intent: Optional[str] = None

    current_task: Optional[str] = None
    current_context: dict[str, Any] = field(default_factory=dict)
    game_run_state: GameRunState = field(default_factory=GameRunState)

    stream: StreamSessionState = field(default_factory=StreamSessionState)
    
