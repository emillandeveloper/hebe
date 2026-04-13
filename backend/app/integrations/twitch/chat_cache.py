from collections import deque
from dataclasses import dataclass
import time


@dataclass
class ChatMessage:
    username: str
    display_name: str
    text: str
    ts: float


class TwitchChatCache:
    def __init__(self, max_messages: int = 200):
        self.messages = deque(maxlen=max_messages)

    def add_message(self, username: str, display_name: str, text: str):
        self.messages.append(ChatMessage(
            username=username,
            display_name=display_name,
            text=text,
            ts=time.time(),
        ))

    def recent_users(self) -> list[tuple[str, str]]:
        seen = {}
        for msg in reversed(self.messages):
            if msg.username not in seen:
                seen[msg.username] = msg.display_name
        return list(seen.items())

    def last_user(self):
        if not self.messages:
            return None
        msg = self.messages[-1]
        return msg.username