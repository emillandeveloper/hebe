# services/command_router.py

from dataclasses import dataclass
from typing import Callable, List, Optional
import re

Decision = str  # "continue" | "sleep" | "stop"

@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    handler: Callable[[str], Decision]


class CommandRouter:
    def __init__(self):
        self.rules: List[Rule] = []

    def add(self, name: str, regex: str, handler: Callable[[str], Decision]) -> None:
        self.rules.append(
            Rule(name=name, pattern=re.compile(regex, re.I), handler=handler)
        )

    Decision = str  # "continue" | "sleep" | "stop"

    def route(self, text: str) -> Optional[Decision]:
        if not text:
            return "continue"

        for rule in self.rules:
            if rule.pattern.search(text):
                return rule.handler(text)

        return None  # <-- IMPORTANTE: no match