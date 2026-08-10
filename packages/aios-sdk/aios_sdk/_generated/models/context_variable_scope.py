from enum import Enum


class ContextVariableScope(str, Enum):
    AGENT = "agent"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
