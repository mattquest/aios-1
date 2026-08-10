from enum import Enum


class ContextVariableCreateScope(str, Enum):
    AGENT = "agent"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
