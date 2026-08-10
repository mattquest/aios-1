from enum import Enum


class ListContextVariablesScopeType0(str, Enum):
    AGENT = "agent"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
