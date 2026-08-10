from enum import Enum


class ContextVariableKind(str, Enum):
    DATA = "data"
    DIGEST = "digest"
    HELPER = "helper"
    SPILL = "spill"

    def __str__(self) -> str:
        return str(self.value)
