from enum import Enum


class ContextVariableCreateKind(str, Enum):
    DATA = "data"
    DIGEST = "digest"
    HELPER = "helper"
    SPILL = "spill"

    def __str__(self) -> str:
        return str(self.value)
