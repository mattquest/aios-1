from enum import Enum


class ContextVariableUpdateKindType0(str, Enum):
    DATA = "data"
    DIGEST = "digest"
    HELPER = "helper"
    SPILL = "spill"

    def __str__(self) -> str:
        return str(self.value)
