"""RLM context variables (docs/rlm.md).

A context variable is a named handle whose content lives out of the prompt
window. Metadata (name, scope, kind, size, schema tag, provenance,
updated_at) is cheap to list; content is fetched only when a tool or API
caller asks for it and is never auto-inlined into a prompt.

Two scopes:

* ``session`` — private to one session; rows cascade with the session.
* ``agent`` — durable across every session of the agent (the world-model
  plane); rows live until archived.

Content follows the ``memories`` discipline: ``text`` column (TOAST handles
out-of-row storage), ``content_sha256`` + ``content_size_bytes`` with an
``octet_length`` CHECK, and a byte cap enforced here at the wire layer.
The cap is deliberately larger than the memories cap — spilled tool
results land here (see ``sandbox/tool_result_spill.py``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aios.actors import Actor

MAX_CONTENT_BYTES = 8_388_608  # 8 MiB
MAX_DESCRIPTION_CHARS = 1024

VariableScope = Literal["session", "agent"]
VariableKind = Literal["data", "helper", "spill", "digest"]

VARIABLE_NAME_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$"

VariableName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=VARIABLE_NAME_PATTERN,
        description=(
            "Variable handle: letters, digits, underscore, dot, dash; "
            "must start with a letter, digit, or underscore."
        ),
    ),
]


def check_content_size(content: str) -> int:
    """Return the utf-8 byte size of ``content``, raising past the cap."""
    size = len(content.encode("utf-8"))
    if size > MAX_CONTENT_BYTES:
        raise ValueError(f"content exceeds {MAX_CONTENT_BYTES}-byte cap ({size} bytes)")
    return size


class ContextVariableCreate(BaseModel):
    """Request body for ``POST /v1/context-variables``.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. The operator plane uses this for seeding world-model
    variables; sessions write through the ``ctx_write`` tool instead.
    """

    model_config = ConfigDict(extra="forbid")

    name: VariableName
    scope: VariableScope
    session_id: str | None = None
    agent_id: str | None = None
    content: str
    kind: VariableKind = "data"
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    schema_tag: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ContextVariableCreate:
        if self.scope == "session" and (self.session_id is None or self.agent_id is not None):
            raise ValueError("scope 'session' requires session_id and forbids agent_id")
        if self.scope == "agent" and (self.agent_id is None or self.session_id is not None):
            raise ValueError("scope 'agent' requires agent_id and forbids session_id")
        check_content_size(self.content)
        return self


class ContextVariableUpdate(BaseModel):
    """Request body for ``POST /v1/context-variables/{id}``."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    kind: VariableKind | None = None
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)
    schema_tag: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> ContextVariableUpdate:
        if self.content is not None:
            check_content_size(self.content)
        return self


class ContextVariable(BaseModel):
    """Read view. ``content`` only on retrieve — list responses carry
    metadata alone so listings stay cheap regardless of content size."""

    id: str
    type: Literal["context_variable"] = "context_variable"
    scope: VariableScope
    session_id: str | None = None
    agent_id: str | None = None
    name: str
    kind: VariableKind
    description: str
    schema_tag: str | None = None
    content: str | None = None
    content_sha256: str
    content_size_bytes: int
    metadata: dict[str, Any]
    created_by: Actor | None = None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
