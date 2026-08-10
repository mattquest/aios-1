from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.context_variable_create_kind import ContextVariableCreateKind
from ..models.context_variable_create_scope import ContextVariableCreateScope
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.context_variable_create_metadata import ContextVariableCreateMetadata


T = TypeVar("T", bound="ContextVariableCreate")


@_attrs_define
class ContextVariableCreate:
    """Request body for ``POST /v1/context-variables``.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. The operator plane uses this for seeding world-model
    variables; sessions write through the ``ctx_write`` tool instead.

        Attributes:
            name (str): Variable handle: letters, digits, underscore, dot, dash; must start with a letter, digit, or
                underscore.
            scope (ContextVariableCreateScope):
            content (str):
            session_id (None | str | Unset):
            agent_id (None | str | Unset):
            kind (ContextVariableCreateKind | Unset):  Default: ContextVariableCreateKind.DATA.
            description (str | Unset):  Default: ''.
            schema_tag (None | str | Unset):
            metadata (ContextVariableCreateMetadata | Unset):
    """

    name: str
    scope: ContextVariableCreateScope
    content: str
    session_id: None | str | Unset = UNSET
    agent_id: None | str | Unset = UNSET
    kind: ContextVariableCreateKind | Unset = ContextVariableCreateKind.DATA
    description: str | Unset = ""
    schema_tag: None | str | Unset = UNSET
    metadata: ContextVariableCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        scope = self.scope.value

        content = self.content

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        else:
            session_id = self.session_id

        agent_id: None | str | Unset
        if isinstance(self.agent_id, Unset):
            agent_id = UNSET
        else:
            agent_id = self.agent_id

        kind: str | Unset = UNSET
        if not isinstance(self.kind, Unset):
            kind = self.kind.value

        description = self.description

        schema_tag: None | str | Unset
        if isinstance(self.schema_tag, Unset):
            schema_tag = UNSET
        else:
            schema_tag = self.schema_tag

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "scope": scope,
                "content": content,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if kind is not UNSET:
            field_dict["kind"] = kind
        if description is not UNSET:
            field_dict["description"] = description
        if schema_tag is not UNSET:
            field_dict["schema_tag"] = schema_tag
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.context_variable_create_metadata import (
            ContextVariableCreateMetadata,
        )

        d = dict(src_dict)
        name = d.pop("name")

        scope = ContextVariableCreateScope(d.pop("scope"))

        content = d.pop("content")

        def _parse_session_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_id = _parse_session_id(d.pop("session_id", UNSET))

        def _parse_agent_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        agent_id = _parse_agent_id(d.pop("agent_id", UNSET))

        _kind = d.pop("kind", UNSET)
        kind: ContextVariableCreateKind | Unset
        if isinstance(_kind, Unset):
            kind = UNSET
        else:
            kind = ContextVariableCreateKind(_kind)

        description = d.pop("description", UNSET)

        def _parse_schema_tag(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        schema_tag = _parse_schema_tag(d.pop("schema_tag", UNSET))

        _metadata = d.pop("metadata", UNSET)
        metadata: ContextVariableCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ContextVariableCreateMetadata.from_dict(_metadata)

        context_variable_create = cls(
            name=name,
            scope=scope,
            content=content,
            session_id=session_id,
            agent_id=agent_id,
            kind=kind,
            description=description,
            schema_tag=schema_tag,
            metadata=metadata,
        )

        return context_variable_create
