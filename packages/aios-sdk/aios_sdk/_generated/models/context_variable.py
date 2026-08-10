from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.context_variable_kind import ContextVariableKind
from ..models.context_variable_scope import ContextVariableScope
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.actor import Actor
    from ..models.context_variable_metadata import ContextVariableMetadata


T = TypeVar("T", bound="ContextVariable")


@_attrs_define
class ContextVariable:
    """Read view. ``content`` only on retrieve — list responses carry
    metadata alone so listings stay cheap regardless of content size.

        Attributes:
            id (str):
            scope (ContextVariableScope):
            name (str):
            kind (ContextVariableKind):
            description (str):
            content_sha256 (str):
            content_size_bytes (int):
            metadata (ContextVariableMetadata):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            type_ (Literal['context_variable'] | Unset):  Default: 'context_variable'.
            session_id (None | str | Unset):
            agent_id (None | str | Unset):
            schema_tag (None | str | Unset):
            content (None | str | Unset):
            created_by (Actor | None | Unset):
            archived_at (datetime.datetime | None | Unset):
    """

    id: str
    scope: ContextVariableScope
    name: str
    kind: ContextVariableKind
    description: str
    content_sha256: str
    content_size_bytes: int
    metadata: ContextVariableMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    type_: Literal["context_variable"] | Unset = "context_variable"
    session_id: None | str | Unset = UNSET
    agent_id: None | str | Unset = UNSET
    schema_tag: None | str | Unset = UNSET
    content: None | str | Unset = UNSET
    created_by: Actor | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.actor import Actor

        id = self.id

        scope = self.scope.value

        name = self.name

        kind = self.kind.value

        description = self.description

        content_sha256 = self.content_sha256

        content_size_bytes = self.content_size_bytes

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        type_ = self.type_

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

        schema_tag: None | str | Unset
        if isinstance(self.schema_tag, Unset):
            schema_tag = UNSET
        else:
            schema_tag = self.schema_tag

        content: None | str | Unset
        if isinstance(self.content, Unset):
            content = UNSET
        else:
            content = self.content

        created_by: dict[str, Any] | None | Unset
        if isinstance(self.created_by, Unset):
            created_by = UNSET
        elif isinstance(self.created_by, Actor):
            created_by = self.created_by.to_dict()
        else:
            created_by = self.created_by

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "scope": scope,
                "name": name,
                "kind": kind,
                "description": description,
                "content_sha256": content_sha256,
                "content_size_bytes": content_size_bytes,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if schema_tag is not UNSET:
            field_dict["schema_tag"] = schema_tag
        if content is not UNSET:
            field_dict["content"] = content
        if created_by is not UNSET:
            field_dict["created_by"] = created_by
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.actor import Actor
        from ..models.context_variable_metadata import ContextVariableMetadata

        d = dict(src_dict)
        id = d.pop("id")

        scope = ContextVariableScope(d.pop("scope"))

        name = d.pop("name")

        kind = ContextVariableKind(d.pop("kind"))

        description = d.pop("description")

        content_sha256 = d.pop("content_sha256")

        content_size_bytes = d.pop("content_size_bytes")

        metadata = ContextVariableMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        type_ = cast(Literal["context_variable"] | Unset, d.pop("type", UNSET))
        if type_ != "context_variable" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'context_variable', got '{type_}'")

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

        def _parse_schema_tag(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        schema_tag = _parse_schema_tag(d.pop("schema_tag", UNSET))

        def _parse_content(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content = _parse_content(d.pop("content", UNSET))

        def _parse_created_by(data: object) -> Actor | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                created_by_type_0 = Actor.from_dict(data)

                return created_by_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(Actor | None | Unset, data)

        created_by = _parse_created_by(d.pop("created_by", UNSET))

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        context_variable = cls(
            id=id,
            scope=scope,
            name=name,
            kind=kind,
            description=description,
            content_sha256=content_sha256,
            content_size_bytes=content_size_bytes,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            session_id=session_id,
            agent_id=agent_id,
            schema_tag=schema_tag,
            content=content,
            created_by=created_by,
            archived_at=archived_at,
        )

        context_variable.additional_properties = d
        return context_variable

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
