from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.context_variable_update_kind_type_0 import ContextVariableUpdateKindType0
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.context_variable_update_metadata_type_0 import (
        ContextVariableUpdateMetadataType0,
    )


T = TypeVar("T", bound="ContextVariableUpdate")


@_attrs_define
class ContextVariableUpdate:
    """Request body for ``POST /v1/context-variables/{id}``.

    Attributes:
        content (None | str | Unset):
        kind (ContextVariableUpdateKindType0 | None | Unset):
        description (None | str | Unset):
        schema_tag (None | str | Unset):
        metadata (ContextVariableUpdateMetadataType0 | None | Unset):
    """

    content: None | str | Unset = UNSET
    kind: ContextVariableUpdateKindType0 | None | Unset = UNSET
    description: None | str | Unset = UNSET
    schema_tag: None | str | Unset = UNSET
    metadata: ContextVariableUpdateMetadataType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.context_variable_update_metadata_type_0 import (
            ContextVariableUpdateMetadataType0,
        )

        content: None | str | Unset
        if isinstance(self.content, Unset):
            content = UNSET
        else:
            content = self.content

        kind: None | str | Unset
        if isinstance(self.kind, Unset):
            kind = UNSET
        elif isinstance(self.kind, ContextVariableUpdateKindType0):
            kind = self.kind.value
        else:
            kind = self.kind

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        schema_tag: None | str | Unset
        if isinstance(self.schema_tag, Unset):
            schema_tag = UNSET
        else:
            schema_tag = self.schema_tag

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, ContextVariableUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if content is not UNSET:
            field_dict["content"] = content
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
        from ..models.context_variable_update_metadata_type_0 import (
            ContextVariableUpdateMetadataType0,
        )

        d = dict(src_dict)

        def _parse_content(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content = _parse_content(d.pop("content", UNSET))

        def _parse_kind(data: object) -> ContextVariableUpdateKindType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                kind_type_0 = ContextVariableUpdateKindType0(data)

                return kind_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ContextVariableUpdateKindType0 | None | Unset, data)

        kind = _parse_kind(d.pop("kind", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_schema_tag(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        schema_tag = _parse_schema_tag(d.pop("schema_tag", UNSET))

        def _parse_metadata(
            data: object,
        ) -> ContextVariableUpdateMetadataType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = ContextVariableUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ContextVariableUpdateMetadataType0 | None | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        context_variable_update = cls(
            content=content,
            kind=kind,
            description=description,
            schema_tag=schema_tag,
            metadata=metadata,
        )

        return context_variable_update
