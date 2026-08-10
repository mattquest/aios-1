from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.context_variable import ContextVariable
from ...models.context_variable_create import ContextVariableCreate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ContextVariableCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/context-variables",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ContextVariable | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = ContextVariable.from_dict(response.json())

        return response_201

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ContextVariable | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[ContextVariable | HTTPValidationError]:
    """Create

     Create (or overwrite) a variable at its scope key.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. Writing to an existing live ``(scope target, name)`` handle
    overwrites it in place — a variable is one value, not a merge. Content
    is capped at 8 MiB; the response echoes the row including content.

    Args:
        authorization (None | str | Unset):
        body (ContextVariableCreate): Request body for ``POST /v1/context-variables``.

            Exactly one of ``session_id`` / ``agent_id`` must be set, matching
            ``scope``. The operator plane uses this for seeding world-model
            variables; sessions write through the ``ctx_write`` tool instead.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextVariable | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableCreate,
    authorization: None | str | Unset = UNSET,
) -> ContextVariable | HTTPValidationError | None:
    """Create

     Create (or overwrite) a variable at its scope key.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. Writing to an existing live ``(scope target, name)`` handle
    overwrites it in place — a variable is one value, not a merge. Content
    is capped at 8 MiB; the response echoes the row including content.

    Args:
        authorization (None | str | Unset):
        body (ContextVariableCreate): Request body for ``POST /v1/context-variables``.

            Exactly one of ``session_id`` / ``agent_id`` must be set, matching
            ``scope``. The operator plane uses this for seeding world-model
            variables; sessions write through the ``ctx_write`` tool instead.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextVariable | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[ContextVariable | HTTPValidationError]:
    """Create

     Create (or overwrite) a variable at its scope key.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. Writing to an existing live ``(scope target, name)`` handle
    overwrites it in place — a variable is one value, not a merge. Content
    is capped at 8 MiB; the response echoes the row including content.

    Args:
        authorization (None | str | Unset):
        body (ContextVariableCreate): Request body for ``POST /v1/context-variables``.

            Exactly one of ``session_id`` / ``agent_id`` must be set, matching
            ``scope``. The operator plane uses this for seeding world-model
            variables; sessions write through the ``ctx_write`` tool instead.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextVariable | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableCreate,
    authorization: None | str | Unset = UNSET,
) -> ContextVariable | HTTPValidationError | None:
    """Create

     Create (or overwrite) a variable at its scope key.

    Exactly one of ``session_id`` / ``agent_id`` must be set, matching
    ``scope``. Writing to an existing live ``(scope target, name)`` handle
    overwrites it in place — a variable is one value, not a merge. Content
    is capped at 8 MiB; the response echoes the row including content.

    Args:
        authorization (None | str | Unset):
        body (ContextVariableCreate): Request body for ``POST /v1/context-variables``.

            Exactly one of ``session_id`` / ``agent_id`` must be set, matching
            ``scope``. The operator plane uses this for seeding world-model
            variables; sessions write through the ``ctx_write`` tool instead.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextVariable | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
