from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_context_variables_kind_type_0 import ListContextVariablesKindType0
from ...models.list_context_variables_scope_type_0 import ListContextVariablesScopeType0
from ...models.list_response_context_variable import ListResponseContextVariable
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    cursor: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    scope: ListContextVariablesScopeType0 | None | Unset = UNSET,
    kind: ListContextVariablesKindType0 | None | Unset = UNSET,
    after: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_cursor: None | str | Unset
    if isinstance(cursor, Unset):
        json_cursor = UNSET
    else:
        json_cursor = cursor
    params["cursor"] = json_cursor

    json_session_id: None | str | Unset
    if isinstance(session_id, Unset):
        json_session_id = UNSET
    else:
        json_session_id = session_id
    params["session_id"] = json_session_id

    json_agent_id: None | str | Unset
    if isinstance(agent_id, Unset):
        json_agent_id = UNSET
    else:
        json_agent_id = agent_id
    params["agent_id"] = json_agent_id

    json_scope: None | str | Unset
    if isinstance(scope, Unset):
        json_scope = UNSET
    elif isinstance(scope, ListContextVariablesScopeType0):
        json_scope = scope.value
    else:
        json_scope = scope
    params["scope"] = json_scope

    json_kind: None | str | Unset
    if isinstance(kind, Unset):
        json_kind = UNSET
    elif isinstance(kind, ListContextVariablesKindType0):
        json_kind = kind.value
    else:
        json_kind = kind
    params["kind"] = json_kind

    json_after: None | str | Unset
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/context-variables",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseContextVariable | None:
    if response.status_code == 200:
        response_200 = ListResponseContextVariable.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | ListResponseContextVariable]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    scope: ListContextVariablesScopeType0 | None | Unset = UNSET,
    kind: ListContextVariablesKindType0 | None | Unset = UNSET,
    after: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseContextVariable]:
    """List

     List context variables, newest first, keyset-paginated. Metadata only —
    ``content`` is ``None`` on every row regardless of size; fetch a single
    variable to read its content.

    First page: filters (``?session_id=``, ``?agent_id=``, ``?scope=``,
    ``?kind=``), ``?after=`` (keyset bound: rows with id before it), and
    ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>`` alone.

    Args:
        cursor (None | str | Unset):
        session_id (None | str | Unset):
        agent_id (None | str | Unset):
        scope (ListContextVariablesScopeType0 | None | Unset):
        kind (ListContextVariablesKindType0 | None | Unset):
        after (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseContextVariable]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        session_id=session_id,
        agent_id=agent_id,
        scope=scope,
        kind=kind,
        after=after,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    scope: ListContextVariablesScopeType0 | None | Unset = UNSET,
    kind: ListContextVariablesKindType0 | None | Unset = UNSET,
    after: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseContextVariable | None:
    """List

     List context variables, newest first, keyset-paginated. Metadata only —
    ``content`` is ``None`` on every row regardless of size; fetch a single
    variable to read its content.

    First page: filters (``?session_id=``, ``?agent_id=``, ``?scope=``,
    ``?kind=``), ``?after=`` (keyset bound: rows with id before it), and
    ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>`` alone.

    Args:
        cursor (None | str | Unset):
        session_id (None | str | Unset):
        agent_id (None | str | Unset):
        scope (ListContextVariablesScopeType0 | None | Unset):
        kind (ListContextVariablesKindType0 | None | Unset):
        after (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseContextVariable
    """

    return sync_detailed(
        client=client,
        cursor=cursor,
        session_id=session_id,
        agent_id=agent_id,
        scope=scope,
        kind=kind,
        after=after,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    scope: ListContextVariablesScopeType0 | None | Unset = UNSET,
    kind: ListContextVariablesKindType0 | None | Unset = UNSET,
    after: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseContextVariable]:
    """List

     List context variables, newest first, keyset-paginated. Metadata only —
    ``content`` is ``None`` on every row regardless of size; fetch a single
    variable to read its content.

    First page: filters (``?session_id=``, ``?agent_id=``, ``?scope=``,
    ``?kind=``), ``?after=`` (keyset bound: rows with id before it), and
    ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>`` alone.

    Args:
        cursor (None | str | Unset):
        session_id (None | str | Unset):
        agent_id (None | str | Unset):
        scope (ListContextVariablesScopeType0 | None | Unset):
        kind (ListContextVariablesKindType0 | None | Unset):
        after (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseContextVariable]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        session_id=session_id,
        agent_id=agent_id,
        scope=scope,
        kind=kind,
        after=after,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    scope: ListContextVariablesScopeType0 | None | Unset = UNSET,
    kind: ListContextVariablesKindType0 | None | Unset = UNSET,
    after: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseContextVariable | None:
    """List

     List context variables, newest first, keyset-paginated. Metadata only —
    ``content`` is ``None`` on every row regardless of size; fetch a single
    variable to read its content.

    First page: filters (``?session_id=``, ``?agent_id=``, ``?scope=``,
    ``?kind=``), ``?after=`` (keyset bound: rows with id before it), and
    ``?limit=``. Subsequent pages: ``?cursor=<next_cursor>`` alone.

    Args:
        cursor (None | str | Unset):
        session_id (None | str | Unset):
        agent_id (None | str | Unset):
        scope (ListContextVariablesScopeType0 | None | Unset):
        kind (ListContextVariablesKindType0 | None | Unset):
        after (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseContextVariable
    """

    return (
        await asyncio_detailed(
            client=client,
            cursor=cursor,
            session_id=session_id,
            agent_id=agent_id,
            scope=scope,
            kind=kind,
            after=after,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
