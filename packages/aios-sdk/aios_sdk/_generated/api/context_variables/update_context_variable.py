from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.context_variable import ContextVariable
from ...models.context_variable_update import ContextVariableUpdate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    variable_id: str,
    *,
    body: ContextVariableUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/context-variables/{variable_id}".format(
            variable_id=quote(str(variable_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ContextVariable | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ContextVariable.from_dict(response.json())

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
) -> Response[ContextVariable | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    variable_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[ContextVariable | HTTPValidationError]:
    """Update

     Partially update a variable by id. Omitted fields are preserved;
    ``content`` (when sent) replaces the stored value and recomputes the
    sha256/size pair.

    Args:
        variable_id (str):
        authorization (None | str | Unset):
        body (ContextVariableUpdate): Request body for ``POST /v1/context-variables/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextVariable | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        variable_id=variable_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    variable_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableUpdate,
    authorization: None | str | Unset = UNSET,
) -> ContextVariable | HTTPValidationError | None:
    """Update

     Partially update a variable by id. Omitted fields are preserved;
    ``content`` (when sent) replaces the stored value and recomputes the
    sha256/size pair.

    Args:
        variable_id (str):
        authorization (None | str | Unset):
        body (ContextVariableUpdate): Request body for ``POST /v1/context-variables/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextVariable | HTTPValidationError
    """

    return sync_detailed(
        variable_id=variable_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    variable_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[ContextVariable | HTTPValidationError]:
    """Update

     Partially update a variable by id. Omitted fields are preserved;
    ``content`` (when sent) replaces the stored value and recomputes the
    sha256/size pair.

    Args:
        variable_id (str):
        authorization (None | str | Unset):
        body (ContextVariableUpdate): Request body for ``POST /v1/context-variables/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextVariable | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        variable_id=variable_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    variable_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ContextVariableUpdate,
    authorization: None | str | Unset = UNSET,
) -> ContextVariable | HTTPValidationError | None:
    """Update

     Partially update a variable by id. Omitted fields are preserved;
    ``content`` (when sent) replaces the stored value and recomputes the
    sha256/size pair.

    Args:
        variable_id (str):
        authorization (None | str | Unset):
        body (ContextVariableUpdate): Request body for ``POST /v1/context-variables/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextVariable | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            variable_id=variable_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
