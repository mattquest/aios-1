"""``aios vars ...`` — context-variable CRUD (RLM metadata plane)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aios.cli.commands._shared import get_state_and_client, raw_paginate, raw_single, render_list
from aios.cli.coverage import covers
from aios.cli.files import load_payload
from aios.cli.output import print_success
from aios.cli.runtime import get_state, run_or_die
from aios_sdk import raw_request

app = typer.Typer(name="vars", help="Manage context variables.", no_args_is_help=True)

_VAR_COLS = ("id", "name", "scope", "kind", "content_size_bytes", "updated_at")


@app.command("list", help="List context variables (metadata only).")
@covers("list_context_variables")
def list_(
    ctx: typer.Context,
    session_id: Annotated[
        str | None, typer.Option("--session", help="Filter by owning session id.")
    ] = None,
    agent_id: Annotated[
        str | None, typer.Option("--agent", help="Filter by owning agent id.")
    ] = None,
    scope: Annotated[
        str | None, typer.Option("--scope", help="Filter by scope: session, agent.")
    ] = None,
    kind: Annotated[
        str | None, typer.Option("--kind", help="Filter by kind: data, helper, spill, digest.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
    after: Annotated[
        str | None,
        typer.Option("--after", help="Keyset bound: return variables with id before it."),
    ] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    def _run() -> None:
        state, client = get_state_and_client(ctx)
        params: dict[str, Any] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "scope": scope,
            "kind": kind,
            "after": after,
        }
        with client:
            if all_:
                envelope = raw_paginate(client, "/v1/context-variables", params=params)
            else:
                envelope = raw_request(
                    client,
                    "GET",
                    "/v1/context-variables",
                    params={**params, "limit": limit},
                )
        render_list(state.output_format, envelope, columns=_VAR_COLS)

    run_or_die(_run)


@app.command("get", help="Fetch a context variable by id (includes content).")
@covers("get_context_variable")
def get(ctx: typer.Context, variable_id: str) -> None:
    def _run() -> None:
        raw_single(ctx, "GET", f"/v1/context-variables/{variable_id}")

    run_or_die(_run)


@app.command("create", help="Create a context variable (ContextVariableCreate shape).")
@covers("create_context_variable")
def create(
    ctx: typer.Context,
    file: Annotated[Path | None, typer.Option("--file", help="Read JSON body from a file.")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read JSON body from stdin.")] = False,
    data: Annotated[str | None, typer.Option("--data", help="Inline JSON body.")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        raw_single(ctx, "POST", "/v1/context-variables", json_body=payload)
        return None

    run_or_die(_run)


@app.command("update", help="Update a context variable (ContextVariableUpdate shape).")
@covers("update_context_variable")
def update(
    ctx: typer.Context,
    variable_id: str,
    file: Annotated[Path | None, typer.Option("--file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    data: Annotated[str | None, typer.Option("--data")] = None,
) -> None:
    def _run() -> int | None:
        payload = load_payload(file, stdin, data)
        raw_single(ctx, "POST", f"/v1/context-variables/{variable_id}", json_body=payload)
        return None

    run_or_die(_run)


@app.command("delete", help="Archive a context variable (soft-delete, retained for audit).")
@covers("archive_context_variable")
def delete(ctx: typer.Context, variable_id: str) -> None:
    def _run() -> None:
        with get_state(ctx).sdk_client() as client:
            raw_request(client, "DELETE", f"/v1/context-variables/{variable_id}")
        print_success("archived", variable_id)

    run_or_die(_run)
