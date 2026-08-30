#!/usr/bin/env python3
"""Reject non-database awaits while an asyncpg pool connection is held."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PRAGMA = "pooled-connection-await: allow"
_ISSUE_REF = re.compile(r"(?<![A-Za-z0-9_.-])eumemic/aios#(\d+)(?!\d)")


@dataclass(frozen=True)
class Violation:
    filename: str
    line: int
    column: int
    message: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}: PCA001 {self.message}"


def _expression_name(node: ast.AST) -> str | None:
    """Return an exact dotted expression; never collapse ``self.conn`` to ``self``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


# Explicit repository DB surfaces.  Additions are reviewable: arbitrary names,
# suffixes, and service objects are deliberately not inferred as database I/O.
_DB_HELPER_SYMBOLS = frozenset(
    {
        "_advance_open_request_scan_floor_best_effort",
        "_allocate_version_seq",
        "_append_fire_event",
        "_append_transition",
        "_archive_binding_or_raise",
        "_assert_env_var_creds_contained",
        "_assert_no_residue",
        "_batch_list_all_echoes",
        "_cancel_run",
        "_classify_existing_tool_result",
        "_complete_run",
        "_current_alembic_version",
        "_dedup_skip",
        "_enrich_agent_result",
        "_enrich_session",
        "_errored_session_ids",
        "_fail_child_requests_for_terminal_error",
        "_insert_workflow_version",
        "_journal_agent_rejection",
        "_latest_cumulative_state",
        "_list_all_echoes",
        "_list_all_writable_store_ids",
        "_list_attached_resource_ids",
        "_load_for_session_conn",
        "_load_surfaces",
        "_open_agent_capability",
        "_open_invoke_workflow_capability",
        "_queries.list_session_memory_store_echoes",
        "_quiescence_owed_surfacing",
        "_record_timer_audit",
        "_rekey_column",
        "_resolve_agent_call",
        "_session_owned",
        "_walk",
        "account_purge.emit_account_purge_invalidations",
        "accounts_service.resolve_effective_timezone_on",
        "agents_service.load_for_session",
        "agents_service.validate_pinned_agent_version",
        "append_tool_result",
        "archive_session_conn",
        "attach_to_session",
        "calibration_telemetry",
        "db_queries.insert_run_completion_fires",
        "db_queries.insert_session_cancel_marker",
        "db_queries.write_response_if_absent",
        "fail_open_child_requests_conn",
        "find_parked_servicer",
        "find_unharvested_model_dispatch_parks",
        "get_agent",
        "get_environment",
        "get_session_bare",
        "get_session_vault_ids",
        "get_session_workspace_path",
        "get_workflow",
        "github_repo_service.add_one",
        "github_repo_service.attach_to_session",
        "github_repo_service.detach_all_from_session",
        "github_repo_service.get_session_token",
        "github_repo_service.remove_one",
        "github_repo_service.set_session_resources",
        "list_session_ids_with_unharvested_cancel_marker",
        "materialize_store_to_host",
        "memory_service.add_one",
        "memory_service.attach_to_session",
        "memory_service.remove_one",
        "memory_service.set_session_resources",
        "prune",
        "queries.acquire_account_cascade_cleanup_lock",
        "queries.acquire_account_triggers_lock",
        "queries.acquire_session_resources_lock",
        "queries.acquire_workspace_advisory_xact_lock",
        "queries.add_rlm_child_tokens",
        "queries.add_trigger",
        "queries.admit_rlm_child",
        "queries.advance_open_request_scan_floor",
        "queries.append_event",
        "queries.append_request_opened",
        "queries.archive_account",
        "queries.archive_active_binding",
        "queries.archive_agent",
        "queries.archive_connection",
        "queries.archive_context_variable",
        "queries.archive_environment",
        "queries.archive_memory_store",
        "queries.archive_model_provider",
        "queries.archive_session",
        "queries.archive_session_template",
        "queries.archive_skill",
        "queries.archive_vault",
        "queries.archive_vault_credential",
        "queries.attach_github_repos_to_session",
        "queries.attach_memory_stores_to_session",
        "queries.audit_credentialless_root",
        "queries.batch_get_session_vault_ids",
        "queries.batch_list_session_github_repo_echoes",
        "queries.batch_list_session_memory_store_echoes",
        "queries.batch_list_session_triggers",
        "queries.begin_account_cascade_cleanup_attempt",
        "queries.bootstrap_root_account",
        "queries.build_account_cascade_purge_manifest",
        "queries.cascade_hard_delete_account_resources",
        "queries.children_of",
        "queries.claim_trigger_run",
        "queries.clone_session",
        "queries.complete_account_cascade_cleanup",
        "queries.copy_session_github_resources",
        "queries.copy_session_resources",
        "queries.count_account_resources",
        "queries.count_account_triggers",
        "queries.count_active_child_accounts",
        "queries.count_active_vault_credentials",
        "queries.count_child_accounts",
        "queries.count_request_nudges",
        "queries.count_session_triggers",
        "queries.count_stuck_running_trigger_runs",
        "queries.decrement_open_tool_call_count",
        "queries.default_inbound_policy_if_unset",
        "queries.delete_chat_session",
        "queries.delete_expired_oauth_flows",
        "queries.delete_memory_store",
        "queries.delete_memory_with_version",
        "queries.delete_oauth_flow",
        "queries.delete_session",
        "queries.delete_session_github_repo",
        "queries.delete_session_github_repos",
        "queries.delete_session_memory_store",
        "queries.delete_trigger_by_id",
        "queries.delete_vault",
        "queries.delete_vault_credential",
        "queries.derive_response",
        "queries.derive_session_status",
        "queries.disable_trigger",
        "queries.fetch_and_claim_due_triggers",
        "queries.fetch_next_trigger_event",
        "queries.finalize_trigger_run",
        "queries.find_latest_interrupt_seq",
        "queries.find_latest_model_workflow_park",
        "queries.find_model_workflow_harvest",
        "queries.find_parked_servicer",
        "queries.find_tool_confirmed_event",
        "queries.find_tool_confirmed_seqs",
        "queries.find_tool_result_event",
        "queries.find_user_message_by_client_message_id",
        "queries.gc_snapshot_session_states",
        "queries.get_account",
        "queries.get_account_cascade_purge_receipt",
        "queries.get_account_for_update",
        "queries.get_account_spent_microusd",
        "queries.get_account_subtree_spent_microusd",
        "queries.get_active_binding",
        "queries.get_active_credential_by_target_url",
        "queries.get_agent",
        "queries.get_agent_version",
        "queries.get_chat_session_row",
        "queries.get_closed_request",
        "queries.get_connection",
        "queries.get_connection_secret_blob",
        "queries.get_context_variable",
        "queries.get_environment",
        "queries.get_environment_config_for_id",
        "queries.get_environment_config_for_session",
        "queries.get_event",
        "queries.get_management_call",
        "queries.get_memory",
        "queries.get_memory_by_path",
        "queries.get_memory_store",
        "queries.get_memory_version",
        "queries.get_model_provider",
        "queries.get_oauth_flow_for_complete",
        "queries.get_open_obligations",
        "queries.get_open_obligations_batch",
        "queries.get_open_request_ids",
        "queries.get_or_create_account_placeholder_salt",
        "queries.get_request_caller",
        "queries.get_request_output_schema",
        "queries.get_rlm_spawn_budget",
        "queries.get_session",
        "queries.get_session_bare",
        "queries.get_session_env",
        "queries.get_session_event_stats",
        "queries.get_session_focal_channel",
        "queries.get_session_frozen_litellm_extra",
        "queries.get_session_frozen_surface",
        "queries.get_session_github_repo",
        "queries.get_session_github_repo_with_blob",
        "queries.get_session_last_user_seq",
        "queries.get_session_model",
        "queries.get_session_provisioning",
        "queries.get_session_template",
        "queries.get_session_usage",
        "queries.get_session_vault_ids",
        "queries.get_session_workflow_context",
        "queries.get_session_workspace_path",
        "queries.get_skill",
        "queries.get_skill_version",
        "queries.get_tool_call_parent_seq",
        "queries.get_trigger_by_name",
        "queries.get_vault",
        "queries.get_vault_credential",
        "queries.get_vault_credential_with_blob",
        "queries.get_wake_priority_context",
        "queries.hard_delete_account",
        "queries.has_active_root_account",
        "queries.increment_session_usage",
        "queries.insert_account_cascade_purge_receipt",
        "queries.insert_account_key",
        "queries.insert_agent",
        "queries.insert_binding",
        "queries.insert_chat_session",
        "queries.insert_child_account",
        "queries.insert_child_session",
        "queries.insert_connection",
        "queries.insert_context_variable_grants",
        "queries.insert_environment",
        "queries.insert_external_event_fire",
        "queries.insert_file",
        "queries.insert_management_call",
        "queries.insert_memory_store",
        "queries.insert_memory_with_version",
        "queries.insert_model_provider",
        "queries.insert_oauth_flow",
        "queries.insert_runtime_token",
        "queries.insert_session",
        "queries.insert_session_cancel_marker",
        "queries.insert_session_github_repo",
        "queries.insert_session_memory_store",
        "queries.insert_session_template",
        "queries.insert_skill",
        "queries.insert_skill_version",
        "queries.insert_vault",
        "queries.insert_vault_credential",
        "queries.is_session_bound_to_connection",
        "queries.is_session_focal_locked",
        "queries.list_account_keys",
        "queries.list_account_triggers",
        "queries.list_active_memory_paths_and_content",
        "queries.list_agent_versions",
        "queries.list_agents",
        "queries.list_attachment_paths_for_sessions",
        "queries.list_chat_sessions_for_connection",
        "queries.list_child_accounts",
        "queries.list_confirmed_unresolved_tool_calls",
        "queries.list_connection_capabilities_for_session",
        "queries.list_connection_tools_for_session",
        "queries.list_connections",
        "queries.list_context_variables",
        "queries.list_environments",
        "queries.list_memories",
        "queries.list_memory_stores",
        "queries.list_memory_versions",
        "queries.list_model_providers",
        "queries.list_pending_calls_for_connector",
        "queries.list_pending_calls_for_session_and_connection",
        "queries.list_pending_management_calls_for_connector",
        "queries.list_pending_trigger_run_refs",
        "queries.list_readable_variables",
        "queries.list_recent_chat_ids",
        "queries.list_routing_rules_for_connection",
        "queries.list_run_env_var_credentials",
        "queries.list_runtime_tokens",
        "queries.list_session_channels",
        "queries.list_session_env_var_credential_echoes",
        "queries.list_session_env_var_credentials",
        "queries.list_session_github_repo_echoes",
        "queries.list_session_github_repo_ranks",
        "queries.list_session_ids_for_connection",
        "queries.list_session_memory_store_echoes",
        "queries.list_session_memory_store_ranks",
        "queries.list_session_templates",
        "queries.list_sessions",
        "queries.list_skill_versions",
        "queries.list_skills",
        "queries.list_trigger_runs",
        "queries.list_triggers",
        "queries.list_unharvested_session_cancel_markers",
        "queries.list_unresolved_tool_calls_batch",
        "queries.list_vault_credentials",
        "queries.list_vaults",
        "queries.lock_active_session_for_update",
        "queries.lock_and_cancel_queued_account_jobs",
        "queries.lock_oauth_credential_for_refresh",
        "queries.lock_writable_session_for_update",
        "queries.lookup_account_by_key_hash",
        "queries.lookup_chat_session",
        "queries.lookup_tool_name_by_call_id",
        "queries.mark_management_call_resolved",
        "queries.mark_session_cancel_marker_harvested",
        "queries.model_token_class_ratios",
        "queries.notify_connection_change",
        "queries.notify_management_call_dispatch",
        "queries.notify_management_call_result",
        "queries.precompute_event_append",
        "queries.prune_trigger_runs",
        "queries.read_events",
        "queries.read_message_events",
        "queries.read_request_response",
        "queries.read_session_watermarks",
        "queries.read_windowed_context_events",
        "queries.read_windowed_events",
        "queries.reclaim_session_if_idle",
        "queries.recompute_session_channels",
        "queries.record_account_cascade_cleanup_error",
        "queries.record_trigger_fire",
        "queries.record_trigger_run",
        "queries.redact_memory_version",
        "queries.release_account_cascade_cleanup_lock",
        "queries.release_trigger_claim",
        "queries.remove_trigger",
        "queries.reparent_connection",
        "queries.replace_event_data",
        "queries.resolve_account_by_path",
        "queries.resolve_effective_sandbox_snapshot_bytes",
        "queries.resolve_effective_spend_limit_usd",
        "queries.resolve_effective_timezone",
        "queries.resolve_external_event_trigger",
        "queries.resolve_model_provider",
        "queries.resolve_readable_variable",
        "queries.resolve_run_credential",
        "queries.resolve_runtime_token",
        "queries.resolve_session_credential",
        "queries.resolve_skill_refs",
        "queries.resolve_vault_credential",
        "queries.revoke_account_key",
        "queries.revoke_runtime_token",
        "queries.set_connection_inbound_policy",
        "queries.set_connection_secrets",
        "queries.set_session_channels",
        "queries.set_session_focal_channel",
        "queries.set_session_stop_reason",
        "queries.set_session_vaults",
        "queries.spill_tool_result_variable",
        "queries.sum_account_session_tokens",
        "queries.summarize_session_tool_calls",
        "queries.tool_error_counts_since_last_user",
        "queries.try_record_inbound_ack",
        "queries.unscoped_clear_session_snapshot",
        "queries.unscoped_clear_session_snapshot_if_matches",
        "queries.unscoped_get_session_account_id",
        "queries.unscoped_get_session_snapshot_bytes",
        "queries.unscoped_get_session_spec_version",
        "queries.unscoped_get_trigger_row",
        "queries.unscoped_live_session_account_id",
        "queries.unscoped_live_session_ids",
        "queries.unscoped_live_workspace_volume_paths",
        "queries.unscoped_reapable_archived_workspaces",
        "queries.unscoped_set_session_snapshot",
        "queries.unscoped_workspace_path_is_live",
        "queries.update_account",
        "queries.update_agent",
        "queries.update_connector_capabilities",
        "queries.update_connector_tools_schema",
        "queries.update_context_variable",
        "queries.update_environment",
        "queries.update_memory_store",
        "queries.update_memory_with_version",
        "queries.update_model_provider",
        "queries.update_session",
        "queries.update_session_github_repo_blob",
        "queries.update_session_template",
        "queries.update_trigger",
        "queries.update_vault",
        "queries.update_vault_credential",
        "queries.upsert_context_variable",
        "queries.write_response_if_absent",
        "read_request_response",
        "resolve_effective_spend_limit_usd_on",
        "resolve_effective_timezone_on",
        "resolve_run_env_var_credentials",
        "resolve_session_env_var_credentials",
        "respond_to_request_conn",
        "seed_outbound_cancel_conn",
        "service.append_tool_result",
        "session_has_pending_work",
        "sessions_service.append_tool_result",
        "sessions_service.archive_session_conn",
        "sessions_service.respond_to_request_conn",
        "trace_q.children_of",
        "trace_q.list_caller_tasks",
        "trace_q.read_run_journal_batched",
        "trace_q.read_run_meta_batched",
        "trace_q.read_session_journal_batched",
        "trace_q.read_session_meta_batched",
        "triggers_service.validate_trigger_spec",
        "validate_trigger_spec",
        "wf_queries.acquire_account_wf_runs_lock",
        "wf_queries.add_run_call_llm_cost_microusd",
        "wf_queries.append_run_event",
        "wf_queries.archive_run",
        "wf_queries.archive_workflow",
        "wf_queries.count_active_runs",
        "wf_queries.derive_run_response",
        "wf_queries.find_open_gate_call_key",
        "wf_queries.get_run_call_llm_cost_microusd",
        "wf_queries.get_run_depth",
        "wf_queries.get_run_for_step",
        "wf_queries.get_run_vault_ids",
        "wf_queries.get_wf_run",
        "wf_queries.get_workflow",
        "wf_queries.get_workflow_version",
        "wf_queries.insert_run_signal",
        "wf_queries.insert_wf_run",
        "wf_queries.insert_workflow",
        "wf_queries.list_run_events",
        "wf_queries.list_run_events_scoped",
        "wf_queries.list_run_ids_needing_step",
        "wf_queries.list_run_signals",
        "wf_queries.list_wf_runs",
        "wf_queries.list_workflow_versions",
        "wf_queries.list_workflows",
        "wf_queries.read_run_signal",
        "wf_queries.resolve_run_error",
        "wf_queries.run_children_usage",
        "wf_queries.runs_children_usage",
        "wf_queries.set_run_status",
        "wf_queries.set_run_terminal",
        "wf_queries.set_run_vaults",
        "wf_queries.unarchive_workflow",
        "wf_queries.unscoped_terminal_run_ids",
        "wf_queries.update_workflow",
        "write_gate_opened",
    }
)


class _Checker(ast.NodeVisitor):
    def __init__(self, filename: str, lines: list[str]) -> None:
        self.filename = filename
        self.lines = lines
        self.held: list[set[str]] = []
        self.violations: list[Violation] = []

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        acquired: set[str] = set()
        transaction_connections: set[str] = set()
        for item in node.items:
            expression = item.context_expr
            if not isinstance(expression, ast.Call) or not isinstance(
                expression.func, ast.Attribute
            ):
                continue
            if expression.func.attr == "acquire" and isinstance(item.optional_vars, ast.Name):
                acquired.add(item.optional_vars.id)
            elif expression.func.attr == "transaction":
                receiver = _expression_name(expression.func.value)
                if receiver is not None:
                    transaction_connections.add(receiver)

        names = acquired | transaction_connections
        if names:
            self.held.append(names)
            for statement in node.body:
                self.visit(statement)
            self.held.pop()
            for item in node.items:
                self.visit(item.context_expr)
            return
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        held = set().union(*self.held) if self.held else set()
        if held and not self._is_db_await(node.value, held) and not self._has_linked_pragma(node):
            connection = sorted(held)[0]
            self.violations.append(
                Violation(
                    self.filename,
                    node.lineno,
                    node.col_offset + 1,
                    f"pooled connection '{connection}' held across non-DB await",
                )
            )
        self.generic_visit(node)

    def _is_db_await(self, expression: ast.AST, held: set[str]) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        if isinstance(expression.func, ast.Attribute):
            receiver = _expression_name(expression.func.value)
            if receiver in held:
                return True
        symbol = _expression_name(expression.func)
        if symbol is None:
            return False
        if symbol not in _DB_HELPER_SYMBOLS:
            return False
        values = [*expression.args, *(keyword.value for keyword in expression.keywords)]
        return sum(_expression_name(value) in held for value in values) == 1

    def _has_linked_pragma(self, node: ast.Await) -> bool:
        line = self.lines[node.lineno - 1]
        return _PRAGMA in line and _ISSUE_REF.search(line) is not None


def check_source(source: str, *, filename: str = "<unknown>") -> list[Violation]:
    tree = ast.parse(source, filename=filename)
    checker = _Checker(filename, source.splitlines())
    checker.visit(tree)
    return checker.violations


def check_paths(paths: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            violations.extend(check_source(path.read_text(), filename=str(path)))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    violations = check_paths(args.paths)
    for violation in violations:
        print(violation)
    return bool(violations)


if __name__ == "__main__":
    sys.exit(main())
