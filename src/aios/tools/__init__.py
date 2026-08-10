"""Aios v1 built-in tools.

Importing this package triggers registration of every built-in tool
against the module-level :data:`aios.tools.registry.registry` singleton.
The worker imports this module once at startup (via ``aios.harness.worker``),
after which :func:`aios.tools.registry.to_openai_tools` can translate any
agent's ``tools`` list into a LiteLLM ``tools`` parameter.

Tools are deliberately minimal: ``bash`` is the workhorse, and ``read`` /
``write`` / ``edit`` exist only to give the model capabilities bash can't
cleanly offer (line-numbered structured reads, safe-arbitrary-content
writes via base64 stdin, strict find-and-replace with in-process diff).
No defensive guards, no heuristics, no model-specific shims — the model
sees raw tool errors and retries through the session log.
"""

from __future__ import annotations

# Side-effect imports: each module's top-level _register() call adds the
# tool to the registry singleton. Order doesn't matter semantically but
# matches the agent-tools declaration order for readability.
from aios.tools import agent_management as _agent_management  # noqa: F401
from aios.tools import bash as _bash  # noqa: F401
from aios.tools import ctx as _ctx  # noqa: F401
from aios.tools import defer_obligations as _defer_obligations  # noqa: F401
from aios.tools import edit as _edit  # noqa: F401
from aios.tools import glob as _glob  # noqa: F401
from aios.tools import goal_management as _goal_management  # noqa: F401
from aios.tools import grep as _grep  # noqa: F401
from aios.tools import http_request as _http_request  # noqa: F401
from aios.tools import invoke_session as _invoke_session  # noqa: F401
from aios.tools import list_obligations as _list_obligations  # noqa: F401
from aios.tools import list_related_sessions as _list_related_sessions  # noqa: F401
from aios.tools import memory_search as _memory_search  # noqa: F401
from aios.tools import read as _read  # noqa: F401
from aios.tools import rlm as _rlm  # noqa: F401
from aios.tools import schedule_wake as _schedule_wake  # noqa: F401
from aios.tools import search_events as _search_events  # noqa: F401
from aios.tools import skill_management as _skill_management  # noqa: F401
from aios.tools import switch_channel as _switch_channel  # noqa: F401
from aios.tools import tasks as _tasks  # noqa: F401
from aios.tools import trigger_account_list as _trigger_account_list  # noqa: F401
from aios.tools import trigger_create as _trigger_create  # noqa: F401
from aios.tools import trigger_list as _trigger_list  # noqa: F401
from aios.tools import trigger_remove as _trigger_remove  # noqa: F401
from aios.tools import trigger_update as _trigger_update  # noqa: F401
from aios.tools import wake_self as _wake_self  # noqa: F401
from aios.tools import wake_session as _wake_session  # noqa: F401
from aios.tools import web_fetch as _web_fetch  # noqa: F401
from aios.tools import web_search as _web_search  # noqa: F401
from aios.tools import workflow_completion as _workflow_completion  # noqa: F401
from aios.tools import workflow_management as _workflow_management  # noqa: F401
from aios.tools import write as _write  # noqa: F401
