"""Langfuse (self-hosted) tracing — the single integration point for the agent graph.

Everything funnels through here so the rest of the codebase never imports the
Langfuse SDK directly. When ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
absent (or the SDK isn't installed) every function degrades to a no-op, so the
agent runs unchanged without a Langfuse backend.

Three pieces cover the whole "every detail" requirement:
  - :func:`get_handler` — the LangChain ``CallbackHandler`` attached to the graph
    ``config["callbacks"]``; auto-traces every LLM call, tool, node, and the
    planner→subagent tree.
  - :func:`traced_run` — opens one root span per workflow run so the handler's
    spans *and* the manual spans below all roll up under a single trace; sets the
    Langfuse session id / tags and flushes on exit.
  - :func:`observe` / :func:`record_event` / :func:`record_output` — manual spans
    and events for the blind spots the callback bus can't see (raw HTTP, Mongo
    writes, conditional-edge routing decisions, middleware delegation).

All manual helpers are best-effort: they never raise into the agent run.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"


def is_enabled() -> bool:
    """True when Langfuse credentials are present in the environment."""
    return bool(os.getenv(_PUBLIC_KEY_ENV) and os.getenv(_SECRET_KEY_ENV))


@functools.lru_cache(maxsize=1)
def _client() -> Optional[Any]:
    """Cached Langfuse v4 client, or None when unconfigured / unavailable."""
    if not is_enabled():
        return None
    try:
        from langfuse import get_client

        client = get_client()
        try:
            if not client.auth_check():
                logger.warning("Langfuse auth_check failed; tracing disabled")
                return None
        except Exception:
            # auth_check can fail transiently (network); keep the client and let
            # individual operations degrade rather than disabling tracing outright.
            logger.debug("Langfuse auth_check raised; proceeding", exc_info=True)
        return client
    except Exception:
        logger.warning("Langfuse SDK import failed; tracing disabled", exc_info=True)
        return None


def client() -> Optional[Any]:
    """The cached Langfuse client, or None when unconfigured. Shared with prompt
    management (:mod:`infrastructure.observability.langfuse_prompts`)."""
    return _client()


@functools.lru_cache(maxsize=1)
def get_handler() -> Optional[Any]:
    """Cached LangChain ``CallbackHandler``, or None when Langfuse is unconfigured."""
    if _client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        logger.warning(
            "Could not build Langfuse CallbackHandler; tracing disabled", exc_info=True
        )
        return None


@contextmanager
def traced_run(
    session_id: str,
    *,
    name: str = "agent-run",
    tags: Optional[Sequence[str]] = None,
    **metadata: Any,
) -> Iterator[Optional[Any]]:
    """Open a root span for one workflow run; yield the ``CallbackHandler`` (or None).

    All nested LLM/tool spans (via the yielded handler) and manual ``@observe`` /
    :func:`record_event` spans auto-parent under this root via OTEL contextvars, so
    a single Langfuse trace captures the whole run. The Langfuse session id and tags
    are set on the trace so runs are groupable in the UI. Flushes on exit.

    Degrades to ``yield None`` (no span, no flush) when Langfuse is unconfigured.
    """
    client = _client()
    handler = get_handler()
    if client is None or handler is None:
        yield None
        return

    with client.start_as_current_observation(as_type="span", name=name):
        try:
            client.update_current_trace(
                session_id=session_id,
                tags=list(tags) if tags else None,
                metadata=metadata or None,
            )
        except Exception:
            logger.debug("update_current_trace failed", exc_info=True)
        try:
            yield handler
        finally:
            try:
                client.flush()
            except Exception:
                logger.debug("Langfuse flush failed", exc_info=True)


def record_output(output: Any) -> None:
    """Attach a final output payload to the current (root) span. Best-effort."""
    client = _client()
    if client is None:
        return
    try:
        client.update_current_span(output=output)
    except Exception:
        logger.debug("record_output failed", exc_info=True)


def record_event(name: str, **metadata: Any) -> None:
    """Emit a point-in-time event under the current trace. Best-effort.

    Used for things the callback bus never sees as runnables: baseline
    conditional-edge routing decisions and middleware delegation choices.
    """
    client = _client()
    if client is None:
        return
    try:
        client.create_event(name=name, metadata=metadata or None)
    except Exception:
        logger.debug("record_event failed", exc_info=True)


def observe(func: Optional[Callable] = None, **kwargs: Any) -> Callable:
    """``@observe`` decorator that becomes a plain pass-through when Langfuse is off.

    Supports both ``@observe`` and ``@observe(name=..., as_type=...)``. When Langfuse
    is unconfigured the wrapped function is returned unchanged (no import, no span).
    """

    def deco(fn: Callable) -> Callable:
        if not is_enabled():
            return fn
        try:
            from langfuse import observe as _lf_observe
        except Exception:
            return fn
        return _lf_observe(**kwargs)(fn)

    return deco(func) if callable(func) else deco
