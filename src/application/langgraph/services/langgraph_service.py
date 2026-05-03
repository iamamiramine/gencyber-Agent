from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from langchain_community.chat_message_histories import ChatMessageHistory

from application.langgraph.helpers.langraph_helpers import (
    build_chain,
    build_prompt_template,
    create_llm,
    load_agent_prompt,
    read_shell_context,
)
from application.langgraph.models.langraph_model import WorkflowGraph
from core.helpers.chat_history_helper import ChatHistoryFormatter
from core.tools.script_execution_tool import ExecuteScriptTool
from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters
from domain.models.langgraph.agents_models import AgentRuntime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _json_safe(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively convert objects to JSON-serializable structures for NDJSON streams."""
    if _depth > 14:
        return "<max depth>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= 20000 else obj[:20000] + "…(truncated)"
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 200:
                out["…"] = f"{len(obj) - i} more keys"
                break
            out[str(k)] = _json_safe(v, _depth=_depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, _depth=_depth + 1) for x in obj[:500]]
    return str(obj)[:12000]


def _merge_astream_v2_event(event: Any, merged: Dict[str, Any]) -> None:
    """Pull partial graph channel updates from LangGraph astream_events (v2) on_chain_end."""
    if not isinstance(event, dict):
        return
    if event.get("event") != "on_chain_end":
        return
    data = event.get("data")
    if not isinstance(data, dict):
        return
    out = data.get("output")
    if hasattr(out, "model_dump"):
        try:
            merged.update(out.model_dump())
            return
        except Exception:
            pass
    if isinstance(out, dict):
        merged.update(out)


def _merge_stream_updates_chunk(chunk: Any, merged: Dict[str, Any]) -> None:
    """Merge graph.stream(stream_mode='updates') chunks into cumulative state."""
    if isinstance(chunk, dict):
        for _node, partial in chunk.items():
            if isinstance(partial, dict):
                merged.update(partial)


class LangGraphService:
    """
    Params-only service.
    Expected input should come from PipelineConfigService.load_runtime_config().
    """

    def __init__(self) -> None:
        self.workflow: Optional[WorkflowGraph] = None
        self.workflow_initialized: bool = False
        self.session_id: Optional[str] = None

        self.formatter = ChatHistoryFormatter()

        self.agent_definitions: Dict[str, Any] = {}
        self.model_params: Dict[str, LoadModelParameters] = {}
        self.pipeline_params: Dict[str, PipelineParameters] = {}
        self.generation_configs: Dict[str, Dict[str, Any]] = {}
        self.model_config_raw: Dict[str, Dict[str, Any]] = {}
        self.histories: Dict[str, ChatMessageHistory] = {}
        self.agents: Dict[str, AgentRuntime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_workflow(
        self,
        session_id: str = "default",
        agent_definitions: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, LoadModelParameters]] = None,
        pipeline_params: Optional[Dict[str, PipelineParameters]] = None,
        generation_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        model_config_raw: Optional[Dict[str, Dict[str, Any]]] = None,
        history_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        try:
            if agent_definitions is None:
                return {"error": "agent_definitions is required"}
            if model_params is None:
                return {"error": "model_params is required"}
            if pipeline_params is None:
                return {"error": "pipeline_params is required"}
            if generation_configs is None:
                return {"error": "generation_configs is required"}
            if model_config_raw is None:
                return {"error": "model_config_raw is required"}

            self.session_id = session_id
            self._reset_agent_runtimes()

            self.agent_definitions = dict(agent_definitions)
            self.model_params = dict(model_params)
            self.pipeline_params = dict(pipeline_params)
            self.generation_configs = dict(generation_configs)
            self.model_config_raw = dict(model_config_raw)

            self._initialize_histories(history_keys or [])
            self._initialize_agent_runtimes()
            self._load_all_agents()
            self._build_workflow()

            self.workflow_initialized = True
            return {
                "message": "Workflow initialized successfully",
            }
        except Exception as e:
            logger.exception("Error initializing workflow")
            return {"error": f"Failed to initialize workflow: {str(e)}"}

    def run_workflow(
        self,
        question: str,
        leaderboard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.workflow_initialized or self.workflow is None:
                return {"error": "Workflow not initialized. Please call init_workflow first"}

            logger.info(
                "Running agent graph with query=%s session_id=%s",
                question,
                self.session_id,
            )
            result = self.workflow.invoke(query=question)
            api = self._state_to_api_response(result)
            if (
                leaderboard
                and "error" not in api
                and isinstance(result, dict)
            ):
                try:
                    from infrastructure.leaderboard_client import record_leaderboard_run

                    record_leaderboard_run(leaderboard, api, result)
                except Exception as e:
                    logger.warning("Leaderboard record failed: %s", e)
            return api
        except Exception as e:
            logger.exception("Error running workflow")
            return {"error": f"Failed to run workflow: {str(e)}"}

    def _state_to_api_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {
                "llm_output": "",
                "submitted_goal": None,
                "should_stop": False,
                "session_id": self.session_id,
            }
        if state.get("error"):
            return {
                "error": str(state.get("error")),
                "llm_output": state.get("generated_response", ""),
                "session_id": self.session_id,
            }
        return {
            "llm_output": state.get("generative_agent_response", ""),
            "submitted_goal": state.get("submitted_goal"),
            "should_stop": state.get("should_stop", False),
            "session_id": self.session_id,
        }

    def _iter_stream_updates_ndjson(
        self,
        question: str,
        leaderboard: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Fallback: sync graph.stream(stream_mode='updates') as NDJSON lines (type=step).
        """
        if not self.workflow_initialized or self.workflow is None:
            yield json.dumps({"type": "error", "message": "Workflow not initialized. Please call init_workflow first."}) + "\n"
            return

        wf = self.workflow
        state_dict = wf._build_initial_state(question)
        config = {
            "configurable": {"thread_id": wf.session_id},
            "recursion_limit": wf.recursion_limit,
        }
        graph = wf.graph
        merged: Dict[str, Any] = dict(state_dict)

        try:
            stream_iter = graph.stream(state_dict, config, stream_mode="updates")
        except Exception as e:
            logger.warning("stream(stream_mode=updates) failed (%s); trying default stream", e)
            try:
                stream_iter = graph.stream(state_dict, config)
            except Exception as e2:
                yield json.dumps({"type": "error", "message": f"Cannot stream workflow: {e2!s}"}) + "\n"
                return

        seq = 0
        try:
            for chunk in stream_iter:
                seq += 1
                _merge_stream_updates_chunk(chunk, merged)
                yield json.dumps(
                    {
                        "type": "step",
                        "seq": seq,
                        "chunk": _json_safe(chunk),
                        "state": _json_safe(merged),
                    }
                ) + "\n"
        except Exception as e:
            logger.exception("workflow stream failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return

        merged = self._finalize_merged_from_graph(graph, config, merged)
        api = self._state_to_api_response(merged)
        if leaderboard and "error" not in api and isinstance(merged, dict):
            try:
                from infrastructure.leaderboard_client import record_leaderboard_run

                record_leaderboard_run(leaderboard, api, merged)
            except Exception as e:
                logger.warning("Leaderboard record failed: %s", e)
        yield json.dumps({"type": "done", "result": api}) + "\n"

    def _finalize_merged_from_graph(
        self,
        graph: Any,
        config: Dict[str, Any],
        merged: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            snap = graph.get_state(config)
            if snap is not None and getattr(snap, "values", None) is not None:
                return dict(snap.values)
        except Exception as e:
            logger.warning("get_state after stream failed: %s", e)
        return merged

    async def astream_workflow_ndjson(
        self,
        question: str,
        leaderboard: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream LangGraph execution as NDJSON:
        - Prefer graph.astream_events(..., version='v2') for rich tracing
        - Fall back to graph.stream(stream_mode='updates')
        Final line is always {\"type\":\"done\",\"result\":{...}} or {\"type\":\"error\",...}.
        """
        if not self.workflow_initialized or self.workflow is None:
            yield json.dumps({"type": "error", "message": "Workflow not initialized. Please call init_workflow first."}) + "\n"
            return

        wf = self.workflow
        state_dict = wf._build_initial_state(question)
        config = {
            "configurable": {"thread_id": wf.session_id},
            "recursion_limit": wf.recursion_limit,
        }
        graph = wf.graph
        merged: Dict[str, Any] = dict(state_dict)

        astream_events_fn = getattr(graph, "astream_events", None)
        if astream_events_fn is None:
            for line in self._iter_stream_updates_ndjson(question, leaderboard=leaderboard):
                yield line
            return

        try:
            try:
                async for event in astream_events_fn(state_dict, config, version="v2"):
                    _merge_astream_v2_event(event, merged)
                    yield json.dumps(
                        {
                            "type": "event",
                            "payload": _json_safe(event),
                            "state": _json_safe(merged),
                        }
                    ) + "\n"
            except TypeError:
                async for event in astream_events_fn(state_dict, config):
                    _merge_astream_v2_event(event, merged)
                    yield json.dumps(
                        {
                            "type": "event",
                            "payload": _json_safe(event),
                            "state": _json_safe(merged),
                        }
                    ) + "\n"
        except Exception as e:
            # Do not fall back to graph.stream here — that would run the workflow a second time.
            logger.exception("astream_events failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return

        merged = await asyncio.to_thread(self._finalize_merged_from_graph, graph, config, merged)
        api = self._state_to_api_response(merged)
        if leaderboard and "error" not in api and isinstance(merged, dict):
            try:
                from infrastructure.leaderboard_client import record_leaderboard_run

                await asyncio.to_thread(record_leaderboard_run, leaderboard, api, merged)
            except Exception as e:
                logger.warning("Leaderboard record failed: %s", e)
        yield json.dumps({"type": "done", "result": api}) + "\n"

    def get_workflow_status(self) -> Dict[str, Any]:
        try:
            return {
                "initialized": self.workflow_initialized,
                "session_id": self.session_id,
                "workflow_loaded": self.workflow is not None,
                "agents": {
                    name: {
                        "model_loaded": runtime.model is not None,
                        "chain_loaded": runtime.chain is not None,
                        "agent_loaded": runtime.agent is not None,
                        "history_scope": runtime.config.share_history_key or name,
                        "model_name": getattr(self.model_params.get(name), "model_name", None),
                    }
                    for name, runtime in self.agents.items()
                },
            }
        except Exception as e:
            logger.exception("Error getting workflow status")
            return {"error": f"Failed to get workflow status: {str(e)}"}

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _reset_agent_runtimes(self) -> None:
        self.workflow = None
        self.workflow_initialized = False
        self.histories = {}
        self.agents = {}

    def _required_agent_refs(self) -> set[str]:
        return {"pm", "recon", "reasoning", "generative"}

    def _initialize_histories(self, history_keys: list[str]) -> None:
        active = self._required_agent_refs()
        keys: set[str] = set()
        for name in active:
            cfg = self.agent_definitions[name]
            keys.add(cfg.share_history_key or name)
        self.histories = {k: ChatMessageHistory() for k in keys}

    def _initialize_agent_runtimes(self) -> None:
        active = self._required_agent_refs()
        missing = active - set(self.agent_definitions.keys())
        if missing:
            raise ValueError(
                f"Pipeline registry must define agents {sorted(missing)} (pm, recon, reasoning, generative)"
            )
        self.agents = {
            name: AgentRuntime(config=self.agent_definitions[name])
            for name in active
        }

    def _load_all_agents(self) -> None:
        shell_context = read_shell_context()

        for name, runtime in self.agents.items():
            model_params = self.model_params[name]
            pipeline_params = self.pipeline_params[name]
            generation_config = dict(self.generation_configs[name])
            generation_config.setdefault("model_name", model_params.model_name)
            raw_model_config = self.model_config_raw[name]

            prompt = build_prompt_template(model_params.model_name)

            runtime.model = create_llm(
                model_params=model_params,
                pipeline_params=pipeline_params,
                model_config_raw=raw_model_config,
            )
            runtime.system_prompt = load_agent_prompt(
                runtime.config,
                pipeline_params,
                shell_context,
            )
            runtime.chat_history = self._resolve_chat_history(runtime.config)
            runtime.chain = build_chain(
                prompt=prompt,
                llm=runtime.model,
                structured_output=runtime.config.structured_output,
                max_tokens=generation_config["max_tokens"],
                bind_max_tokens=runtime.config.bind_max_tokens,
            )
            runtime.agent = runtime.config.agent_cls(
                runtime.chain,
                generation_config,
                runtime.chat_history,
                runtime.system_prompt,
                self.formatter,
                model_params=model_params,
                pipeline_params=pipeline_params,
                model_config_raw=raw_model_config,
                node_id=runtime.config.name,
            )

    def _build_workflow(self) -> None:
        if self.session_id is None:
            raise ValueError("session_id must be set before building workflow")

        self.workflow = WorkflowGraph(
            session_id=self.session_id,
            pm_agent=self.agents["pm"].agent,
            recon_agent=self.agents["recon"].agent,
            reasoning_agent=self.agents["reasoning"].agent,
            generative_agent=self.agents["generative"].agent,
            execute_script_tool=ExecuteScriptTool(session_id=self.session_id),
        )

    def _resolve_chat_history(self, config) -> ChatMessageHistory:
        history_key = config.share_history_key or config.name
        return self.histories[history_key]
