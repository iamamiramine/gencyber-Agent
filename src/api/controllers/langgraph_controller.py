from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional

import logging
import os
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from application.pipeline.helpers import pipeline_helper as ph
from application.pipeline.services.pipeline_service import PipelineConfigService
from application.langgraph.services.langgraph_service import LangGraphService
from infrastructure.repository.mongodb_repository import get_readable_checkpoint
from infrastructure.repository.mongodb_repository import get_mongodb_client
from langgraph.checkpoint.mongodb import MongoDBSaver

_DEFAULT_REGISTRY = "config/pipeline/default_pipeline_registry.yaml"
config_service = PipelineConfigService(registry_path=os.getenv("REGISTRY_PATH", _DEFAULT_REGISTRY))
runtime_config = config_service.load_runtime_config()

langgraph_service = LangGraphService()


logger = logging.getLogger(__name__)
router = APIRouter()


class WorkflowInitParams(BaseModel):
    """Parameters for initializing the LangGraph agent workflow from a pipeline registry."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = "default"
    pipeline_registry_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "pipeline_registry_id",
            "agentic_workflow_id",
        ),
        description=(
            "Pipeline registry YAML stem next to REGISTRY_PATH (e.g. 'default_pipeline_registry'). "
        ),
    )
    # Optional per-request model override applied to every agent in the pipeline.
    model_name: Optional[str] = Field(
        default=None,
        description="Override model for all agents (e.g. 'openai/gpt-4o-mini', 'WhiteRabbit-Qwen', 'LLama').",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Optional explicit model path; defaults to resolve_model_path(model_name).",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional provider override ('openrouter', 'openai', or 'ollama'). Inferred from model_name if omitted.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL (required for Ollama / vLLM / OpenAI-compatible models).",
    )
    # Per-run escalation ladder controls. The escalation target is chosen by the
    # operator here (not by the agent) and overrides GENCYBER_ESCALATION_* env
    # defaults for THIS run only. Omit any field to inherit the env default.
    escalation_model: Optional[str] = Field(
        default=None,
        description=(
            "Strong model this run escalates to when a specialist stalls / submits a "
            "bad flag (e.g. 'openai/gpt-5-mini', 'deepseek/deepseek-chat', a vLLM served "
            "name). 'off'/'none' disables escalation for this run. Omit to use the "
            "GENCYBER_ESCALATION_MODEL env default."
        ),
    )
    escalation_after_actions: Optional[int] = Field(
        default=None,
        description="Actions-without-a-flag that trip a stall escalation (env default 12).",
    )
    escalation_provider: Optional[str] = Field(
        default=None,
        description="Optional provider for the escalation model (else inherits the base model's).",
    )
    escalation_base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL for the escalation model (else inherits the base model's).",
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description=(
            "Explicit tool ids the UI attaches to this workflow. The first id naming "
            "a registered submit tool (e.g. 'nyuctf_submit_goal', 'baseline_submit_goal') "
            "selects the submit-goal variant via ``SUBMIT_TOOL_REGISTRY``. When omitted "
            "or none match, ``BaselineSubmitGoalTool`` is attached: a genuinely-recovered "
            "value is accepted, a value with no supporting evidence is rejected. The "
            "core execute_script / write_script tools are always attached regardless."
        ),
    )


def _apply_model_override(
    rt: Dict[str, Any],
    *,
    model_name: str,
    model_path: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a copy of a runtime config with every agent's model fields overridden.

    Does not mutate the input ``rt`` (which may be the cached module-level config).
    """
    resolved_path = model_path or ph.resolve_model_path(model_name)

    new_model_params: Dict[str, Any] = {}
    new_model_config_raw: Dict[str, Dict[str, Any]] = {}
    new_pipeline_params: Dict[str, Any] = {}

    for agent_name, raw in rt["model_config_raw"].items():
        raw_copy = dict(raw)
        raw_copy["model_name"] = model_name
        raw_copy["model_path"] = resolved_path
        if provider is not None:
            raw_copy["provider"] = provider
        if base_url is not None:
            raw_copy["base_url"] = base_url
        new_model_config_raw[agent_name] = raw_copy
        new_model_params[agent_name] = ph.build_model_params(raw_copy)

    for agent_name, pp in rt["pipeline_params"].items():
        try:
            new_pipeline_params[agent_name] = pp.model_copy(update={"model_name": model_name})
        except Exception:
            new_pipeline_params[agent_name] = pp

    merged = dict(rt)
    merged["model_params"] = new_model_params
    merged["model_config_raw"] = new_model_config_raw
    merged["pipeline_params"] = new_pipeline_params
    return merged


class RunWorkflowParams(BaseModel):
    """Parameters for running the LangGraph workflow."""

    question: str

@router.get("/pipeline_catalog")
def get_pipeline_catalog() -> Dict[str, Any]:
    """
    UI catalog: **pipeline registries** (``*_pipeline_registry.yaml`` next to the active registry file).
    """
    try:
        pipeline_registries = config_service.list_pipeline_registries()
        return {
            "pipeline_registries": pipeline_registries,
            "active_pipeline_registry": config_service.registry_path.stem,
            "registry_path": str(config_service.registry_path),
        }
    except Exception as e:
        logger.exception("Failed to list pipeline catalog")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/graph_topology")
def get_graph_topology(pipeline_registry_id: Optional[str] = None) -> Dict[str, Any]:
    """Serialize the compiled LangGraph topology so the UI can render any workflow.

    Resolves the optional ``graph:`` key from the selected pipeline registry
    (defaults to the active registry), then serializes that builder's real
    compiled graph. Side-effect-free — no MongoDB, no initialized workflow.
    """
    try:
        reg_file = Path(config_service.registry_path)
        if pipeline_registry_id:
            reg_dir = Path(config_service.registry_path).resolve().parent
            reg_file = ph.resolve_pipeline_registry_yaml(reg_dir, pipeline_registry_id)

        graph_key = ph.load_yaml_file(Path(reg_file)).get("graph")
        topology = langgraph_service.get_graph_topology(graph_key)
        topology["registry_id"] = Path(reg_file).stem
        return topology
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to serialize graph topology")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/init_workflow")
def init_workflow(workflow_init_params: Optional[WorkflowInitParams] = None) -> dict:
    """
    Initialize the LangGraph **agent workflow**
    """
    global config_service, runtime_config
    params = workflow_init_params or WorkflowInitParams()
    try:
        loaded_service = config_service
        rt = runtime_config
        if params.pipeline_registry_id:
            reg_dir = Path(config_service.registry_path).resolve().parent
            reg_file = ph.resolve_pipeline_registry_yaml(reg_dir, params.pipeline_registry_id)
            loaded_service = PipelineConfigService(registry_path=str(reg_file))
            rt = loaded_service.load_runtime_config()

        if params.model_name:
            rt = _apply_model_override(
                rt,
                model_name=params.model_name,
                model_path=params.model_path,
                provider=params.provider,
                base_url=params.base_url,
            )

        response = langgraph_service.init_workflow(
            session_id=params.session_id,
            agent_definitions=rt["agent_definitions"],
            model_params=rt["model_params"],
            pipeline_params=rt["pipeline_params"],
            generation_configs=rt["generation_configs"],
            model_config_raw=rt["model_config_raw"],
            history_keys=rt["history_keys"],
            tools=params.tools,
            graph_key=rt.get("graph"),
            escalation_model=params.escalation_model,
            escalation_after_actions=params.escalation_after_actions,
            escalation_provider=params.escalation_provider,
            escalation_base_url=params.escalation_base_url,
        )
    except Exception as e:
        logger.exception("Unexpected controller error during workflow initialization")
        raise HTTPException(status_code=400, detail=f"Failed to initialize workflow: {str(e)}")

    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])

    from core.helpers.terminal_session_client import ensure_terminal_session

    term = ensure_terminal_session(params.session_id)
    if not term.get("success"):
        logger.warning(
            "Workbench terminal session not provisioned for %s: %s",
            params.session_id,
            term.get("error"),
        )
    else:
        response["terminal_session"] = {
            "sessionId": term.get("sessionId", params.session_id),
            "ready": True,
        }

    if params.pipeline_registry_id:
        config_service = loaded_service
        runtime_config = rt

    return response


@router.post("/run_workflow_stream")
async def run_workflow_stream(run_params: RunWorkflowParams):
    """
    Stream LangGraph execution as newline-delimited JSON (NDJSON).

    Each line is a JSON object. Typical ``type`` values: ``event`` (from ``astream_events``),
    ``step`` (fallback ``graph.stream`` updates), ``done`` (final API-shaped result),
    ``error``.
    """

    async def ndjson_body():
        async for line in langgraph_service.astream_workflow_ndjson(
            run_params.question,
        ):
            yield line.encode("utf-8") if isinstance(line, str) else line

    return StreamingResponse(ndjson_body(), media_type="application/x-ndjson")


@router.post("/run_workflow")
def run_workflow(run_params: RunWorkflowParams) -> Dict[str, Any]:
    """
    Run the LangGraph workflow with the provided parameters.
    """
    logger.info("Running workflow with question=%s", run_params.question)

    try:
        response = langgraph_service.run_workflow(
            question=run_params.question,
        )
    except Exception as e:
        logger.exception("Unexpected controller error during workflow execution")
        raise HTTPException(status_code=400, detail=f"Failed to run workflow: {str(e)}")

    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.get("/workflow_status")
def get_workflow_status() -> dict:
    """
    Get the current status of the LangGraph workflow.
    """
    try:
        response = langgraph_service.get_workflow_status()
    except Exception as e:
        logger.exception("Unexpected controller error while fetching workflow status")
        raise HTTPException(status_code=400, detail=f"Failed to get workflow status: {str(e)}")

    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])
    return response


@router.get("/latest_checkpoint")
def latest_checkpoint(session_id: str) -> Dict[str, Any]:
    """
    Fetch the latest LangGraph checkpoint from MongoDB for the given session/thread.

    This is intended for UI debugging/visualization while a workflow is running.
    """
    checkpoint = None

    # Prefer reading via MongoDBSaver so checkpoint data is properly deserialized.
    try:
        mongo_client = get_mongodb_client()
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        saver = MongoDBSaver(mongo_client, db_name=db_name)
        config = {"configurable": {"thread_id": session_id}}

        tuple_obj = None
        if hasattr(saver, "get_tuple"):
            tuple_obj = saver.get_tuple(config)
        elif hasattr(saver, "get"):
            # Some versions expose get() returning a checkpoint dict directly
            tuple_obj = saver.get(config)

        if tuple_obj:
            # CheckpointTuple has .checkpoint, .config, .metadata in newer LangGraph
            if hasattr(tuple_obj, "checkpoint"):
                cp = tuple_obj.checkpoint or {}
                state_data = (
                    cp.get("channel_values")
                    or cp.get("values")
                    or cp.get("state")
                    or cp.get("data")
                    or {}
                )
                checkpoint = {
                    "thread_id": session_id,
                    "timestamp": (tuple_obj.metadata or {}).get("ts") if hasattr(tuple_obj, "metadata") else None,
                    "session_id": session_id,
                    "query": state_data.get("query") if isinstance(state_data, dict) else None,
                    "created_at": None,
                    "state_data": state_data if isinstance(state_data, dict) else {},
                    "schema": "mongodb_saver",
                }
            elif isinstance(tuple_obj, dict):
                # If saver.get() returned a dict
                cp = tuple_obj or {}
                state_data = (
                    cp.get("channel_values")
                    or cp.get("values")
                    or cp.get("state")
                    or cp.get("data")
                    or {}
                )
                checkpoint = {
                    "thread_id": session_id,
                    "timestamp": cp.get("ts") or cp.get("timestamp"),
                    "session_id": session_id,
                    "query": state_data.get("query") if isinstance(state_data, dict) else None,
                    "created_at": None,
                    "state_data": state_data if isinstance(state_data, dict) else {},
                    "schema": "mongodb_saver_dict",
                }
    except Exception as e:
        logger.warning("MongoDBSaver read failed for session_id=%s: %s", session_id, e)

    # Fallback: best-effort direct MongoDB read helper
    if checkpoint is None:
        try:
            checkpoint = get_readable_checkpoint(thread_id=session_id)
        except Exception as e:
            logger.exception("Failed to read latest checkpoint for session_id=%s", session_id)
            raise HTTPException(status_code=400, detail=f"Failed to read checkpoint: {str(e)}")

    if not checkpoint:
        return {
            "found": False,
            "session_id": session_id,
            "state": {},
        }

    # Full merged graph state for UI trajectory / debug (was dropped before, so Streamlit always saw {}).
    state_payload = checkpoint.get("state_data")
    if not isinstance(state_payload, dict):
        state_payload = {}

    return {
        "found": True,
        "thread_id": checkpoint.get("thread_id"),
        "session_id": checkpoint.get("session_id", session_id),
        "timestamp": checkpoint.get("timestamp"),
        "created_at": checkpoint.get("created_at"),
        "query": checkpoint.get("query"),
        "state": state_payload,
        "checkpoint_schema": checkpoint.get("schema"),
    }
