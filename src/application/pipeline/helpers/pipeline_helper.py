"""YAML loading and agent config builders for the pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from core.agents.directed_generative_agent import DirectedGenerativeAgent
from core.agents.generative_agent import GenerativeAgent
from core.agents.reasoning_agent import ReasoningAgent
from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters
from domain.models.langgraph.agents_models import AgentConfig
from domain.models.structured_outputs.generative_agent_structured_output import GenerativeAgentResponse
from domain.models.structured_outputs.reasoning_agent_structured_output import ReasoningAgentResponse

logger = logging.getLogger(__name__)


AGENT_CLASS_REGISTRY: Dict[str, type] = {
    "GenerativeAgent": GenerativeAgent,
    "DirectedGenerativeAgent": DirectedGenerativeAgent,
    "ReasoningAgent": ReasoningAgent,
}

STRUCTURED_OUTPUT_REGISTRY: Dict[str, type] = {
    "GenerativeAgentResponse": GenerativeAgentResponse,
    "ReasoningAgentResponse": ReasoningAgentResponse,
}

REQUIRED_AGENT_YAML_SECTIONS = (
    "Agent_Config",
    "Model_Params",
    "Pipeline_Params",
    "Generation_Config",
)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping in file: {path}")

    return data


def parse_registry(data: Dict[str, Any]) -> List[Dict[str, str]]:
    agents = data.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Registry YAML must contain an 'agents' list")

    normalized: List[Dict[str, str]] = []
    seen_names: Set[str] = set()

    for entry in agents:
        if not isinstance(entry, dict):
            raise ValueError("Each registry agent entry must be a mapping")

        name = entry.get("name")
        config_path = entry.get("config_path")

        if not name or not isinstance(name, str):
            raise ValueError("Each registry entry must include a string 'name'")
        if not config_path or not isinstance(config_path, str):
            raise ValueError(
                f"Registry entry '{name}' must include a string 'config_path'"
            )
        if name in seen_names:
            raise ValueError(f"Duplicate agent name in registry: {name}")

        normalized.append({"name": name, "config_path": config_path})
        seen_names.add(name)

    return normalized


def resolve_config_path(registry_path: Path, config_path: str) -> Path:
    raw_path = Path(config_path)
    if raw_path.is_absolute():
        return raw_path

    if raw_path.exists():
        return raw_path.resolve()

    registry_dir = registry_path.resolve().parent
    return (registry_dir / raw_path).resolve()


def load_agent_yaml(config_path: Path) -> Dict[str, Any]:
    data = load_yaml_file(config_path)
    missing = [section for section in REQUIRED_AGENT_YAML_SECTIONS if section not in data]
    if missing:
        raise ValueError(
            f"Agent config '{config_path}' missing required sections: {missing}"
        )
    return data


def resolve_pipeline_registry_yaml(registry_dir: Path, registry_id: str) -> Path:
    """
    Resolve a pipeline registry file under ``registry_dir`` (typically ``config/pipeline``).

    ``registry_id`` is the YAML stem (e.g. ``default_pipeline_registry``) or may include ``.yaml``.
    """
    stem = registry_id.strip()
    if stem.lower().endswith(".yaml"):
        stem = stem[:-5]
    if not stem:
        raise ValueError("pipeline registry id is empty")
    candidate = (registry_dir / f"{stem}.yaml").resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Pipeline registry YAML not found: {candidate}")


def list_pipeline_registry_ids(registry_path: Path) -> List[Dict[str, Any]]:
    """
    List ``*_pipeline_registry.yaml`` files in the same directory as ``registry_path``.

    Each file lists agent entries; graph topology is implemented in ``langraph_model.py``.
    """
    d = registry_path.resolve().parent
    if not d.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(d.glob("*_pipeline_registry.yaml")):
        meta: Dict[str, Any] = {"id": p.stem, "path": str(p.resolve())}
        try:
            data = load_yaml_file(p) or {}
            name = data.get("name")
            if isinstance(name, str):
                meta["name"] = name
        except Exception as exc:
            logger.debug("Skip pipeline registry metadata for %s: %s", p, exc)
        out.append(meta)
    return out


def build_agent_config(raw: Dict[str, Any]) -> AgentConfig:
    structured_output_key = raw["structured_output_key"]
    agent_class_key = raw["agent_class_key"]

    if structured_output_key not in STRUCTURED_OUTPUT_REGISTRY:
        raise ValueError(f"Unknown structured_output_key: {structured_output_key}")
    if agent_class_key not in AGENT_CLASS_REGISTRY:
        raise ValueError(f"Unknown agent_class_key: {agent_class_key}")

    return AgentConfig(
        name=raw["name"],
        prompt_key=raw["prompt_key"],
        structured_output=STRUCTURED_OUTPUT_REGISTRY[structured_output_key],
        agent_cls=AGENT_CLASS_REGISTRY[agent_class_key],
        share_history_key=raw.get("share_history_key"),
        bind_max_tokens=raw.get("bind_max_tokens", True),
        base_path=(raw.get("base_path") or "").strip() or "config",
        extra_prompt_kwargs=raw.get("extra_prompt_kwargs", {}),
    )


def resolve_model_path(model_name: str) -> str:
    if model_name == "WhiteRabbit-LLama":
        return "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B"
    if model_name == "WhiteRabbit-Qwen":
        return "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    if model_name == "LLama":
        return "meta-llama/Meta-Llama-3-8B"
    return model_name


def build_model_params(raw: Dict[str, Any]) -> LoadModelParameters:
    model_name = raw["model_name"]
    model_path = raw.get("model_path") or resolve_model_path(model_name)
    return LoadModelParameters(
        model_name=model_name,
        model_path=model_path,
        bit_quantization=raw.get("bit_quantization", None),
    )


def build_pipeline_params(
    model_params: LoadModelParameters,
    raw: Dict[str, Any],
) -> PipelineParameters:
    return PipelineParameters(
        model_name=model_params.model_name,
        task_type=raw["task_type"],
        max_new_tokens=raw["max_new_tokens"],
        do_sample=raw["do_sample"],
        temperature=raw["temperature"],
        top_p=raw["top_p"],
        top_k=raw["top_k"],
        repetition_penalty=raw["repetition_penalty"],
        presence_penalty=raw["presence_penalty"],
        frequency_penalty=raw["frequency_penalty"],
        no_repeat_ngram_size=raw["no_repeat_ngram_size"],
        load_playbooks=raw.get("load_playbooks", True),
        load_snippets=raw.get("load_snippets", True),
        load_context=raw.get("load_context", True),
    )


def validate_generation_config_alignment(
    pipeline_params: PipelineParameters,
    generation_config: Dict[str, Any],
) -> None:
    comparable_keys = {
        "temperature": pipeline_params.temperature,
        "top_p": pipeline_params.top_p,
        "top_k": pipeline_params.top_k,
        "repetition_penalty": pipeline_params.repetition_penalty,
        "presence_penalty": pipeline_params.presence_penalty,
        "frequency_penalty": pipeline_params.frequency_penalty,
        "no_repeat_ngram_size": pipeline_params.no_repeat_ngram_size,
        "max_tokens": pipeline_params.max_new_tokens,
    }

    mismatches = []
    for key, pipeline_value in comparable_keys.items():
        generation_value = generation_config.get(key)
        if generation_value != pipeline_value:
            mismatches.append(
                f"{key}: pipeline={pipeline_value!r}, generation={generation_value!r}"
            )

    if mismatches:
        raise ValueError(
            "Generation_Config and Pipeline_Params are misaligned: "
            + "; ".join(mismatches)
        )


def build_generation_config(
    raw: Dict[str, Any],
    pipeline_params: PipelineParameters,
) -> Dict[str, Any]:
    generation_config = {
        "temperature": raw["temperature"],
        "top_p": raw["top_p"],
        "top_k": raw["top_k"],
        "max_tokens": raw["max_tokens"],
        "repetition_penalty": raw["repetition_penalty"],
        "presence_penalty": raw["presence_penalty"],
        "frequency_penalty": raw["frequency_penalty"],
        "no_repeat_ngram_size": raw["no_repeat_ngram_size"],
    }
    validate_generation_config_alignment(pipeline_params, generation_config)
    return generation_config
