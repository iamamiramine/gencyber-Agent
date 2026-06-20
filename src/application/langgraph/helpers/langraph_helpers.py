from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Type

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from application.langgraph.helpers.system_prompt_helper import load_system_prompt_for_agent
from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters


logger = logging.getLogger(__name__)

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_runtime_model_name(params: LoadModelParameters) -> str:
    if params.model_name == "LLama":
        return "llama3"
    return params.model_path or params.model_name


def _normalize_model_name_for_checks(model_name: str) -> str:
    """Strip OpenRouter-style provider prefixes (e.g. openai/gpt-4o-mini)."""
    name = model_name.lower().strip()
    if "/" in name:
        return name.split("/", 1)[1]
    return name


def _openai_allows_sampling_kwargs(model_name: str) -> bool:
    """
    Newer OpenAI model families (gpt-5, o-series) reject top_p and penalty kwargs.

    See: https://platform.openai.com/docs/guides/reasoning — restricted parameter sets.
    """
    name = _normalize_model_name_for_checks(model_name)
    restricted_prefixes = ("gpt-5", "o1", "o3", "o4")
    return not any(name.startswith(prefix) for prefix in restricted_prefixes)


def _build_chat_openai_kwargs(
    model_params: LoadModelParameters,
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    top_p: Optional[float],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
) -> Dict[str, Any]:
    model_kwargs: Dict[str, Any] = {}
    if _openai_allows_sampling_kwargs(model_params.model_name):
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            model_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = frequency_penalty

    kwargs: Dict[str, Any] = {
        "model": model_params.model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return kwargs


def create_llm(
    model_params: LoadModelParameters,
    pipeline_params: PipelineParameters,
    model_config_raw: Dict[str, Any],
) -> Any:
    """
    Params-only helper.
    No YAML loading, no registry handling, no config discovery.
    """
    provider = str(model_config_raw.get("provider", "")).lower().strip()
    temperature = pipeline_params.temperature
    max_tokens = pipeline_params.max_new_tokens
    top_p = pipeline_params.top_p
    top_k = pipeline_params.top_k
    repetition_penalty = pipeline_params.repetition_penalty
    presence_penalty = pipeline_params.presence_penalty
    frequency_penalty = pipeline_params.frequency_penalty

    runtime_model_name = resolve_runtime_model_name(model_params)
    base_url = model_config_raw.get("base_url")
    max_retries = model_config_raw.get("max_retries", 1)

    inferred_gpt = "gpt" in model_params.model_name.lower()
    inferred_ollama = (
        "whiterabbit" in model_params.model_name.lower()
        or model_params.model_name == "LLama"
        or provider == "ollama"
    )

    if provider == "openrouter" or (inferred_gpt and provider not in ("openai", "ollama")):
        kwargs = _build_chat_openai_kwargs(
            model_params,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        kwargs["base_url"] = base_url or OPENROUTER_DEFAULT_BASE_URL
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if provider == "openai" or inferred_gpt:
        kwargs = _build_chat_openai_kwargs(
            model_params,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    if provider == "ollama" or inferred_ollama:
        if not base_url:
            raise ValueError(
                f"Ollama model '{model_params.model_name}' requires 'base_url' in Model_Params"
            )

        ollama_kwargs: Dict[str, Any] = {
            "model": runtime_model_name,
            "temperature": temperature,
            "base_url": base_url,
            "num_predict": max_tokens,
        }
        if top_p is not None:
            ollama_kwargs["top_p"] = top_p
        if top_k is not None:
            ollama_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            ollama_kwargs["repeat_penalty"] = repetition_penalty

        return ChatOllama(**ollama_kwargs)

    raise ValueError(
        f"Unsupported provider/model combination for model_name={model_params.model_name!r}, "
        f"provider={provider!r}"
    )


def build_prompt_template(model_name: str) -> PromptTemplate:
    """
    Params-only helper.
    """
    if "LLama" in model_name:
        template = """system
{system}
{history}assistant"""
    elif "gpt" in model_name.lower():
        template = """
System: {system}
History: {history}
"""
    elif "Qwen" in model_name:
        template = """<|im_start|>system
{system}<|im_end|>
{history}<|im_start|>assistant"""
    else:
        raise ValueError(f"Unsupported prompt template for model: {model_name}")

    return PromptTemplate.from_template(template=template)


def read_shell_context() -> str:
    try:
        with open("data/context/ShellIntro.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning("Could not read ShellIntro.md: %s", e)
        return "Shell context not available"


def load_agent_prompt(
    config,
    pipeline_params,
    shell_context: str,
) -> str:
    """
    Params-only helper. Uses load_system_prompt_for_agent (playbooks / snippets / context dirs).
    Workflow-specific overlays belong in core agents, not here.
    """
    load_snippets = getattr(pipeline_params, "load_snippets", True)
    load_playbooks = getattr(pipeline_params, "load_playbooks", True)
    load_context = getattr(pipeline_params, "load_context", True)
    ctx = {
        "shell_context": shell_context,
        **(config.extra_prompt_kwargs or {}),
    }
    return load_system_prompt_for_agent(
        config.prompt_key,
        base_path=config.base_path,
        load_snippets=load_snippets,
        load_playbooks=load_playbooks,
        load_context=load_context,
        context_vars=ctx,
    )


def build_chain(
    prompt,
    llm: Any,
    structured_output: Type,
    max_tokens: int,
    bind_max_tokens: bool,
):
    model_for_chain = llm.bind(max_tokens=max_tokens) if bind_max_tokens else llm
    return prompt | model_for_chain.with_structured_output(structured_output)
