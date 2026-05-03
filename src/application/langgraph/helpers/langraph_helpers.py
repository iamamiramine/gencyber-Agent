from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from application.langgraph.helpers.system_prompt_helper import load_system_prompt_for_agent
from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters


logger = logging.getLogger(__name__)


def resolve_runtime_model_name(params: LoadModelParameters) -> str:
    if params.model_name == "LLama":
        return "llama3"
    return params.model_path or params.model_name


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

    inferred_openai = "gpt" in model_params.model_name.lower()
    inferred_ollama = (
        "whiterabbit" in model_params.model_name.lower()
        or model_params.model_name == "LLama"
        or provider == "ollama"
    )

    if provider == "openai" or inferred_openai:
        model_kwargs: Dict[str, Any] = {}
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
            "model_kwargs": model_kwargs,
        }
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
