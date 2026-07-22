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

# gencyber-Engine: our local OpenAI-compatible server (vLLM) hosting the open-source
# models. Any model_name the engine serves is routed here instead of OpenRouter — the
# agent decides from the model name ALONE (see ``resolve_engine_target``), so callers
# only ever change the model name. Reached by container name on the shared docker net.
GENCYBER_ENGINE_DEFAULT_BASE_URL = "http://gencyber-engine:8000/v1"
_ENGINE_CATALOG_CACHE: Optional[Dict[str, Any]] = None
# Built-in fallback so routing still works if the catalog YAML is absent (minimal
# deploys). The YAML, when present, is the source of truth and extends/overrides this.
_ENGINE_MODELS_FALLBACK = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
}


def _engine_base_url() -> str:
    """The engine endpoint, from env with a shared-network default."""
    return (
        os.environ.get("GENCYBER_ENGINE_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
        or os.environ.get("OPENAI_COMPAT_BASE_URL")
        or GENCYBER_ENGINE_DEFAULT_BASE_URL
    ).strip()


def _load_engine_catalog() -> Dict[str, Any]:
    """Load ``config/models/engine_models.yaml`` (cached).

    Returns ``{"models": {name: base_url|None}, "default_base_url": str|None}``. A
    missing/unreadable file falls back to the built-in Qwen set with no per-model URLs,
    so the engine still works out of the box.
    """
    global _ENGINE_CATALOG_CACHE
    if _ENGINE_CATALOG_CACHE is not None:
        return _ENGINE_CATALOG_CACHE
    path = os.environ.get("GENCYBER_ENGINE_CATALOG", "config/models/engine_models.yaml")
    models: Dict[str, Optional[str]] = {}
    default_base_url: Optional[str] = None
    try:
        import yaml  # pyyaml is already a dependency

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        default_base_url = data.get("default_base_url")
        for entry in data.get("models", []) or []:
            if isinstance(entry, str):
                models[entry] = None
            elif isinstance(entry, dict) and entry.get("name"):
                models[str(entry["name"])] = entry.get("base_url")
    except FileNotFoundError:
        logger.info("engine catalog %s not found; using built-in fallback set", path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("failed to read engine catalog %s (%s); using fallback", path, e)
    if not models:
        models = {name: None for name in _ENGINE_MODELS_FALLBACK}
    _ENGINE_CATALOG_CACHE = {"models": models, "default_base_url": default_base_url}
    return _ENGINE_CATALOG_CACHE


def resolve_engine_target(model_name: str) -> Optional[str]:
    """If ``model_name`` is served by gencyber-Engine, return its base_url; else None.

    Single source of the "OpenRouter vs gencyber-Engine" decision, made from the model
    name alone. Per-model ``base_url`` in the catalog wins (lets each model live on its
    own vLLM instance); otherwise the catalog ``default_base_url`` or the engine env.
    """
    cat = _load_engine_catalog()
    if model_name not in cat["models"]:
        return None
    return cat["models"].get(model_name) or cat.get("default_base_url") or _engine_base_url()


def reset_engine_catalog_cache() -> None:
    """Test/hot-reload hook: drop the cached catalog so the next call re-reads it."""
    global _ENGINE_CATALOG_CACHE
    _ENGINE_CATALOG_CACHE = None


def _engine_serves(model_name: str, base_url: str, timeout: float = 3.0) -> Optional[bool]:
    """Best-effort: does the engine's /models list include ``model_name``?

    Returns True/False when the list is fetched, or None when it can't be reached or is
    empty (so an unreachable engine never blocks init — we only act on a definitive no).
    A vLLM instance serves ONE model, so a name mismatch (wrong ENGINE_MODEL, or a model
    the operator forgot to (re)start the engine with) otherwise surfaces as a cryptic
    404 mid-run; this turns it into a clear, actionable error at init.
    """
    try:
        import json as _json
        import urllib.request

        url = base_url.rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec - internal net
            data = _json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
        if not ids:
            return None
        return model_name in ids
    except Exception:
        return None


def _engine_preflight_enabled() -> bool:
    return os.environ.get("GENCYBER_ENGINE_PREFLIGHT", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


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

    # ── Decision 1 (name-first): does gencyber-Engine serve this model? ──────────
    # If so, route to the engine regardless of any provider hint left in config — the
    # whole point is that changing ONLY the model name moves a run between OpenRouter
    # and the local engine. An explicit ``provider: openrouter/openai/ollama`` still
    # forces the cloud/ollama path for a name NOT in the engine catalog.
    engine_base_url = resolve_engine_target(model_params.model_name)
    if engine_base_url:
        # Preflight: a vLLM instance serves exactly one model. If the engine is up but
        # serving a DIFFERENT model than requested, fail now with an actionable message
        # rather than 404-ing mid-run. Best-effort — an unreachable engine doesn't block
        # (the run then fails later with a normal connection error). Disable with
        # GENCYBER_ENGINE_PREFLIGHT=0.
        if _engine_preflight_enabled():
            served = _engine_serves(model_params.model_name, engine_base_url)
            if served is False:
                raise ValueError(
                    f"gencyber-Engine at {engine_base_url} is not serving "
                    f"'{model_params.model_name}'. A vLLM instance serves one model: set "
                    f"ENGINE_MODEL={model_params.model_name} in gencyber-Engine/.env and "
                    f"restart it (or point GENCYBER_ENGINE_BASE_URL at an instance that "
                    f"serves it). Set GENCYBER_ENGINE_PREFLIGHT=0 to skip this check."
                )
        kwargs = _build_chat_openai_kwargs(
            model_params,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        kwargs["base_url"] = engine_base_url
        # The engine (vLLM) ignores the key; default to the vLLM convention.
        kwargs["api_key"] = (
            os.environ.get("GENCYBER_ENGINE_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or "EMPTY"
        )
        logger.info(
            "routing model %s -> gencyber-Engine at %s", model_params.model_name, engine_base_url
        )
        return ChatOpenAI(**kwargs)

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

    if provider in ("vllm", "openai_compat", "openai-compatible", "local"):
        # Any OpenAI-compatible server: vLLM (recommended for on-prem open-source
        # models), llama.cpp ``--api``, LM Studio, TGI, or Ollama's OpenAI shim. We
        # reuse ChatOpenAI so the entire tool-calling / structured-output stack is
        # byte-for-byte identical to the cloud (OpenRouter) path — only the endpoint
        # and key differ. This is the unified way to run every hosted-or-local
        # open-source model without a new provider branch per family.
        kwargs = _build_chat_openai_kwargs(
            model_params,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        resolved_base = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or os.environ.get("OPENAI_COMPAT_BASE_URL")
        )
        if not resolved_base:
            raise ValueError(
                f"Provider '{provider}' requires a base_url (Model_Params.base_url or "
                f"VLLM_BASE_URL / OPENAI_COMPAT_BASE_URL env) pointing at the "
                f"OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
            )
        kwargs["base_url"] = resolved_base
        # vLLM/llama.cpp accept any key; default to the vLLM convention so a missing
        # key never turns into a confusing auth error.
        kwargs["api_key"] = (
            os.environ.get("VLLM_API_KEY")
            or os.environ.get("OPENAI_COMPAT_API_KEY")
            or "EMPTY"
        )
        return ChatOpenAI(**kwargs)

    if provider == "ollama" or inferred_ollama:
        # base_url is a server-side concern: callers only pass a model name, so fall
        # back to OLLAMA_BASE_URL from the environment.
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL")
        if not base_url:
            raise ValueError(
                f"Ollama model '{model_params.model_name}' requires 'base_url' "
                f"(Model_Params.base_url or OLLAMA_BASE_URL env)"
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
    """Build the {system}/{history} wrapper for the baseline chain.

    Only the legacy locally-served HF checkpoints (the ``WhiteRabbit-*`` / bare
    ``LLama`` names) need us to inject their raw chat template — no serving layer
    applies one for them. Every OpenAI-compatible endpoint (OpenRouter, vLLM,
    Ollama's OpenAI shim) already applies the model's native chat template server
    side, so for all other names — including open-source slugs like
    ``qwen/...``, ``deepseek/...``, ``meta-llama/...`` — we use a neutral
    System/History wrapper. Injecting ``<|im_start|>`` here on top of a server
    that also applies it would double-template the prompt. Unknown names default
    to the neutral wrapper instead of raising, so a new model never breaks init.
    """
    name = model_name or ""
    if name == "LLama" or "WhiteRabbit-LLama" in name:
        template = """system
{system}
{history}assistant"""
    elif "WhiteRabbit-Qwen" in name:
        template = """<|im_start|>system
{system}<|im_end|>
{history}<|im_start|>assistant"""
    else:
        template = """
System: {system}
History: {history}
"""
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
    structured_output_method: Optional[str] = None,
):
    """Compose ``prompt | llm.with_structured_output(schema)``.

    ``structured_output_method`` maps to LangChain's ``with_structured_output(method=...)``.
    Leave it unset (default) for the OpenAI/OpenRouter function-calling behaviour used
    today. For weaker open-source models served by vLLM, set it (via
    ``Model_Params.structured_output_method``) to ``"json_schema"`` (guided decoding —
    most reliable) or ``"json_mode"``; these adhere to the schema far better than
    function calling on small models.
    """
    model_for_chain = llm.bind(max_tokens=max_tokens) if bind_max_tokens else llm
    if structured_output_method:
        return prompt | model_for_chain.with_structured_output(
            structured_output, method=structured_output_method
        )
    return prompt | model_for_chain.with_structured_output(structured_output)
