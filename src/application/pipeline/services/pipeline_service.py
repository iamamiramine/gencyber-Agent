from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

from application.pipeline.helpers import pipeline_helper as ph
from domain.models.langchain.langchain_models import LoadModelParameters, PipelineParameters
from domain.models.langgraph.agents_models import AgentConfig


class PipelineConfigService:
    def __init__(self, registry_path: str = "config/pipeline/default_pipeline_registry.yaml") -> None:
        self.registry_path = Path(registry_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_runtime_config(self) -> Dict[str, Any]:
        registry = ph.load_yaml_file(self.registry_path)
        agent_entries = ph.parse_registry(registry)

        agent_definitions: Dict[str, AgentConfig] = {}
        model_params: Dict[str, LoadModelParameters] = {}
        pipeline_params: Dict[str, PipelineParameters] = {}
        generation_configs: Dict[str, Dict[str, Any]] = {}
        model_config_raw: Dict[str, Dict[str, Any]] = {}

        history_keys: Set[str] = set()

        for entry in agent_entries:
            registry_name = entry["name"]
            config_path = ph.resolve_config_path(self.registry_path, entry["config_path"])
            agent_yaml = ph.load_agent_yaml(config_path)

            agent_config = ph.build_agent_config(agent_yaml["Agent_Config"])
            if agent_config.name != registry_name:
                raise ValueError(
                    f"Registry agent name '{registry_name}' does not match "
                    f"Agent_Config.name '{agent_config.name}' in {config_path}"
                )

            model_param = ph.build_model_params(agent_yaml["Model_Params"])
            pipeline_param = ph.build_pipeline_params(
                model_params=model_param,
                raw=agent_yaml["Pipeline_Params"],
            )
            generation_config = ph.build_generation_config(
                raw=agent_yaml["Generation_Config"],
                pipeline_params=pipeline_param,
            )

            agent_definitions[registry_name] = agent_config
            model_params[registry_name] = model_param
            pipeline_params[registry_name] = pipeline_param
            generation_configs[registry_name] = generation_config
            model_config_raw[registry_name] = dict(agent_yaml["Model_Params"])

            history_keys.add(agent_config.share_history_key or registry_name)

        return {
            "registry_path": str(self.registry_path),
            "agent_definitions": agent_definitions,
            "model_params": model_params,
            "pipeline_params": pipeline_params,
            "generation_configs": generation_configs,
            "model_config_raw": model_config_raw,
            "history_keys": sorted(history_keys),
            "registry_agents": agent_entries,
        }

    def list_pipeline_registries(self) -> List[Dict[str, Any]]:
        """Discover ``*_pipeline_registry.yaml`` files next to the active registry file."""
        return ph.list_pipeline_registry_ids(self.registry_path)
