from typing import Annotated, Optional

from fastapi import Query
from pydantic import BaseModel

from domain.models.base_models import BaseEnum

class TaskType(str, BaseEnum):
    """Available task types for the language model pipeline."""
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"

class LoadModelParameters(BaseModel):
    model_name: Annotated[str, Query(description="The name of the model")] = "WhiteRabbit"
    model_path: Annotated[str, Query(description="The path to the model")] = "models/WhiteRabbitNeo_WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"
    bit_quantization: Optional[Annotated[int, Query(description="The bit quantization to use (4 or 8)")]] = None

class PipelineParameters(BaseModel):
    model_name: Annotated[str, Query(description="The name of the model")] = "WhiteRabbit"
    task_type: Annotated[TaskType, Query(description="The type of task for the pipeline")] = TaskType.TEXT_GENERATION
    max_new_tokens: Annotated[int, Query(description="The maximum number of new tokens to generate")] = 256
    do_sample: Annotated[bool, Query(description="Whether to use sampling")] = True
    temperature: Annotated[float, Query(description="The temperature for sampling")] = 0.8
    top_p: Annotated[float, Query(description="The top-p value for sampling")] = 0.9
    top_k: Annotated[int, Query(description="The top-k value for sampling")] = 50
    repetition_penalty: Annotated[float, Query(description="The repetition penalty for sampling")] = 1.1
    presence_penalty: Annotated[float, Query(description="Penalizes tokens that have already appeared in the text")] = 0.1
    frequency_penalty: Annotated[float, Query(description="Penalizes tokens that appear frequently in the text")] = 0.1
    no_repeat_ngram_size: Annotated[int, Query(description="Prevents repetition of n-grams of this size")] = 3
    # Optional: include playbooks/snippets/context in generative system prompt (default True)
    load_playbooks: Annotated[bool, Query(description="Load playbooks from config/prompts/playbooks into system prompt")] = True
    load_snippets: Annotated[bool, Query(description="Load snippets from config/prompts/snippets into system prompt")] = True
    load_context: Annotated[bool, Query(description="Load context from config/prompts/context into system prompt")] = True

class EnsembleRetrieverParameters(BaseModel):
    force_reload: Annotated[bool, Query(description="Whether to force reload the dataset")] = False
