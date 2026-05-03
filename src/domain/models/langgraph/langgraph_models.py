from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from fastapi import Query


class WorkflowParameters(BaseModel):
    """Parameters for initializing and configuring a LangGraph workflow."""
    
    # Workflow behavior settings
    use_tracing: bool = Field(default=False, description="Whether to enable LangSmith tracing for the workflow")
    max_retries: int = Field(default=3, description="Maximum number of retries for regeneration attempts")
    quality_threshold: float = Field(default=0.7, description="Quality threshold for response acceptance (0.0-1.0)")
    
    # Agent configurations
    document_chunk_size: int = Field(default=1000, description="Chunk size for document splitting")
    document_chunk_overlap: int = Field(default=200, description="Chunk overlap for document splitting")
    
    # Retrieval configurations
    retriever_k: int = Field(default=5, description="Number of documents to retrieve")
    
    # Script execution configurations
    script_timeout: int = Field(default=30, description="Timeout in seconds for script execution")
