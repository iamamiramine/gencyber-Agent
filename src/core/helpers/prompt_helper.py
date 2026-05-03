"""
Utility module for loading XML prompt files and injecting context variables.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_prompt_from_xml(
    prompt_file: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Load a prompt from an XML file and inject context variables.
    
    Args:
        prompt_file: Path to the XML prompt file (relative to base_path or absolute)
        context: Dictionary of context variables to inject (e.g., {"shell_context": "..."})
        
    Returns:
        str: The loaded prompt with context variables injected
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        Exception: If there's an error reading or processing the file
    """    
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        
        # Inject context variables if provided
        if context:
            prompt_content = prompt_content.format(**context)
        
        return prompt_content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_file}: {e}")
        raise
