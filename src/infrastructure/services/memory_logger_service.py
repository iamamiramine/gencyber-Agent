from typing import Dict, Any
import logging
import hashlib
import json
from datetime import datetime
import os

from langgraph.store.mongodb import MongoDBStore, create_vector_index_config

logger = logging.getLogger(__name__)

class MemoryLogger:
    """Minimal memory logging service for agent interactions and structured outputs."""
    
    def __init__(self):
        """Initialize the memory logger."""
        self.db_name = "gencyber"
        self.collection_name = "agent_memories"
        
        # Initialize connection parameters
        self.mongodb_uri = None
        self.index_config = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the MongoDB store for memory logging."""
        try:
            # Get MongoDB URI from environment
            mongodb_uri = os.getenv("MONGODB_URI")
            if not mongodb_uri:
                logger.error("MONGODB_URI environment variable not set")
                return
            
            # Get embeddings from the global repository for vector search
            from infrastructure.repository.mongodb_repository import get_global_mongodb_repository
            repo = get_global_mongodb_repository()
            
            if repo and repo.embeddings:
                # Create vector index config for semantic search
                index_config = create_vector_index_config(
                    embed=repo.embeddings,
                    dims=384,  # all-MiniLM-L6-v2 dimensions
                    fields=["original_query", "query_to_process", "context", "command", "script_output"],
                    filters=["session_id", "agent_type", "timestamp"]
                )
            else:
                index_config = None
            
            # Store the connection parameters for creating the store when needed
            self.mongodb_uri = mongodb_uri
            self.index_config = index_config
            
            logger.info("Memory logger initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory logger: {e}")
            self.mongodb_uri = None
            self.index_config = None

    def log_comprehensive_interaction(self, 
                                    session_id: str, 
                                    agent_type: str, 
                                    original_query: str,
                                    query_to_process: str,
                                    state: Dict[str, Any],
                                    structured_response: Dict[str, Any]) -> bool:
        """
        Log a comprehensive agent interaction including all information in a single entry.
        
        Args:
            session_id: The session identifier
            agent_type: Type of agent (e.g., 'generative_agent', 'rag_agent', 'hyde_agent')
            original_query: The original query from the user
            query_to_process: The query that was actually processed by the LLM
            state: Current state of the agent
            structured_response: The structured response from the LLM
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        if not self.mongodb_uri:
            logger.warning("MongoDB URI not set, skipping comprehensive memory logging")
            return False
        
        try:
            # Create comprehensive memory content
            memory_content = {
                "session_id": session_id,
                "agent_type": agent_type,
                "timestamp": datetime.now().isoformat(),
                "original_query": original_query,
                "query_to_process": query_to_process,
                "context": state.get("context", ""),
                "command": state.get("command", ""),
                "script_output": state.get("script_output", ""),
                "submission_verified": state.get("submission_verified", False),
                "structured_response": structured_response
            }
            
            # Create unique key for the memory
            content_hash = hashlib.md5(
                json.dumps(memory_content, sort_keys=True).encode()
            ).hexdigest()
            
            # Create store and store the memory
            with MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.db_name,
                collection_name=self.collection_name,
                index_config=self.index_config
            ) as store:
                store.put(
                    namespace=(session_id, "agent_memories"),
                    key=f"memory_{agent_type}_{content_hash}",
                    value=memory_content
                )
            
            logger.debug(f"Logged comprehensive memory for {agent_type} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log comprehensive memory for {agent_type}: {e}")
            return False

    def log_execution_event(self,
                            session_id: str,
                            command: str,
                            result: Dict[str, Any]) -> bool:
        """Log a single command/script execution event (audit trail).

        Stored separately from agent interactions so command outcomes (exit
        code, timeout, backgrounding) can be replayed without inflating LLM
        context. Best-effort: never raises.

        Args:
            session_id: The session identifier.
            command: The command that was executed.
            result: Structured outcome (exit_code, timed_out, running, etc.).

        Returns:
            bool: True if logging was successful, False otherwise.
        """
        if not self.mongodb_uri:
            return False
        try:
            event = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "command": command,
                **(result or {}),
            }
            content_hash = hashlib.md5(
                json.dumps(event, sort_keys=True, default=str).encode()
            ).hexdigest()
            with MongoDBStore.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.db_name,
                collection_name="execution_events",
            ) as store:
                store.put(
                    namespace=(session_id, "execution_events"),
                    key=f"exec_{content_hash}",
                    value=event,
                )
            return True
        except Exception as e:
            logger.debug(f"Failed to log execution event: {e}")
            return False

    def close(self):
        """Close the memory logger."""
        # No persistent store to close
        pass

# Global memory logger instance
_global_memory_logger = None

def get_memory_logger() -> MemoryLogger:
    """Get the global memory logger instance."""
    global _global_memory_logger
    if _global_memory_logger is None:
        _global_memory_logger = MemoryLogger()
    return _global_memory_logger
