# https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/build-agents/

import os
import logging
import json
from typing import Dict, Any, Optional
from types import SimpleNamespace

class CheckpointObject:
    """Proper checkpoint object that LangGraph expects."""
    def __init__(self, checkpoint_data, config_data):
        self.checkpoint = checkpoint_data
        self.config = config_data
        self.parent_config = None
        self.metadata = {"step": 0}  # LangGraph expects step in metadata
        self.parent = None
        self.pending_writes = None

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import create_fulltext_search_index
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# Global repository instance
_global_mongodb_repository = None

class MongoDBRepository:
    """Repository for MongoDB operations."""
    
    def __init__(self):
        self.vector_store = None
        self.mongodb_client = None
        self.embeddings = None
        self.initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all infrastructure components in the correct order."""
        try:
            logger.info("Starting application initialization...")

            self.mongodb_client = get_mongodb_client()
            
            # 1. Initialize embeddings model
            await self._initialize_embeddings()
            
            # 2. Initialize MongoDB and vector store
            await self._initialize_vector_store()
            
            # 3. Mark as initialized
            self.initialized = True
            
            logger.info("Application initialization completed successfully")
            
            return {
                'mongodb': self.mongodb_client,
                'vector_store': self.vector_store,
                'embeddings': self.embeddings,
                'initialized': True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            logger.info("Initializing embedding model...")
            # self.embeddings = HuggingFaceEmbeddings(
            #     model_name="sentence-transformers/all-MiniLM-L6-v2" # "sentence-transformers/all-mpnet-base-v2"
            # )
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                task="sentence-transformers",
                api_key=os.getenv("HF_TOKEN")
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    async def _initialize_vector_store(self):
        """Initialize MongoDB vector store."""
        try:
            # Create main vector store for cybersecurity docs
            self.vector_store = create_vector_store(
                collection=None,
                collection_name="cybersecurity_docs",
                embedding_model=self.embeddings
            )
            
            # Create separate vector store for HyDE docs
            self.hyde_vector_store = create_vector_store(
                collection=None,
                collection_name="hyde_docs",
                embedding_model=self.embeddings
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB vector store: {e}")
            raise
    
    def get_vector_store(self):
        """Get the initialized vector store."""
        if not self.initialized:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        return self.vector_store
    
    def get_hyde_vector_store(self):
        """Get the initialized HyDE vector store."""
        if not self.initialized:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        return self.hyde_vector_store
    
    def get_embeddings(self):
        """Get the initialized embeddings model."""
        if not self.initialized:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        return self.embeddings
    
    def is_initialized(self) -> bool:
        """Check if the application is initialized."""
        return self.initialized

def get_global_mongodb_repository() -> Optional[MongoDBRepository]:
    """Get the global MongoDB repository instance."""
    return _global_mongodb_repository

def set_global_mongodb_repository(repository: MongoDBRepository):
    """Set the global MongoDB repository instance."""
    global _global_mongodb_repository
    _global_mongodb_repository = repository

async def initialize_global_mongodb_repository() -> MongoDBRepository:
    """Initialize and set the global MongoDB repository."""
    repository = MongoDBRepository()
    await repository.initialize()
    set_global_mongodb_repository(repository)
    return repository

def get_mongodb_client() -> MongoClient:
    """Get a MongoDB client instance."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable not set")
    return MongoClient(mongodb_uri)

def get_database(database_name: str = "cybersecurity"):
    """Get a MongoDB database instance."""
    client = get_mongodb_client()
    return client[database_name]

def get_collection(collection_name: str, database_name: str = "cybersecurity"):
    """Get a MongoDB collection instance."""
    database = get_database(database_name)
    return database[collection_name]

def create_vector_store(collection: Optional[any], collection_name: str, embedding_model) -> MongoDBAtlasVectorSearch:
    """
    Create and configure a MongoDB vector store.
    
    Args:
        collection: MongoDB collection (can be None, will be created)
        collection_name: Name for the vector store collection
        embedding_model: The embedding model to use
        
    Returns:
        MongoDBAtlasVectorSearch: Configured vector store instance
    """
    # Get MongoDB connection details
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE")
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable not set")
    
    # Create MongoDB client and get database
    client = MongoClient(mongodb_uri)
    database = client[database_name]
    
    # Get or create collection
    if collection is None:
        collection = database[collection_name]
    
    # LangChain vector store setup
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=mongodb_uri,
        namespace=f"{database_name}.{collection_name}",
        embedding=embedding_model,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="dotProduct",
    )

    # Create indexes on startup
    logger.info("Setting up vector store and indexes...")
    try:
        existing_indexes = list(collection.list_search_indexes())
        vector_index_exists = any(idx.get('name') == 'vector_index' for idx in existing_indexes)
        if vector_index_exists:
            logger.info("Vector search index already exists, skipping creation...")
        else:
            logger.info("Creating vector search index...")
            vector_store.create_vector_search_index(
                dimensions=2048,  # The dimensions of the vector embeddings to be indexed
                wait_until_complete=60  # Number of seconds to wait for the index to build (can take around a minute)
            )
            logger.info("Vector search index created successfully!")
    except Exception as e:
        logger.error(f"Error creating vector search index: {e}")
        
    try:
        fulltext_index_exists = any(idx.get('name') == 'search_index' for idx in existing_indexes)
        if fulltext_index_exists:
            logger.info("Search index already exists, skipping creation...")
        else:
            logger.info("Creating search index...")
            create_fulltext_search_index(
                collection=collection,
                field="title",
                index_name="search_index",
                wait_until_complete=60  # Number of seconds to wait for the index to build (can take around a minute)
            )
            logger.info("Search index created successfully!")
    except Exception as e:
        logger.error(f"Error creating Search index: {e}")
    
    return vector_store

def get_readable_checkpoint(thread_id: str, db_name: str = None) -> Optional[Dict[str, Any]]:
    """
    Helper function to retrieve a checkpoint in readable format.
    
    Args:
        thread_id: The thread ID to retrieve
        db_name: The database name (if None, uses MONGODB_DATABASE env var)
        
    Returns:
        Dict containing the checkpoint data in readable format, or None if not found
    """
    if db_name is None:
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
    
    client = get_mongodb_client()
    db = client[db_name]

    # ------------------------------------------------------------------
    # 1) Backwards-compatible: legacy/custom checkpoints collection
    # ------------------------------------------------------------------
    try:
        legacy = db["checkpoints"].find_one(
            {"thread_id": thread_id},
            sort=[("timestamp", -1), ("created_at", -1), ("_id", -1)],
        )
        if legacy and isinstance(legacy, dict) and "state_data" in legacy:
            try:
                state_data = json.loads(legacy["state_data"]) if isinstance(legacy["state_data"], str) else legacy["state_data"]
            except Exception:
                state_data = legacy.get("state_data")
            return {
                "thread_id": legacy.get("thread_id", thread_id),
                "timestamp": legacy.get("timestamp"),
                "session_id": legacy.get("session_id", thread_id),
                "query": legacy.get("query"),
                "created_at": legacy.get("created_at"),
                "state_data": state_data or {},
                "raw_doc": {k: v for k, v in legacy.items() if k != "state_data"},
                "schema": "legacy_checkpoints",
            }
    except Exception:
        # If the legacy collection doesn't exist or query fails, fall through.
        pass

    # ------------------------------------------------------------------
    # 2) Official LangGraph MongoDBSaver schema (best-effort)
    # ------------------------------------------------------------------
    # MongoDBSaver collection names differ by version; try any collection
    # that looks like it contains checkpoints and has thread_id documents.
    collection_names = []
    try:
        collection_names = list(db.list_collection_names())
    except Exception:
        collection_names = []

    candidate_names = [
        name for name in collection_names
        if "checkpoint" in name.lower()
    ] or collection_names

    best_doc = None
    best_collection = None

    for name in candidate_names:
        try:
            col = db[name]
            doc = col.find_one(
                {"thread_id": thread_id},
                sort=[("ts", -1), ("timestamp", -1), ("checkpoint_id", -1), ("created_at", -1), ("_id", -1)],
            )
            if not doc:
                continue
            best_doc = doc
            best_collection = name
            break
        except Exception:
            continue

    if not best_doc:
        return None

    checkpoint_obj = best_doc.get("checkpoint") or {}
    # LangGraph checkpoint objects typically store channel/state values under one of these keys.
    state_data = (
        checkpoint_obj.get("channel_values")
        or checkpoint_obj.get("values")
        or checkpoint_obj.get("state")
        or checkpoint_obj.get("data")
        or {}
    )

    # Attempt to surface the original query if present
    query = None
    if isinstance(state_data, dict):
        query = state_data.get("query")

    return {
        "thread_id": best_doc.get("thread_id", thread_id),
        "timestamp": best_doc.get("ts") or best_doc.get("timestamp") or best_doc.get("created_at"),
        "session_id": thread_id,
        "query": query,
        "created_at": best_doc.get("created_at"),
        "state_data": state_data if isinstance(state_data, dict) else {},
        "raw_doc": best_doc,
        "collection": best_collection,
        "schema": "langgraph_mongodb_saver_best_effort",
    }

def list_all_checkpoints(db_name: str = None) -> list:
    """
    Helper function to list all checkpoints in the database.
    
    Args:
        db_name: The database name (if None, uses MONGODB_DATABASE env var)
        
    Returns:
        List of all checkpoints with metadata
    """
    if db_name is None:
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
    
    client = get_mongodb_client()
    db = client[db_name]
    collection = db["checkpoints"]
    
    checkpoints = list(collection.find(
        {},
        {"_id": 0, "thread_id": 1, "timestamp": 1, "session_id": 1, "query": 1, "created_at": 1}
    ).sort("timestamp", -1))
    
    return checkpoints

def delete_langgraph_checkpoint_thread(thread_id: str, db_name: Optional[str] = None) -> None:
    """
    Remove all LangGraph MongoDBSaver data for ``thread_id`` (checkpoints + pending writes).

    Without this, ``graph.invoke`` with the same ``configurable.thread_id`` merges new input
    with persisted channel state from previous runs.
    """
    if db_name is None:
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")

    from langgraph.checkpoint.mongodb import MongoDBSaver

    client = get_mongodb_client()
    saver = MongoDBSaver(client, db_name=db_name)
    if hasattr(saver, "delete_thread"):
        saver.delete_thread(thread_id)
        return

    # Older saver: same collections as MongoDBSaver defaults
    db = client[db_name]
    db["checkpoints"].delete_many({"thread_id": thread_id})
    db["checkpoint_writes"].delete_many({"thread_id": thread_id})


def clear_checkpoints(thread_id: str = None, db_name: str = None) -> int:
    """
    Helper function to clear checkpoints.
    
    Args:
        thread_id: If provided, only clear checkpoints for this thread. If None, clear all.
        db_name: The database name (if None, uses MONGODB_DATABASE env var)
        
    Returns:
        Number of checkpoints deleted
    """
    if db_name is None:
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
    
    client = get_mongodb_client()
    db = client[db_name]
    collection = db["checkpoints"]
    
    if thread_id:
        result = collection.delete_many({"thread_id": thread_id})
    else:
        result = collection.delete_many({})
    
    return result.deleted_count

