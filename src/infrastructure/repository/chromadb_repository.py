import os
import logging
from typing import Dict, Any, Optional, List

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Global repository instance
_global_chromadb_repository = None

class ChromaDBRepository:
    """Repository for ChromaDB operations."""
    
    def __init__(self):
        self.main_vector_store = None
        self.hyde_vector_store = None
        self.embeddings = None
        self.initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all infrastructure components in the correct order."""
        try:
            logger.info("Starting ChromaDB initialization...")

            # 1. Initialize embeddings model
            await self._initialize_embeddings()
            
            # 2. Initialize ChromaDB vector stores
            await self._initialize_vector_stores()
            
            # 3. Mark as initialized
            self.initialized = True
            
            logger.info("ChromaDB initialization completed successfully")
            
            return {
                'main_vector_store': self.main_vector_store,
                'hyde_vector_store': self.hyde_vector_store,
                'embeddings': self.embeddings,
                'initialized': True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            logger.info("Initializing embedding model...")
            self.embeddings = HuggingFaceEndpointEmbeddings(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    async def _initialize_vector_stores(self):
        """Initialize ChromaDB vector stores."""
        try:
            # Create main vector store for cybersecurity docs
            self.main_vector_store = self._create_vector_store(
                collection_name="cybersecurity_docs",
                persist_directory="./data/store/cybersecurity_docs"
            )
            
            # Create separate vector store for HyDE docs
            self.hyde_vector_store = self._create_vector_store(
                collection_name="hyde_docs",
                persist_directory="./data/store/hyde_docs"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector stores: {e}")
            raise
    
    def _create_vector_store(self, collection_name: str, persist_directory: str) -> Chroma:
        """Create or load a ChromaDB vector store."""
        os.makedirs(persist_directory, exist_ok=True)
        
        db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        return db
    
    def get_main_vector_store(self):
        """Get the main vector store."""
        if not self.initialized:
            raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
        return self.main_vector_store
    
    def get_hyde_vector_store(self):
        """Get the HyDE vector store."""
        if not self.initialized:
            raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
        return self.hyde_vector_store
    
    def get_embeddings(self):
        """Get the initialized embeddings."""
        if not self.initialized:
            raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
        return self.embeddings
    
    def is_initialized(self) -> bool:
        """Check if the repository is initialized."""
        return self.initialized

def get_chromadb_repository() -> ChromaDBRepository:
    """Get the global ChromaDB repository instance."""
    global _global_chromadb_repository
    if _global_chromadb_repository is None:
        _global_chromadb_repository = ChromaDBRepository()
    return _global_chromadb_repository
