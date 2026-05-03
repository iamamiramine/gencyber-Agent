from typing import Dict, Any, List, Type, Optional
import logging
import os
import pickle
import time
import json
from pathlib import Path
from pydantic import BaseModel, Field

# External dependencies
from langchain_core.documents import Document
from langchain.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

# Internal dependencies
from infrastructure.repository.chromadb_repository import get_chromadb_repository, ChromaDBRepository

# Configure logging
logger = logging.getLogger(__name__)    


class VectorDBInput(BaseModel):
    """Input for VectorDBTool."""
    source_path: str = Field(description="Path to the directory or URL containing source documents")
    collection_name: str = Field(default="chroma", description="Name for the vector database collection")
    hypothetical_document: str = Field(default=None, description="Hypothetical document to add to the vector database")

class VectorDBTool(BaseTool):
    """
    Tool for checking and managing vector databases.
    
    This tool is responsible for:
    1. Checking if a vector database exists at a given location
    2. Determining if the vector database is up-to-date
    3. Requesting document processing if needed
    4. Managing vector database lifecycle
    """
    
    name: str = "vector_db_manager"
    description: str = """
    Check and manage vector databases for document retrieval.
    The tool verifies if a vector database exists and is up-to-date compared to source documents.
    It coordinates with the document processing tool to update the database when needed.
    """
    args_schema: Type[BaseModel] = VectorDBInput

    def __init__(self, **kwargs):
        """Initialize the vector database tool."""
        super().__init__(**kwargs)
        # Get the global ChromaDB repository - use private attribute to avoid Pydantic issues
        self._chromadb_repo: ChromaDBRepository = get_chromadb_repository()
        

    def _run(self, source_path: str, collection_name: str = "chroma", hypothetical_document: str = None) -> Dict[str, Any]:
        """
        Check if a vector database exists and is up-to-date.
        
        Args:
            source_path: Path to source documents (local directory or URL)
            collection_name: Name for the vector database collection
            hypothetical_document: Hypothetical document to add to the HyDE vector database
            
        Returns:
            Dict with results including:
            - vector_db_loaded: Whether the main vector database is loaded
        """
        # Check if ChromaDB repository is initialized
        if not self._chromadb_repo.is_initialized():
            logger.warning("ChromaDB repository not initialized")
            return {"vector_db_loaded": False}
        
        # Always attempt to get/create the vector store and update it.
        main_store = self._chromadb_repo.get_main_vector_store()
        self._process_and_add_documents(source_path, collection_name)
        self._log_persist_dir(main_store)
        
        # Add HyDE document if provided
        if hypothetical_document:
            self._add_hyde_document(hypothetical_document)
        
        return {"vector_db_loaded": True}
    
    def _process_and_add_documents(self, source_path: str, collection_name: str = "chroma"):
        """
        Smart indexing with manifest check + hybrid retrieval prep.
        
        Args:
            source_path: Path to source documents
            collection_name: Name for the collection
        """
        try:
            manifest_path = Path("data/store/index_manifest.json")
            previous = {}
            if manifest_path.exists():
                try:
                    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Manifest corrupt; reindexing entire dataset")

            current_files = self._list_supported_files(source_path)
            current_index = {str(p): p.stat().st_mtime for p in current_files}
            to_update = [p for p in current_files if str(p) not in previous or current_index[str(p)] > float(previous.get(str(p), 0))]

            if not to_update:
                logger.info("No new or modified documents detected; skipping indexing")
                return

            logger.info(f"Indexing {len(to_update)} new/updated documents...")
            docs = self._load_documents_for_paths(to_update)
            splits = self.split_documents(docs, chunk_size=700, chunk_overlap=150)
            valid = [d for d in splits if d.page_content.strip()]

            if not valid:
                logger.warning("No valid text chunks; aborting update")
                return

            store = self._chromadb_repo.get_main_vector_store()
            self._add_in_batches(store, valid)

            store.persist()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(current_index, indent=2), encoding="utf-8")
            logger.info("Vector DB updated successfully")

            # Create auxiliary BM25 retriever cache
            self.save_text_chunks(valid)
            logger.info("BM25 text cache updated")

        except Exception as e:
            logger.error(f"VectorDBTool error: {e}")
            raise

    def _add_hyde_document(self, hypothetical_doc: str):
        """
        Add a hypothetical document to the HyDE vector store.
        
        Args:
            hypothetical_doc: The hypothetical document to add
        """
        try:
            if not hypothetical_doc:
                return
                
            # Get the HyDE vector store from the repository
            hyde_store = self._chromadb_repo.get_hyde_vector_store()
            
            # Create a document from the hypothetical text
            doc = Document(page_content=hypothetical_doc)
            hyde_store.add_documents([doc])
            
            logger.info("HyDE document added to vector store successfully")
            
        except Exception as e:
            logger.error(f"Error adding HyDE document: {e}")
            raise

    def _list_supported_files(self, data_dir: str):
        """
        List all supported files in the directory.
        
        Args:
            data_dir: Directory to search for files
            
        Returns:
            List of Path objects for supported files
        """
        exts = {".txt", ".csv", ".pdf", ".json", ".md", ".markdown"}
        return [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts]

    def _load_documents_for_paths(self, paths: List[Path]):
        """
        Load documents from a list of file paths.
        
        Args:
            paths: List of Path objects to load documents from
            
        Returns:
            List of Document objects
        """
        docs: List[Document] = []
        for path in paths:
            try:
                suffix = path.suffix.lower()
                if suffix == ".txt":
                    docs.extend(TextLoader(str(path)).load())
                elif suffix == ".pdf":
                    docs.extend(PyPDFLoader(path).load())
                elif suffix == ".csv":
                    docs.extend(CSVLoader(file_path=str(path)).load())
                elif suffix == ".json":
                    text = path.read_text(encoding="utf-8")
                    docs.append(Document(page_content=text))
                elif suffix in {".md", ".markdown"}:
                    # Markdown files are plain text, use TextLoader
                    docs.extend(TextLoader(str(path)).load())
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return docs

    def _add_in_batches(self, store, docs: list, batch_size: int = 128):
        """
        Add documents to the vector store in batches with retry logic.
        
        Args:
            store: Vector store to add documents to
            docs: List of documents to add
            batch_size: Size of each batch
        """
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                store.add_documents(batch)
            except Exception as e:
                logger.warning(f"Retrying batch {i} due to {e}")
                time.sleep(1)
                store.add_documents(batch)


    def split_documents(self, docs, chunk_size=700, chunk_overlap=150):
        """
        Split documents into smaller chunks for processing.
        
        Args:
            docs: List of documents to split. Can be either raw text content
                 or Document objects with page_content attribute.
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
                 
        Returns:
            list: List of Document objects containing the chunked text.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        contents = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs]
        texts = splitter.create_documents(contents)
        logger.info(f"Split {len(texts)} text chunks.")
        return texts


    def save_text_chunks(self, texts, file_path: str = "./data/store/chunks.pkl"):
        """
        Save text chunks to a pickle file and update dataset metadata.
        
        Args:
            texts: List of text chunks to save
            file_path (str): Path where chunks will be saved. Defaults to "./data/store/chunks.pkl"
        """
        # Create directories in path if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save text chunks to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(texts, f)


    def load_text_chunks(self, file_path: str = "./data/store/chunks.pkl"):
        """
        Load text chunks from a pickle file.
        
        Args:
            file_path (str): Path to the pickle file containing text chunks
            
        Returns:
            list: The loaded text chunks from the pickle file
        """
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'rb') as f:
            return pickle.load(f)


    def _log_persist_dir(self, store):
        """
        Log the persist directory for the vector store.
        
        Args:
            store: The vector store to check
        """
        try:
            client = getattr(store, "_client", None)
            settings = getattr(client, "_settings", None)
            persist_dir = getattr(settings, "persist_directory", None)
            logger.info(f"Chroma persist directory: {persist_dir}")
        except Exception:
            logger.info("Could not log persist directory.")
