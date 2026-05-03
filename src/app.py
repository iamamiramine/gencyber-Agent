from fastapi import FastAPI
import asyncio

from api.controllers import (
    health_controller,
    langgraph_controller,
)
from handlers.exception_handler import add_exception_handlers
from infrastructure.repository.chromadb_repository import get_chromadb_repository

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "langchain",
        "description": "Langchain",
    },
    {
        "name": "langgraph",
        "description": "LangGraph Agentic Workflow",
    },
]


app = FastAPI(
    version="1.0",
    title="Generative Cybersecurity API",
    description="API for Generative Cybersecurity",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    langgraph_controller.router,
    prefix="/langgraph",
    tags=["langgraph"],
    responses={404: {"description": "Not found"}},
)

add_exception_handlers(app=app)


# @app.on_event("startup")
# async def startup_event():
#     """Initialize ChromaDB repository on startup."""
#     try:
#         chromadb_repo = get_chromadb_repository()
#         await chromadb_repo.initialize()
#     except Exception as e:
#         print(f"Failed to initialize ChromaDB repository: {e}")
#         # Don't fail startup, just log the error


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)