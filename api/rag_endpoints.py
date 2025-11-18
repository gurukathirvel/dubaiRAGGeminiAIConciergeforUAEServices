# app/api/rag_endpoints.py
from fastapi import APIRouter
from app.services.rag_service import retrieve_relevant_docs, index_all_domains

rag_router = APIRouter(prefix="/api/rag")

@rag_router.get("/query")
async def rag_query(q: str):
    results = retrieve_relevant_docs(q)
    return {"query": q, "results": results}

@rag_router.post("/reindex")
async def rag_reindex():
    index_all_domains()
    return {"status": "success", "message": "RAG reindexing completed."}
