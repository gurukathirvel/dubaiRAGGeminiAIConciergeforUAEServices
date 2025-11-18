# app/api/rag_endpoints.py
from fastapi import APIRouter
from app.services.rag_service import retrieve_relevant_docs, index_all_domains

router = APIRouter(prefix="/api/rag")

@router.get("/query")
async def rag_query(q: str):
    results = retrieve_relevant_docs(q)
    return {"query": q, "results": results}

@router.post("/reindex")
async def rag_reindex():
    index_all_domains()
    return {"status": "success", "message": "RAG reindexing completed."}
