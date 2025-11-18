from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.services.rag_service import index_all_domains, retrieve_relevant_docs

app = FastAPI(title="Real Estate Concierge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("🔄 Starting RAG auto indexing...")
    index_all_domains()
    print("RAG indexing completed on startup.")

@app.get("/api/rag/query")
def rag_query(q: str):
    docs = retrieve_relevant_docs(q)
    return {"query": q, "documents": docs}

@app.post("/api/rag/reindex")
def rag_reindex():
    index_all_domains()
    return {"status": "RAG reindexed"}

app.include_router(api_router)
