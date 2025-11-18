# app/services/gemini_service.py
import google.generativeai as genai
from app.services.rag_service import retrieve_relevant_docs

genai.configure(api_key="YOUR_GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

def generate_rag_answer(query: str, top_k=3):
    # Retrieve relevant docs from RAG
    docs = retrieve_relevant_docs(query, top_k=top_k)
    context_text = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
Use the following context to answer the question.
Context:
{context_text}

Question:
{query}
Answer concisely and cite sources.
"""
    if not context_text.strip():
        return {"answer": "Information not available in official documents.", "docs": docs}

    resp = genai.generate_text(model=MODEL, prompt=prompt)
    return {"answer": resp.text, "docs": docs}
