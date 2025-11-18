# frontend/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG + Gemini Demo", layout="wide")
st.title("📄 RAG + Gemini Demo")

# ---------------------
# Sidebar: Reindex
# ---------------------
if st.sidebar.button("🔄 Reindex RAG Now"):
    try:
        res = requests.post("http://localhost:8000/api/rag/reindex")
        st.sidebar.success("RAG reindexed successfully!")
    except Exception as e:
        st.sidebar.error(f"Reindex failed: {e}")

# ---------------------
# Ask a question
# ---------------------
query = st.text_input("Enter your question here:")

if st.button("Ask"):
    if query.strip():
        try:
            res = requests.get(f"http://localhost:8000/api/rag/query?q={query}")
            data = res.json()
            st.subheader("✅ AI Answer:")
            st.write(data.get("answer"))

            st.subheader("🔍 Retrieved Documents:")
            for doc in data.get("docs", []):
                st.markdown(f"- **{doc['metadata']['source']} (chunk {doc['metadata']['chunk']})**: {doc['text'][:300]}...")
        except Exception as e:
            st.error(f"Query failed: {e}")
