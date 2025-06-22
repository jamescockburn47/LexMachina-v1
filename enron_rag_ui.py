import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
import os

# --- CONFIGURATION ---
VECTORSTORE_PATH = '/home/jcockburn/LexMachina/vectorstore/db.faiss'
META_PATH = '/home/jcockburn/LexMachina/vectorstore/meta.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MISTRAL_URL = 'http://localhost:11434/api/generate'  # Local Mistral LLM endpoint
N_CANDIDATES = 50  # Number of FAISS candidates to check with LLM
N_DISPLAY = 10     # Number of results to display after LLM filtering

# --- UTILS ---
def load_faiss_index(index_path: str):
    if not os.path.exists(index_path):
        st.error(f"FAISS index not found at {index_path}")
        st.stop()
    return faiss.read_index(index_path)

def load_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        st.error(f"Metadata file not found at {meta_path}")
        st.stop()
    with open(meta_path, 'rb') as f:
        return pickle.load(f)

def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True)

def mistral_relevance_filter(question, chunk_text):
    """Call local Mistral LLM to check if chunk is relevant to the question."""
    prompt = f"Is the following email chunk relevant to the user's question?\n\nQuestion: {question}\n\nEmail chunk: {chunk_text}\n\nAnswer 'yes' or 'no'."
    payload = {
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(MISTRAL_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        answer = data.get('response', '').strip().lower()
        return 'yes' in answer
    except Exception as e:
        st.warning(f"[Mistral LLM filter error]: {e}")
        return True  # Fail open: treat as relevant if unsure

# --- MAIN APP ---
st.set_page_config(page_title="Enron RAG Legal Research", layout="wide")
st.title("Enron RAG Legal Research UI")
st.markdown("""
A professional research tool for exploring the Enron email corpus using Retrieval-Augmented Generation (RAG).\
Enter a legal or factual question below to search the most relevant email content.\
Results are filtered by a local Mistral LLM for true semantic relevance.
""")

# --- LOAD DATA ON STARTUP ---
@st.cache_resource(show_spinner=True)
def load_resources():
    index = load_faiss_index(VECTORSTORE_PATH)
    meta = load_metadata(META_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return index, meta, model

index, meta_list, embed_model = load_resources()

# --- USER INPUT ---
with st.form(key="query_form"):
    user_query = st.text_input("Enter your legal or factual question:", value="", help="Type your question and click Search.")
    submit = st.form_submit_button("Search")

# --- TWO-STAGE SEARCH LOGIC ---
search_results = []
if submit and user_query.strip():
    with st.spinner("Embedding and searching..."):
        query_vec = embed_text(user_query, embed_model)
        D, I = index.search(query_vec, N_CANDIDATES)
        candidates = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(meta_list):
                continue
            meta = meta_list[idx]
            chunk_text = meta.get('chunk_text', '[No text]')
            # --- LLM FILTER: Only keep chunks Mistral rates as relevant ---
            if mistral_relevance_filter(user_query, chunk_text):
                candidates.append({'text': chunk_text, 'metadata': meta, 'score': float(score)})
            if len(candidates) >= N_DISPLAY:
                break
        search_results = candidates

# --- RESULTS DISPLAY ---
st.subheader(f"Top {N_DISPLAY} Most Relevant Results (LLM-Filtered)")
if submit:
    if not search_results:
        st.info("No results found. Try a different query.")
    else:
        for i, r in enumerate(search_results, 1):
            with st.container():
                st.markdown(f"**Result #{i}**")
                st.markdown(f"<div style='background-color:#f5f5f5;padding:1em;border-radius:8px;'><b>Chunk:</b><br><span style='font-size:1.1em;color:#222;'>{r['text']}</span></div>", unsafe_allow_html=True)
                st.markdown("**Metadata:**")
                meta = r['metadata']
                meta_table = pd.DataFrame([meta]).T.rename(columns={0: 'Value'})
                st.table(meta_table)
                st.markdown(f"**Similarity score:** {r['score']:.4f}")

# --- TEST FUNCTION ---
def test_rag_pipeline():
    """Test function to check RAG pipeline: embedding, search, and retrieval."""
    test_query = "What is Enron's policy on trading?"
    st.write("\n---\n**RAG Pipeline Test**\n---")
    st.write(f"Test query: {test_query}")
    try:
        test_vec = embed_text(test_query, embed_model)
        D, I = index.search(test_vec, 3)
        st.write(f"Top 3 result indices: {I[0]}")
        st.write(f"Top 3 scores: {D[0]}")
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(meta_list):
                continue
            meta = meta_list[idx]
            chunk_text = meta.get('chunk_text', '[No text]')
            st.write(f"Result idx: {idx}, Score: {score}")
            st.write(f"Text: {chunk_text[:300]}")
            st.write(f"Metadata: {meta}")
    except Exception as e:
        st.error(f"RAG pipeline test failed: {e}")

if __name__ == "__main__":
    test_rag_pipeline() 