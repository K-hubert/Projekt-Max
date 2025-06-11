import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource(show_spinner=False)
def init_vector_store(recipes: list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{r['title']}: {r.get('ingredients','')}" for r in recipes]
    vectors = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return model, index


def query_similar_recipes(query: str, recipes: list, model, index, top_k: int=5) -> list:
    q_vec = model.encode([query], convert_to_numpy=True)
    dists, ids = index.search(q_vec, top_k)
    return [recipes[i] for i in ids[0]]