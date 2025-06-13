import faiss #Wyszukiwanie wektorowe- Przepisy
from sentence_transformers import SentenceTransformer #Zmiana tekstu i składników na wektory
import streamlit as st
import unicodedata

# Normalizacja dziwnych znaków
def fix_encoding(text):
    return unicodedata.normalize("NFKC", text)


#Przechowywanie wyników w pamięci
@st.cache_resource(show_spinner=False)

def init_vector_store(recipes: list): 
    
    model = SentenceTransformer('all-MiniLM-L6-v2') #Zmiana na vektory
    texts = [f"{r['title']}: {r.get('ingredients', '')}" for r in recipes]
    vectors = model.encode(texts, convert_to_numpy=True)
    
    #Tworzenie indexu faiss do trzymania wektorów
    index = faiss.IndexFlatL2(vectors.shape[1]) 
    index.add(vectors)
    return model, index #zwraca model i gotowy index do późniejszego użycia w wyszukiwaniu


#semantyczne wyszukiwanie podobnych przepisów na podstawie zapytania użytkownika
def query_similar_recipes(query: str, recipes: list, model, index, top_k: int = 5) -> list:

    q_vec = model.encode([query], convert_to_numpy=True) #Zmiana zapytania użytkownika na wektorową reprezentacje
    dists, ids = index.search(q_vec, top_k)
    return [recipes[i] for i in ids[0]]

#Tworzy gotowy tekst dla RAGu
def get_rag_context(query: str, recipes: list, model, index, top_k: int = 3) -> str:

    similar = query_similar_recipes(query, recipes, model, index, top_k)
    return "\n\n".join([
        f"Przepis: {r['title']}\nSkładniki: {r['ingredients']}\nKategoria: {r.get('category', 'brak')}"
        for r in similar
    ])
