import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


st.title("🔍 Búsqueda Semántica de Repositorios en GitHub")
st.write("Basado en Sentence-BERT + FAISS")


# Cargar modelo, datos y embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDINGS_PATH = "embeddings/repos_ml.npy"
DATA_PATH = "data/repos_ml.json"
embeddings = np.load(EMBEDDINGS_PATH)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    repos = json.load(f)


# Construir índice FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# Función para búsqueda
def search(query_text, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        repo = repos[idx]
        results.append(repo)
    return results


# UI en Streamlit
query = st.text_input("Describe tu repositorio ideal (tema, tecnología, etc.):")

if query:
    results = search(query)
    st.subheader(f"Resultados para: **{query}**")

    for i, repo in enumerate(results):
        st.markdown(f"### {i+1}. [{repo['full_name']}](https://github.com/{repo['full_name']})")
        st.write(f"⭐ Stars: {repo['stars']} | 🍴 Forks: {repo['forks']}")
        st.write(f"📝 Descripción: {repo['description']}")
        st.write(f"🏷️ Topics: {', '.join(repo['topics'])}")
        st.markdown("---")
    