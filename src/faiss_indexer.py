import faiss
import numpy as np
import json


# Cargar embeddings y datos
EMBEDDINGS_PATH = "embeddings/repos_ml.npy"
DATA_PATH = "data/repos_ml.json"

embeddings = np.load(EMBEDDINGS_PATH)
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    repos = json.load(f)


# Crear índice FAISS (L2 euclídeo rápido y simple)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def search(query_text, model, top_k=5):
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding, top_k)

    for i, idx in enumerate(indices[0]):
        repo = repos[idx]
        print(f"\nTop {i+1}: {repo['full_name']}")
        print(f"Description: {repo['description']}")
        print(f"Topics: {repo['topics']}")
        print(f"Language: {repo['language']}, Stars: {repo['stars']}")
        print(f"Distance: {distances[0][i]:.4f}")


# Prueba rápida (para ver que funciona)
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = input("\nDescribe el repositorio que buscas (ej: librería de machine learning, app móvil, etc):\n> ")
    search(query, model)
