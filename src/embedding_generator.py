from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

# Cargamos el modelo preentrenado de Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_corpus(repos):
    texts = []
    for repo in repos:
        desc = repo["description"] or ""
        topics = ", ".join(repo["topics"])
        text = f"{repo['name']} {desc} {topics}"
        texts.append(text.strip())
    return texts


def generate_embeddings(texts):
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    print("Cargando datos...")
    repos = load_data("data/repos_ml.json")
    corpus = build_corpus(repos)
    print("Generando embeddings con Sentence-BERT...")
    embeddings = generate_embeddings(corpus)

    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    np.save("embeddings/repos_ml.npy", embeddings)
    print(f"Embeddings generados y guardados correctamente: {embeddings.shape}")
