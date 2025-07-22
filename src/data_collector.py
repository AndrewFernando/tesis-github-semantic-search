import requests
import json
import os
from tqdm import tqdm



GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}



def get_repositories(query, pages=2):
    all_repos = []
    for page in range(1, pages + 1):
        url = f"https://api.github.com/search/repositories?q={query}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        data = response.json()

        for repo in data.get('items', []):
            all_repos.append({
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo["description"],
                "topics": repo.get("topics", []),
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "language": repo["language"]
            })

    return all_repos


if __name__ == "__main__":
    print("Recolectando datos de GitHub...")
    repos = get_repositories("machine learning", pages=5)
    with open("data/repos_ml.json", "w", encoding="utf-8") as f:
        json.dump(repos, f, ensure_ascii=False, indent=2)
    print(f"Repositorios guardados: {len(repos)}")
