"""Link-based ranking: PageRank over a graph built from real title similarities."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import tokenize


def similarity_graph(titles, threshold=0.15):
    """Weighted undirected adjacency matrix: edge i-j if TF-IDF cosine >= threshold."""
    n = len(titles)
    if n == 0:
        return np.zeros((0, 0))
    try:
        m = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(titles)
        sim = cosine_similarity(m)
    except ValueError:
        return np.zeros((n, n))
    np.fill_diagonal(sim, 0.0)
    sim[sim < threshold] = 0.0
    return sim


def pagerank(adj, damping=0.85, tol=1e-10, max_iter=200):
    """Weighted PageRank by power iteration; dangling/disconnected nodes spread uniformly."""
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    out = adj.sum(axis=1)
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangling = rank[out == 0].sum()
        contrib = np.zeros(n)
        nz = out > 0
        contrib += (adj[nz] / out[nz][:, None]).T @ rank[nz]
        new = (1 - damping) / n + damping * (contrib + dangling / n)
        if np.abs(new - rank).sum() < tol:
            rank = new
            break
        rank = new
    return rank / rank.sum()


def rank_projects(items, threshold=0.15, text_key="title"):
    """items: [{'id','title'}] -> ranked list with pagerank score, degree, position."""
    items = [i for i in items if i.get(text_key, "").strip()]
    if not items:
        return {"nodes": 0, "edges": 0, "results": []}
    adj = similarity_graph([i[text_key] for i in items], threshold)
    pr = pagerank(adj)
    order = sorted(range(len(items)), key=lambda i: pr[i], reverse=True)
    results = [dict(items[i], pagerank=round(float(pr[i]), 6), links=int((adj[i] > 0).sum()),
                    position=pos) for pos, i in enumerate(order, 1)]
    return {"nodes": len(items), "edges": int((adj > 0).sum() // 2), "threshold": threshold,
            "results": results}
