"""Recommender systems: content-based, knowledge-based, and collaborative filtering."""
import numpy as np

from .ranking import tfidf_rank
from .text import tokenize


def content_based(query, items, top_k=3, exclude_id=None, text_key="title"):
    """Top-k most similar projects by TF-IDF + cosine; the current project is excluded."""
    pool = [i for i in items if i.get("id") != exclude_id and i.get(text_key, "").strip()]
    scores = tfidf_rank(query, [i[text_key] for i in pool])
    ranked = sorted(((s, i) for s, i in zip(scores, pool) if s > 0), key=lambda x: -x[0])[:top_k]
    return [dict(i, score=round(s, 4), similarity_percent=round(s * 100, 2), position=p)
            for p, (s, i) in enumerate(ranked, 1)]


def knowledge_based(items, status=None, include=(), exclude=(), min_words=0, text_key="title"):
    """Filter by explicit user-chosen criteria; each result states which rules it satisfied."""
    include = [t for w in include for t in tokenize(w)]
    exclude = [t for w in exclude for t in tokenize(w)]
    results = []
    for item in items:
        toks = set(tokenize(item[text_key]))
        reasons = []
        if status and item.get("status") != status:
            continue
        if status:
            reasons.append("status = %s" % status)
        if include:
            hit = [t for t in include if t in toks]
            if len(hit) != len(include):
                continue
            reasons.append("contains: " + ", ".join(include))
        if exclude and any(t in toks for t in exclude):
            continue
        if exclude:
            reasons.append("avoids: " + ", ".join(exclude))
        if len(item[text_key].split()) < min_words:
            continue
        if min_words:
            reasons.append("at least %d words" % min_words)
        results.append(dict(item, reasons=reasons))
    return results


def collaborative(ratings, user_id, top_k=3, min_users=3, min_ratings=6):
    """User-based CF. ratings: [(user_id, project_id, rating)]. Returns (recs, message)."""
    users = sorted({r[0] for r in ratings})
    if len(ratings) < min_ratings or len(users) < min_users:
        return [], ("Not enough rating data yet (need at least %d ratings from %d users; have %d from %d)."
                    % (min_ratings, min_users, len(ratings), len(users)))
    if user_id not in users:
        return [], "Rate a few projects first to receive collaborative recommendations."
    projects = sorted({r[1] for r in ratings})
    uidx = {u: i for i, u in enumerate(users)}
    pidx = {p: i for i, p in enumerate(projects)}
    mat = np.zeros((len(users), len(projects)))
    for u, p, r in ratings:
        mat[uidx[u], pidx[p]] = r
    me = mat[uidx[user_id]]
    norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(me) or 1)
    sims = np.divide(mat @ me, norms, out=np.zeros(len(users)), where=norms > 0)
    sims[uidx[user_id]] = 0
    if sims.sum() <= 0:
        return [], "No similar users found yet."
    pred = (sims @ mat) / (np.abs(sims).sum() or 1)
    recs = [{"project_id": p, "predicted_rating": round(float(pred[j]), 3)}
            for p, j in pidx.items() if me[j] == 0 and pred[j] > 0]
    recs.sort(key=lambda r: -r["predicted_rating"])
    return recs[:top_k], ("" if recs else "You have already rated every project others rated.")
