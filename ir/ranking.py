"""Ranked retrieval: TF-IDF/cosine, unigram language model, DFR, passages, metrics."""
import math
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import tokenize

METHODS = ("tfidf", "lm", "dfr")


def _with_positions(items):
    for pos, item in enumerate(items, 1):
        item["position"] = pos
    return items


def tfidf_rank(query, docs):
    """Cosine similarity between TF-IDF vectors of query and documents (0..1)."""
    if not docs or not tokenize(query):
        return []
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    try:
        matrix = vec.fit_transform([query] + list(docs))
    except ValueError:  # empty vocabulary
        return []
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return [float(s) for s in scores]


def lm_scores(query, docs):
    """Query log-likelihood under a multinomial unigram model with Laplace smoothing.

    P(w|d) = (c(w,d) + 1) / (|d| + |V|);  score(q,d) = sum_{w in q} log P(w|d)
    V is the collection vocabulary plus query terms (so unseen terms stay valid).
    """
    q_terms = tokenize(query)
    if not q_terms or not docs:
        return []
    tokenized = [tokenize(d) for d in docs]
    vocab = {w for d in tokenized for w in d} | set(q_terms)
    v = len(vocab)
    out = []
    for toks in tokenized:
        counts, n = Counter(toks), len(toks)
        out.append(sum(math.log((counts[t] + 1) / (n + v)) for t in q_terms))
    return out


def dfr_scores(query, docs, c=1.0):
    """Divergence from Randomness, I(n_e)B2 model (Amati & van Rijsbergen; as in Terrier InexpB2).

    tfn = tf * log2(1 + c * avg_len / len)                  (normalisation 2)
    n_e = N * (1 - ((N-1)/N)^F)                              (expected doc frequency)
    w(t,d) = qtf * (F+1)/(df*(tfn+1)) * tfn * log2((N+1)/(n_e+0.5))
    Only terms occurring in the collection contribute; scores are >= 0.
    """
    q_counts = Counter(tokenize(query))
    if not q_counts or not docs:
        return []
    tokenized = [tokenize(d) for d in docs]
    n_docs = len(tokenized)
    lengths = [len(t) for t in tokenized]
    avg_len = (sum(lengths) / n_docs) or 1.0
    counters = [Counter(t) for t in tokenized]
    coll_freq = Counter()
    doc_freq = Counter()
    for cnt in counters:
        coll_freq.update(cnt)
        doc_freq.update(cnt.keys())
    scores = []
    for cnt, length in zip(counters, lengths):
        s = 0.0
        if length:
            for term, qtf in q_counts.items():
                tf = cnt.get(term, 0)
                if not tf:
                    continue
                f, df = coll_freq[term], doc_freq[term]
                tfn = tf * math.log2(1 + c * avg_len / length)
                ne = n_docs * (1 - ((n_docs - 1) / n_docs) ** f)
                s += qtf * (f + 1) / (df * (tfn + 1)) * tfn * math.log2((n_docs + 1) / (ne + 0.5))
        scores.append(s)
    return scores


_SCORERS = {"tfidf": tfidf_rank, "lm": lm_scores, "dfr": dfr_scores}


def rank(query, items, method="tfidf", text_key="title", min_score=None):
    """Rank dict items by `text_key` for `query`. Returns items + score + position.

    TF-IDF and DFR drop zero-score (non-matching) documents; the language model ranks all.
    """
    if method not in _SCORERS:
        raise ValueError("Unknown ranking method: %s" % method)
    items = list(items)
    scores = _SCORERS[method]( query, [i[text_key] for i in items])
    if not scores:
        return []
    out = []
    for item, s in zip(items, scores):
        if method != "lm" and s <= 0:
            continue
        row = dict(item)
        row["score"] = round(float(s), 6)
        out.append(row)
    out.sort(key=lambda r: r["score"], reverse=True)
    return _with_positions(out)


def split_passages(text, max_words=60):
    """Split text into passages on blank lines, then into chunks of <= max_words."""
    passages = []
    for block in [b.strip() for b in str(text or "").replace("\r", "").split("\n\n") if b.strip()]:
        words = block.split()
        for i in range(0, len(words), max_words):
            passages.append(" ".join(words[i:i + max_words]))
    return passages


def rank_passages(query, text, method="tfidf", max_words=60):
    passages = split_passages(text, max_words)
    items = [{"passage": p, "id": i + 1} for i, p in enumerate(passages)]
    return rank(query, items, method, text_key="passage")


# ---------------- evaluation (needs real relevance labels) ----------------

def precision_at_k(ranked_ids, relevant, k):
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked_ids[:k] if d in relevant) / k


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for d in ranked_ids[:k] if d in relevant) / len(relevant)


def average_precision(ranked_ids, relevant):
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def evaluate(judgments, items, method, k=3):
    """judgments: {query: set(ids)}; items: [{'id','title'}]. Returns per-query + MAP."""
    rows = []
    for query, relevant in judgments.items():
        relevant = set(relevant)
        if not relevant:
            continue
        ranked_ids = [r["id"] for r in rank(query, items, method)]
        rows.append({"query": query, "relevant": len(relevant),
                     "precision_at_k": round(precision_at_k(ranked_ids, relevant, k), 4),
                     "recall_at_k": round(recall_at_k(ranked_ids, relevant, k), 4),
                     "average_precision": round(average_precision(ranked_ids, relevant), 4)})
    if not rows:
        return None
    n = len(rows)
    return {"method": method, "k": k, "queries": rows,
            "mean_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 4),
            "mean_recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
            "MAP": round(sum(r["average_precision"] for r in rows) / n, 4)}
