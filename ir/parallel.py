"""Parallel execution helpers and a MapReduce-style demo (threads on ONE machine; not a cluster)."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .ranking import METHODS, rank
from .text import tokenize


def rank_all_methods(query, items, workers=3):
    """Run the independent TF-IDF / LM / DFR rankers concurrently."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {m: pool.submit(rank, query, items, m) for m in METHODS}
        return {m: f.result() for m, f in futures.items()}


def map_phase(doc):
    """Map: document -> (term, count) pairs."""
    return Counter(tokenize(doc))


def reduce_phase(partials):
    """Reduce: merge partial counts."""
    total = Counter()
    for p in partials:
        total.update(p)
    return total


def mapreduce_term_frequency(docs, workers=4, top_n=20):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        partials = list(pool.map(map_phase, docs))
    counts = reduce_phase(partials)
    total = sum(counts.values())
    terms = [{"term": t, "count": c, "tf": round(c / total, 6)} for t, c in counts.most_common(top_n)]
    return {"documents": len(docs), "total_terms": total, "unique_terms": len(counts), "terms": terms,
            "note": "Executed with a local thread pool on a single machine; not a distributed cluster."}
