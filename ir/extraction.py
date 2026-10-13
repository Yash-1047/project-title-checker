"""Information extraction (keywords, phrases, entities, JSON-LD) and semantic-search demo."""
import json
import re
from collections import Counter

from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import STOPWORDS, tokenize

_ENTITY_PATTERNS = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("url", re.compile(r"https?://[^\s<>\"]+")),
    ("year", re.compile(r"\b(?:19|20)\d{2}\b")),
    ("acronym", re.compile(r"\b[A-Z]{2,6}s?\b")),
    ("proper_noun", re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")),
)


def extract_keywords(text, top_n=10):
    """Top terms by frequency (stop-words removed, light stemming; shown as most frequent surface form)."""
    surface = {}
    counts = Counter()
    for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", str(text or "")):
        lw = w.lower()
        if lw in STOPWORDS or len(lw) < 3:
            continue
        key = tokenize(lw)[0] if tokenize(lw) else lw
        counts[key] += 1
        surface.setdefault(key, Counter())[lw] += 1
    return [{"keyword": surface[k].most_common(1)[0][0], "count": c} for k, c in counts.most_common(top_n)]


def extract_phrases(text, top_n=10):
    """Candidate phrases: runs of 2-4 consecutive non-stopwords, ranked by frequency (>=1)."""
    phrases = Counter()
    for sentence in re.split(r"[.!?\n;:,()]+", str(text or "").lower()):
        run = []
        for w in re.findall(r"[a-z][a-z0-9-]+", sentence) + [None]:
            if w and w not in STOPWORDS and len(w) > 2:
                run.append(w)
                continue
            for n in (2, 3, 4):
                for i in range(len(run) - n + 1):
                    phrases[" ".join(run[i:i + n])] += 1
            run = []
    return [{"phrase": p, "count": c} for p, c in sorted(phrases.items(), key=lambda x: (-x[1], -len(x[0])))[:top_n]]


def extract_entities(text):
    """Regex-based entities (lightweight, no model): email, url, year, acronym, proper noun."""
    found, seen = [], set()
    text = str(text or "")
    for etype, pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            if (val.lower(), etype) in seen or (etype == "acronym" and val in {"I", "A"}):
                continue
            seen.add((val.lower(), etype))
            found.append({"text": val, "type": etype})
    return found


def extract_jsonld(html):
    """Return the list of JSON-LD objects (flattening @graph) found in <script type=application/ld+json>."""
    out = []
    for tag in BeautifulSoup(str(html or ""), "html.parser").find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except ValueError:
            continue
        for obj in (data if isinstance(data, list) else [data]):
            if isinstance(obj, dict):
                out.extend(obj["@graph"] if isinstance(obj.get("@graph"), list) else [obj])
    return [o for o in out if isinstance(o, dict)]


def normalise_jsonld(objects):
    """Flatten JSON-LD objects to {type, name, description, url, author}."""
    def first(v):
        v = v[0] if isinstance(v, list) and v else v
        return v.get("name") or v.get("@id") if isinstance(v, dict) else v
    return [{"type": first(o.get("@type")), "name": first(o.get("name") or o.get("headline")),
             "description": first(o.get("description")), "url": first(o.get("url")),
             "author": first(o.get("author"))} for o in objects]


def extract_all(text="", html=""):
    """Combine everything into one consistent structure."""
    if html and html.strip():
        soup = BeautifulSoup(html, "html.parser")
        for n in soup(["script", "style", "noscript"]):
            n.decompose()
        text = (text + " " + soup.get_text(" ", strip=True)).strip()
    return {"keywords": extract_keywords(text), "phrases": extract_phrases(text),
            "entities": extract_entities(text), "jsonld": normalise_jsonld(extract_jsonld(html)),
            "characters": len(text)}


def semantic_search(query, items, text_key="title", top_n=10):
    """LSA demo: TF-IDF -> truncated SVD -> cosine in latent space. Needs >= 3 documents."""
    items = [i for i in items if i.get(text_key, "").strip()]
    if len(items) < 3:
        return {"error": "Semantic search needs at least 3 project titles in the database.", "results": []}
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    try:
        m = vec.fit_transform([i[text_key] for i in items])
    except ValueError:
        return {"error": "No searchable terms in the stored titles.", "results": []}
    k = min(50, m.shape[0] - 1, m.shape[1] - 1)
    if k < 2:
        return {"error": "Not enough vocabulary diversity for semantic analysis.", "results": []}
    svd = TruncatedSVD(n_components=k, random_state=0)
    docs = svd.fit_transform(m)
    q = svd.transform(vec.transform([query]))
    if not q.any():
        return {"error": None, "results": []}
    scores = cosine_similarity(q, docs).ravel()
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    res = [dict(items[i], score=round(float(scores[i]), 4), position=p)
           for p, i in enumerate(order, 1) if scores[i] > 0][:top_n]
    return {"error": None, "components": int(k), "results": res}
