"""Shared text preprocessing: tokenisation, stop-words, light stemming."""
import re

STOPWORDS = frozenset("""a an and are as at be by for from has have in into is it its of on or that the
this to was were will with using use based via system systems application app new study towards toward
their there these those over under than then them they we you your our can may""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(word):
    """Very small suffix-stripping stemmer (deterministic, no external data)."""
    for suffix, repl in (("ational", "ate"), ("izing", "ize"), ("ing", ""), ("ies", "y"),
                         ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                continue
            return word[: len(word) - len(suffix)] + repl
    return word


def tokenize(text, remove_stopwords=True, do_stem=True):
    tokens = _TOKEN.findall(str(text or "").lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_stem:
        tokens = [stem(t) for t in tokens]
    return tokens
