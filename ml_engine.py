"""
AI/ML-based Project Title Similarity Detection Engine
=====================================================

This module implements a multi-method NLP/ML pipeline for detecting
similarity between project titles.

Pipeline:
    1. NLP Preprocessing (NLTK) — Tokenization, Stopword Removal, Stemming
    2. Feature Extraction (scikit-learn) — TF-IDF Vectorization
    3. Multi-Method Similarity Scoring:
       - TF-IDF + Cosine Similarity  (semantic similarity)
       - Jaccard Similarity           (word overlap)
       - Sequence Matching            (structural similarity)
    4. Weighted Combined Score
    5. Threshold-based Decision (Accept / Reject)

Tech: Python, scikit-learn, NLTK, NumPy
"""

import re
import numpy as np
from difflib import SequenceMatcher

# Scikit-learn — ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK — NLP
import nltk

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ============================================================
# Similarity Engine Class
# ============================================================

class SimilarityEngine:
    """
    AI/ML-based Similarity Detection Engine.

    Uses a weighted ensemble of three similarity methods
    to determine whether a new project title is too similar
    to any existing title in the database.
    """

    # Titles with a combined score >= this are REJECTED
    SIMILARITY_THRESHOLD = 55  # percent

    # Ensemble weights (must sum to 1.0)
    TFIDF_WEIGHT    = 0.45
    JACCARD_WEIGHT  = 0.30
    SEQUENCE_WEIGHT = 0.25

    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

        # Domain-specific stopwords (common filler words in project titles)
        self.stop_words.update([
            'using', 'based', 'system', 'project', 'application',
            'app', 'development', 'design', 'implementation',
            'study', 'approach', 'method', 'technique', 'model',
        ])

    # --------------------------------------------------------
    # NLP Preprocessing
    # --------------------------------------------------------

    def preprocess(self, text):
        """
        Full NLP preprocessing pipeline.

        Steps:
            1. Lowercase normalisation
            2. Special character removal
            3. Tokenization (NLTK word_tokenize)
            4. Stopword removal (English + domain-specific)
            5. Stemming (Porter Stemmer)

        Returns:
            (processed_string, analysis_dict)
        """
        # Step 1 — lowercase
        lowered = text.lower().strip()

        # Step 2 — clean special characters, keep alphanumeric
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', lowered)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Step 3 — tokenize
        try:
            tokens = word_tokenize(cleaned)
        except LookupError:
            tokens = cleaned.split()

        # Step 4 — remove stopwords & very short tokens
        filtered = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 5 — stemming
        stemmed = [self.stemmer.stem(t) for t in filtered]

        analysis = {
            'original':        text,
            'lowered':         lowered,
            'cleaned':         cleaned,
            'tokens':          tokens,
            'after_stopwords': filtered,
            'stemmed':         stemmed,
            'processed_text':  ' '.join(stemmed),
        }
        return ' '.join(stemmed), analysis

    # --------------------------------------------------------
    # Individual Similarity Methods
    # --------------------------------------------------------

    def tfidf_cosine_similarity(self, text1, text2, corpus=None):
        """
        Method 1 — TF-IDF + Cosine Similarity.

        TF-IDF converts text into weighted feature vectors.
        Cosine similarity measures the angle between vectors.

        Returns: (score_0_to_1, details_dict)
        """
        docs = [text1, text2] + (corpus or [])

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                max_features=500,
            )
            tfidf_matrix = vectorizer.fit_transform(docs)
            features = vectorizer.get_feature_names_out().tolist()

            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Extract top features for each title
            v1 = tfidf_matrix[0].toarray()[0]
            v2 = tfidf_matrix[1].toarray()[0]

            top1 = sorted(
                [(features[i], round(float(v1[i]), 3))
                 for i in range(len(features)) if v1[i] > 0],
                key=lambda x: x[1], reverse=True,
            )[:6]

            top2 = sorted(
                [(features[i], round(float(v2[i]), 3))
                 for i in range(len(features)) if v2[i] > 0],
                key=lambda x: x[1], reverse=True,
            )[:6]

            common = list(
                set(f[0] for f in top1) & set(f[0] for f in top2)
            )

            return float(sim), {
                'score': round(sim * 100),
                'top_features_new':      top1,
                'top_features_existing': top2,
                'common_features':       common,
            }
        except ValueError:
            return 0.0, {
                'score': 0,
                'top_features_new': [],
                'top_features_existing': [],
                'common_features': [],
            }

    def jaccard_similarity(self, text1, text2):
        """
        Method 2 — Jaccard Similarity.

        J(A, B) = |A intersection B| / |A union B|

        Measures word-set overlap.

        Returns: (score_0_to_1, details_dict)
        """
        set1 = set(text1.split())
        set2 = set(text2.split())

        if not set1 or not set2:
            return 0.0, {'score': 0, 'common_words': [], 'all_words': []}

        intersection = set1 & set2
        union = set1 | set2
        sim = len(intersection) / len(union)

        return sim, {
            'score':        round(sim * 100),
            'common_words': list(intersection),
            'all_words':    list(union),
        }

    def sequence_similarity(self, text1, text2):
        """
        Method 3 — Sequence Matching (Ratcliff/Obershelp).

        Finds longest common subsequences.

        Returns: (score_0_to_1, details_dict)
        """
        sim = SequenceMatcher(None, text1, text2).ratio()
        return sim, {'score': round(sim * 100)}

    # --------------------------------------------------------
    # Combined Similarity (main entry point)
    # --------------------------------------------------------

    def check_title(self, new_title, existing_titles):
        """
        Check a new title against all existing titles.

        Returns:
            (is_rejected: bool, similar_list: list, analysis: dict)
        """
        if not existing_titles:
            _, new_analysis = self.preprocess(new_title)
            return False, [], self._build_analysis(new_title, new_analysis, [])

        # Preprocess new title
        new_processed, new_analysis = self.preprocess(new_title)

        # Preprocess all existing titles (used as TF-IDF corpus context)
        all_processed = []
        for t in existing_titles:
            p, _ = self.preprocess(t)
            all_processed.append(p)

        similar = []

        for idx, existing_title in enumerate(existing_titles):
            # --- exact match shortcut ---
            if new_title.strip().lower() == existing_title.strip().lower():
                similar.append({
                    'title':          existing_title,
                    'similarity':     100,
                    'tfidf_score':    100,
                    'jaccard_score':  100,
                    'sequence_score': 100,
                    'is_exact_match': True,
                    'common_keywords': [],
                })
                continue

            existing_processed = all_processed[idx]

            # Method 1
            tfidf_sim, tfidf_det = self.tfidf_cosine_similarity(
                new_processed, existing_processed, all_processed,
            )
            # Method 2
            jaccard_sim, jaccard_det = self.jaccard_similarity(
                new_processed, existing_processed,
            )
            # Method 3
            seq_sim, seq_det = self.sequence_similarity(
                new_processed, existing_processed,
            )

            # Weighted combination
            combined = (
                tfidf_sim   * self.TFIDF_WEIGHT +
                jaccard_sim * self.JACCARD_WEIGHT +
                seq_sim     * self.SEQUENCE_WEIGHT
            )
            combined_pct = round(combined * 100)

            if combined_pct >= self.SIMILARITY_THRESHOLD:
                similar.append({
                    'title':           existing_title,
                    'similarity':      combined_pct,
                    'tfidf_score':     round(tfidf_sim * 100),
                    'jaccard_score':   round(jaccard_sim * 100),
                    'sequence_score':  round(seq_sim * 100),
                    'is_exact_match':  False,
                    'common_keywords': tfidf_det.get('common_features', []),
                })

        similar.sort(key=lambda x: x['similarity'], reverse=True)
        is_rejected = len(similar) > 0
        analysis = self._build_analysis(new_title, new_analysis, similar)

        return is_rejected, similar, analysis

    # --------------------------------------------------------
    # Faculty: pairwise similarities for all titles
    # --------------------------------------------------------

    def pairwise_max_similarity(self, titles):
        """
        For each title, compute max combined similarity to any other title.
        Used in the faculty dashboard.

        Returns: dict mapping title -> (max_sim, most_similar_title)
        """
        if len(titles) < 2:
            return {t: (0, '') for t in titles}

        processed = {}
        for t in titles:
            p, _ = self.preprocess(t)
            processed[t] = p

        all_proc = list(processed.values())
        result = {}

        for i, t1 in enumerate(titles):
            best_sim = 0
            best_match = ''
            for j, t2 in enumerate(titles):
                if i == j:
                    continue
                tfidf_sim, _ = self.tfidf_cosine_similarity(
                    processed[t1], processed[t2], all_proc,
                )
                jaccard_sim, _ = self.jaccard_similarity(
                    processed[t1], processed[t2],
                )
                seq_sim, _ = self.sequence_similarity(
                    processed[t1], processed[t2],
                )
                combined = (
                    tfidf_sim   * self.TFIDF_WEIGHT +
                    jaccard_sim * self.JACCARD_WEIGHT +
                    seq_sim     * self.SEQUENCE_WEIGHT
                )
                pct = round(combined * 100)
                if pct > best_sim:
                    best_sim = pct
                    best_match = t2
            result[t1] = (best_sim, best_match)

        return result

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _build_analysis(self, title, preprocess_analysis, similar_list):
        """Build the analysis dict returned to the frontend."""
        return {
            'preprocessing': {
                'original': title,
                'steps': [
                    {'name': 'Lowercase',        'result': preprocess_analysis['lowered']},
                    {'name': 'Clean',             'result': preprocess_analysis['cleaned']},
                    {'name': 'Tokenize',          'result': preprocess_analysis['tokens']},
                    {'name': 'Remove Stopwords',  'result': preprocess_analysis['after_stopwords']},
                    {'name': 'Stemming (Porter)', 'result': preprocess_analysis['stemmed']},
                ],
                'final': preprocess_analysis['processed_text'],
            },
            'methods': [
                {
                    'name':        'TF-IDF + Cosine Similarity',
                    'weight':      f'{int(self.TFIDF_WEIGHT * 100)}%',
                    'description': 'Converts titles to weighted numerical vectors using '
                                   'Term Frequency-Inverse Document Frequency, then '
                                   'measures cosine of angle between vectors.',
                },
                {
                    'name':        'Jaccard Similarity',
                    'weight':      f'{int(self.JACCARD_WEIGHT * 100)}%',
                    'description': 'Measures word-set overlap: '
                                   '|intersection| / |union| of keyword sets.',
                },
                {
                    'name':        'Sequence Matching',
                    'weight':      f'{int(self.SEQUENCE_WEIGHT * 100)}%',
                    'description': 'Finds longest common subsequences using the '
                                   'Ratcliff/Obershelp pattern-matching algorithm.',
                },
            ],
            'threshold': self.SIMILARITY_THRESHOLD,
            'decision':  'REJECTED' if similar_list else 'ACCEPTED',
        }


# ============================================================
# Global Engine Instance
# ============================================================

engine = SimilarityEngine()
