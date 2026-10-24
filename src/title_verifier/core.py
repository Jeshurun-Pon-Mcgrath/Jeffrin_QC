"""Explainable retrieval and pair classification for educational title screening."""
import re
import unicodedata
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Title must be text.")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))
    if not text:
        raise ValueError("Enter a title containing letters or numbers.")
    if len(text) > 500:
        raise ValueError("Title must contain at most 500 normalized characters.")
    return text


def features(a, b):
    """Symmetric pair features; no fitted vocabulary or registry labels required."""
    a, b = normalize(a), normalize(b)
    x, y = set(a.split()), set(b.split())
    grams = lambda s: set(s[i:i+3] for i in range(max(1, len(s)-2)))
    u, v = grams(a), grams(b)
    return [float(a == b), len(x & y)/len(x | y),
            SequenceMatcher(None, a, b, autojunk=False).ratio(),
            min(len(a), len(b))/max(len(a), len(b)), len(u & v)/len(u | v)]


class TitleVerifier:
    def __init__(self, registry):
        required = {"title_id", "title"}
        if not required.issubset(registry.columns):
            raise ValueError("Registry requires title_id and title columns.")
        if registry.empty or registry.title_id.isna().any() or registry.title_id.duplicated().any():
            raise ValueError("Registry needs nonempty, unique title IDs.")
        self.registry = registry.copy().reset_index(drop=True)
        self.normalized = self.registry.title.map(normalize)
        if self.normalized.duplicated().any():
            raise ValueError("Registry contains duplicate normalized titles; clean it first.")
        self.word = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        self.w = self.word.fit_transform(self.normalized)
        self.c = self.char.fit_transform(self.normalized)
        self.model = None
        self.threshold = 0.5

    def fit_pairs(self, pairs):
        if not {0, 1}.issubset(set(pairs.label)):
            raise ValueError("Training needs both positive and negative pairs.")
        self.model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        self.model.fit([features(a, b) for a, b in zip(pairs["query"], pairs.candidate)], pairs.label)
        return self

    def pair_scores(self, pairs):
        if self.model is None:
            raise RuntimeError("Train the pair classifier first.")
        return self.model.predict_proba([features(a, b) for a, b in zip(pairs["query"], pairs.candidate)])[:, 1]

    def search(self, title, top_k=5):
        q = normalize(title)
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        score = .4*cosine_similarity(self.word.transform([q]), self.w)[0] + .6*cosine_similarity(self.char.transform([q]), self.c)[0]
        ids = np.argsort(-score, kind="stable")[:top_k]
        # Always prioritize normalized exact matches, even for a registry with tied scores.
        exact_ids = np.flatnonzero(self.normalized.to_numpy() == q)
        ids = list(dict.fromkeys([*exact_ids, *ids]))[:top_k]
        out = self.registry.iloc[ids].copy()
        out["similarity"] = score[ids]
        out["exact_match"] = self.normalized.iloc[ids].to_numpy() == q
        if self.model is not None:
            out["pair_score"] = self.model.predict_proba([features(q, t) for t in out.title])[:, 1]
            out["decision"] = ["EXACT MATCH" if e else "REVIEW" if p >= self.threshold else "LOW SIMILARITY" for e, p in zip(out.exact_match, out.pair_score)]
        else:
            out["decision"] = np.where(out.exact_match, "EXACT MATCH", "REVIEW (UNTRAINED)")
        return out.reset_index(drop=True)
