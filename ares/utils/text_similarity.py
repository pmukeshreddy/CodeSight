from __future__ import annotations

import threading
from difflib import SequenceMatcher

_lock = threading.Lock()
_model = None
_model_loaded = False


def _get_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    with _lock:
        if _model_loaded:
            return _model
        _model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-mpnet-base-v2")
        except Exception:
            _model = None
    return _model


def text_similarity(a: str, b: str) -> float:
    """Semantic similarity in [0, 1]. Uses cosine of sentence embeddings when available,
    falls back to SequenceMatcher character similarity."""
    if not a or not b:
        return 0.0
    model = _get_model()
    if model is not None:
        try:
            import numpy as np
            embs = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            embs = embs / norms
            return float(np.dot(embs[0], embs[1]))
        except Exception:
            pass
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def batch_similarities(query: str, candidates: list[str]) -> list[float]:
    """Compute similarity between query and each candidate in one encoder pass."""
    if not query or not candidates:
        return [0.0] * len(candidates)
    model = _get_model()
    if model is not None:
        try:
            import numpy as np
            all_texts = [query] + candidates
            embs = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            embs = embs / norms
            q = embs[0]
            return [float(np.dot(q, embs[i + 1])) for i in range(len(candidates))]
        except Exception:
            pass
    return [SequenceMatcher(None, query.lower(), c.lower()).ratio() for c in candidates]
