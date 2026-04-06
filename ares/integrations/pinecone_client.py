from __future__ import annotations

import sys
from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - optional dependency
    Pinecone = None

from ares.utils.text_similarity import _get_model as _get_embedding_model


SEEDED_PATTERNS = [
    ("nit-rename", "consider renaming this variable", "downvote"),
    ("nit-docstring", "add a docstring here", "downvote"),
    ("nit-concise", "this could be more concise", "downvote"),
    ("nit-type-hints", "consider adding type hints", "downvote"),
    ("nit-unused-import", "unused import", "downvote"),
]


@dataclass(slots=True)
class PineconeMatch:
    id: str
    score: float
    label: str
    text: str


class PineconeClient:
    def __init__(self, api_key: str = "", index_name: str = "ares-comments", namespace: str = "default"):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.remote_index = self._build_remote()
        self._seeded = False
        self.seed_common_nits()

    def _build_remote(self):
        if not self.api_key or Pinecone is None:
            raise RuntimeError(
                "Pinecone is required. Install 'pinecone' and set PINECONE_API_KEY."
            )
        client = Pinecone(api_key=self.api_key)
        return client.Index(self.index_name)

    def seed_common_nits(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        self.upsert_feedback([
            {"id": item_id, "text": text, "label": label}
            for item_id, text, label in SEEDED_PATTERNS
        ])

    def upsert_feedback(self, items: list[dict]) -> None:
        vectors = []
        for item in items:
            vec = self._embed(item["text"])
            vectors.append({
                "id": item["id"],
                "values": vec,
                "metadata": {"text": item["text"], "label": item.get("label", "downvote")},
            })
        if vectors:
            try:
                self.remote_index.upsert(vectors=vectors, namespace=self.namespace)
            except Exception as exc:
                print(
                    f"[pinecone] Upsert failed ({len(vectors)} vectors): {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def query_similar(self, text: str, top_k: int = 5) -> list[dict]:
        query_vector = self._embed(text)
        results = self.remote_index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "label": match.get("metadata", {}).get("label", ""),
                "text": match.get("metadata", {}).get("text", ""),
            }
            for match in results.get("matches", [])
        ]

    def _embed(self, text: str) -> list[float]:
        """Encode text using sentence-transformers (all-mpnet-base-v2)."""
        import numpy as np
        model = _get_embedding_model()
        if model is None:
            raise RuntimeError(
                "Embedding model (sentence-transformers/all-mpnet-base-v2) is required "
                "but failed to load. Install sentence-transformers."
            )
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        norm = float(np.linalg.norm(emb)) + 1e-10
        vec = (emb / norm).tolist()
        return vec
