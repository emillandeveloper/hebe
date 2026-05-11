"""
Embedder abstraction for memory_store.

Default impl: LocalEmbedder using sentence-transformers (multilingual MiniLM).
The Embedder protocol exists so we can swap to OpenAI or a bigger local model
later without touching memory_store.

IMPORTANT — model migration:
If you ever change DEFAULT_MODEL_NAME, all stored vectors become incomparable
with new queries (different geometry). Workflow for changing:
  1. Update DEFAULT_MODEL_NAME below.
  2. Run scripts/reembed_memory.py (TODO) which iterates memory_chunks where
     embedding_model != current and regenerates the vector column.
The model name is stored alongside each vector for exactly this reason.
"""

from __future__ import annotations

from typing import Protocol, Optional
import numpy as np


# =============================================================================
# Protocol
# =============================================================================

class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


# =============================================================================
# Default impl: local sentence-transformers
# =============================================================================

# Multilingual MiniLM — buena calidad para español, ~120 MB, va bien en CPU,
# mejor en GPU. Si más adelante quieres más calidad y tienes VRAM libre,
# 'BAAI/bge-m3' es un upgrade natural (1024 dim vs 384, más pesado).
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class LocalEmbedder:
    """
    sentence-transformers wrapper. normalize_embeddings=True garantiza que
    cosine similarity == dot product, lo que simplifica el ranking en search_chunks.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        # Import diferido para que el módulo se pueda importar sin tener
        # sentence-transformers instalado (útil para tests con fake embedder).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)


# =============================================================================
# Singleton accessor (lazy init para no cargar el modelo al importar)
# =============================================================================

_default_embedder: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    """Devuelve el embedder por defecto, cargándolo en la primera llamada."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = LocalEmbedder()
    return _default_embedder


def set_default_embedder(embedder: Embedder) -> None:
    """Inyecta otro embedder. Útil en tests o para swap a OpenAI más adelante."""
    global _default_embedder
    _default_embedder = embedder


# =============================================================================
# Helpers de serialización
# =============================================================================

def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serialize a numpy float32 array para guardar como BLOB en SQLite."""
    return vec.astype(np.float32).tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes desde SQLite BLOB a un numpy array."""
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)
