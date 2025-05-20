import numpy as np
from typing import List, Dict, Any, Optional, Callable
from utils.logger import setup_logger
from embedding_model.embedding import embedding_model

logger = setup_logger()

# Try to import FAISS, but continue with numpy fallback if unavailable
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Falling back to numpy-based similarity search.")
    FAISS_AVAILABLE = False

class SemanticSearch:
    """Semantic search using FAISS or numpy with embeddings from EmbeddingModel."""
    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.corpus: List[str] = []
        self.dimension = 1536  # embedding dimension
        self.embeddings = None
        self.metadata = {}  # Store additional metadata for each document

        if self.use_faiss:
            # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index initialized with IP metric")
        else:
            logger.info("Using numpy-based similarity search")

    def build_index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Build the search index from a list of documents with optional metadata."""
        self.corpus = corpus
        if metadata:
            self.metadata = {i: meta for i, meta in enumerate(metadata)}

        # Get embeddings and normalize them
        embeddings = embedding_model.get_batch_embeddings(corpus)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        if self.use_faiss:
            self.index.reset()  # Clear existing index
            emb_np = normalized_embeddings.astype('float32')
            self.index.add(emb_np)
            logger.info(f"FAISS index built with {len(corpus)} documents")
        else:
            self.embeddings = normalized_embeddings
            logger.info(f"Stored {len(corpus)} embeddings for numpy-based search")

    def query(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.6,
        filter_fn: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Return top_k similar documents above a similarity threshold with optional filtering."""
        results: List[Dict[str, Any]] = []
        try:
            # Get and normalize query embedding
            query_emb = embedding_model.get_batch_embeddings([query])[0]
            query_emb = query_emb / np.linalg.norm(query_emb)

            if self.use_faiss:
                q_np = query_emb.astype('float32').reshape(1, -1)
                distances, indices = self.index.search(q_np, top_k)
                similarities = (distances[0] + 1) / 2  # Convert distance to similarity score
                
                for idx, score in zip(indices[0], similarities):
                    if score >= threshold:
                        result = {
                            'text': self.corpus[idx],
                            'score': float(score),
                            'index': int(idx)
                        }
                        if idx in self.metadata:
                            result.update(self.metadata[idx])
                        if not filter_fn or filter_fn(result):
                            results.append(result)
            else:
                # Compute similarities with numpy
                sims = np.dot(self.embeddings, query_emb)
                top_idxs = np.argsort(sims)[-top_k:][::-1]
                
                for idx in top_idxs:
                    score = float(sims[idx])
                    if score >= threshold:
                        result = {
                            'text': self.corpus[idx],
                            'score': score,
                            'index': int(idx)
                        }
                        if idx in self.metadata:
                            result.update(self.metadata[idx])
                        if not filter_fn or filter_fn(result):
                            results.append(result)

            # Sort results by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
        
        return results[:top_k]

    def add_to_index(self, new_texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add new documents to the existing index with optional metadata."""
        if not new_texts:
            return

        start_idx = len(self.corpus)
        
        # Update metadata
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata[start_idx + i] = meta

        # Get and normalize new embeddings
        new_embeddings = embedding_model.get_batch_embeddings(new_texts)
        normalized_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1)[:, np.newaxis]

        if self.use_faiss:
            emb_np = normalized_embeddings.astype('float32')
            self.index.add(emb_np)
        else:
            if self.embeddings is None:
                self.embeddings = normalized_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
        
        self.corpus.extend(new_texts)
        logger.info(f"Added {len(new_texts)} new documents to index")

    def remove_from_index(self, indices: List[int]) -> None:
        """Remove documents from the index by their indices."""
        if not indices:
            return

        # Sort indices in descending order to avoid shifting issues
        indices = sorted(indices, reverse=True)
        
        # Remove from corpus and metadata
        for idx in indices:
            if idx < len(self.corpus):
                self.corpus.pop(idx)
                if idx in self.metadata:
                    del self.metadata[idx]

        if self.use_faiss:
            # For FAISS, we need to rebuild the index
            remaining_embeddings = embedding_model.get_batch_embeddings(self.corpus)
            normalized_embeddings = remaining_embeddings / np.linalg.norm(remaining_embeddings, axis=1)[:, np.newaxis]
            self.index.reset()
            self.index.add(normalized_embeddings.astype('float32'))
        else:
            # For numpy, we can just delete rows
            mask = np.ones(len(self.embeddings), dtype=bool)
            mask[indices] = False
            self.embeddings = self.embeddings[mask]

        logger.info(f"Removed {len(indices)} documents from index")