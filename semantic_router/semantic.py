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
        self.logger = setup_logger()

        if self.use_faiss:
            # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS index initialized with IP metric")
        else:
            logger.info("Using numpy-based similarity search")

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length for cosine similarity."""
        if embeddings is None or len(embeddings) == 0:
            return np.zeros((1, self.dimension))
        
        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def build_index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Build the search index from a list of documents with optional metadata."""
        if not corpus:
            logger.warning("Empty corpus provided to build_index")
            return

        # Validate corpus
        valid_corpus = []
        for i, text in enumerate(corpus):
            if not text or not isinstance(text, str):
                logger.warning(f"Skipping invalid text at index {i}")
                continue
            text = text.strip()
            if text:
                valid_corpus.append(text)
            else:
                logger.warning(f"Skipping empty text at index {i}")

        if not valid_corpus:
            logger.error("No valid texts in corpus after validation")
            return

        self.corpus = valid_corpus
        if metadata:
            self.metadata = {i: meta for i, meta in enumerate(metadata)}

        try:
            # Get embeddings and normalize them
            logger.info(f"Getting embeddings for {len(valid_corpus)} texts")
            embeddings = embedding_model.get_batch_embeddings(valid_corpus)
            
            # Validate embeddings
            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to get embeddings")
                return
                
            if embeddings.shape[0] != len(valid_corpus):
                logger.error(f"Embedding shape mismatch: got {embeddings.shape[0]}, expected {len(valid_corpus)}")
                return
                
            normalized_embeddings = self.normalize_embeddings(embeddings)
            
            if self.use_faiss:
                self.index.reset()  # Clear existing index
                emb_np = normalized_embeddings.astype('float32')
                self.index.add(emb_np)
                logger.info(f"FAISS index built with {len(valid_corpus)} documents")
            else:
                self.embeddings = normalized_embeddings
                logger.info(f"Stored {len(valid_corpus)} embeddings for numpy-based search")
                
            # Verify index
            if self.use_faiss:
                if self.index.ntotal != len(valid_corpus):
                    logger.error(f"FAISS index size mismatch: got {self.index.ntotal}, expected {len(valid_corpus)}")
            else:
                if self.embeddings.shape[0] != len(valid_corpus):
                    logger.error(f"Numpy embeddings size mismatch: got {self.embeddings.shape[0]}, expected {len(valid_corpus)}")
                    
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)
            raise

    def query(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        filter_fn: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Return top_k similar documents above a similarity threshold with optional filtering."""
        results: List[Dict[str, Any]] = []
        
        if not query or not self.corpus:
            self.logger.warning("Empty query or corpus")
            return results

        try:
            # Get and normalize query embedding
            query_emb = embedding_model.get_batch_embeddings([query])[0]
            if query_emb is None or np.all(query_emb == 0):
                self.logger.warning("Failed to get valid embedding for query")
                return results
                
            query_emb = self.normalize_embeddings(query_emb.reshape(1, -1))[0]

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
            
            # Enhanced logging for debugging
            if results:
                self.logger.info(f"Found {len(results)} matches above threshold {threshold}")
                self.logger.info(f"Query: '{query}'")
                for r in results[:3]:  # Log top 3 matches
                    self.logger.info(
                        f"Match {r['index']}: '{r['text'][:50]}...' "
                        f"(score: {r['score']:.3f}, "
                        f"category: {r.get('category', 'unknown')})"
                    )
                    
                # Log potential issues
                if len(results) > 1:
                    score_diff = results[0]['score'] - results[1]['score']
                    if score_diff < 0.1:
                        self.logger.warning(
                            f"Close scores detected: {results[0]['score']:.3f} vs "
                            f"{results[1]['score']:.3f} (diff: {score_diff:.3f})"
                        )
            else:
                self.logger.info(f"No matches found above threshold {threshold}")
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}", exc_info=True)
        
        return results[:top_k]

    def add_to_index(self, new_texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add new documents to the existing index with optional metadata."""
        if not new_texts:
            return

        try:
            start_idx = len(self.corpus)
            
            # Update metadata
            if metadata:
                for i, meta in enumerate(metadata):
                    self.metadata[start_idx + i] = meta

            # Get and normalize new embeddings
            new_embeddings = embedding_model.get_batch_embeddings(new_texts)
            normalized_embeddings = self.normalize_embeddings(new_embeddings)

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
        except Exception as e:
            logger.error(f"Error adding to index: {e}", exc_info=True)
            raise

    def remove_from_index(self, indices: List[int]) -> None:
        """Remove documents from the index by their indices."""
        if not indices:
            return

        try:
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
                normalized_embeddings = self.normalize_embeddings(remaining_embeddings)
                self.index.reset()
                self.index.add(normalized_embeddings.astype('float32'))
            else:
                # For numpy, we can just delete rows
                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices] = False
                self.embeddings = self.embeddings[mask]

            logger.info(f"Removed {len(indices)} documents from index")
        except Exception as e:
            logger.error(f"Error removing from index: {e}", exc_info=True)
            raise