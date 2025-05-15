import numpy as np
from typing import List, Dict, Any
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

        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("FAISS index initialized successfully")
        else:
            logger.info("Using numpy-based similarity search")

    def build_index(self, corpus: List[str]) -> None:
        """Build the search index from a list of documents."""
        self.corpus = corpus
        embeddings = embedding_model.get_batch_embeddings(corpus)
        
        if self.use_faiss:
            emb_np = embeddings.astype('float32')
            faiss.normalize_L2(emb_np)
            self.index.add(emb_np)
            logger.info(f"FAISS index built with {len(corpus)} documents")
        else:
            self.embeddings = embeddings
            logger.info(f"Stored {len(corpus)} embeddings for numpy-based search")

    def query(self, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Return top_k similar documents above a similarity threshold."""
        results: List[Dict[str, Any]] = []
        try:
            query_emb = embedding_model.get_batch_embeddings([query])[0]

            if self.use_faiss:
                q_np = np.array([query_emb]).astype('float32')
                faiss.normalize_L2(q_np)
                distances, indices = self.index.search(q_np, top_k)
                similarities = 1 - distances[0] / 2
                for idx, score in zip(indices[0], similarities):
                    if score >= threshold:
                        results.append({'text': self.corpus[idx], 'score': float(score), 'index': int(idx)})
            else:
                sims = np.dot(self.embeddings, query_emb) / (
                    np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
                )
                top_idxs = np.argsort(sims)[-top_k:][::-1]
                for idx in top_idxs:
                    score = float(sims[idx])
                    if score >= threshold:
                        results.append({'text': self.corpus[idx], 'score': score, 'index': int(idx)})
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
        return results

    def add_to_index(self, new_texts: List[str]) -> None:
        """Add new documents to the existing index."""
        if not new_texts:
            return
        new_embeddings = embedding_model.get_batch_embeddings(new_texts)

        if self.use_faiss:
            emb_np = new_embeddings.astype('float32')
            faiss.normalize_L2(emb_np)
            self.index.add(emb_np)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.corpus.extend(new_texts)
        logger.info(f"Added {len(new_texts)} new documents to index")

    def remove_from_index(self, text: str) -> None:
        """Remove a document from the index and rebuild if necessary."""
        if text in self.corpus:
            self.corpus.remove(text)
            if self.corpus:
                self.build_index(self.corpus)
            else:
                self.index = None