import os
import numpy as np
from typing import List, Dict, Any, Optional
from chromadb.utils import embedding_functions
from utils.logger import setup_logger

logger = setup_logger()

# Try to import FAISS, but don't fail if it's not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Falling back to numpy-based similarity search.")
    FAISS_AVAILABLE = False

class SemanticSearch:
    def __init__(self, use_faiss: bool = True):
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.corpus = []
        self.dimension = 1536  # text-embedding-ada-002 dimension
        
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("FAISS index initialized successfully")
        else:
            logger.info("Using numpy-based similarity search")

    def build_index(self, corpus: List[str]) -> None:
        """Build search index from corpus."""
        self.corpus = corpus
        embeddings = self.ef(corpus)
        
        if self.use_faiss:
            # Convert to numpy array and normalize for L2 distance
            embeddings_np = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_np)
            self.index.add(embeddings_np)
            logger.info(f"FAISS index built with {len(corpus)} documents")
        else:
            # Store embeddings for numpy-based search
            self.embeddings = np.array(embeddings)
            logger.info(f"Stored {len(corpus)} embeddings for numpy-based search")

    def query(self, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search for similar documents in the corpus."""
        try:
            query_embedding = self.ef([query])[0]
            
            if self.use_faiss:
                # Convert query to numpy and normalize
                query_np = np.array([query_embedding]).astype('float32')
                faiss.normalize_L2(query_np)
                
                # Search using FAISS
                distances, indices = self.index.search(query_np, top_k)
                
                # Convert distances to similarities (1 - normalized distance)
                similarities = 1 - distances[0] / 2
                
                results = []
                for idx, score in zip(indices[0], similarities):
                    if score >= threshold:
                        results.append({
                            'text': self.corpus[idx],
                            'score': float(score),
                            'index': int(idx)
                        })
                return results
            else:
                # Numpy-based search using cosine similarity
                similarities = np.dot(self.embeddings, query_embedding) / (
                    np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    score = float(similarities[idx])
                    if score >= threshold:
                        results.append({
                            'text': self.corpus[idx],
                            'score': score,
                            'index': int(idx)
                        })
                return results
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def add_to_index(self, new_texts: List[str]) -> None:
        """Add new documents to the index."""
        if not new_texts:
            return
            
        new_embeddings = self.ef(new_texts)
        
        if self.use_faiss:
            # Convert to numpy and normalize
            new_embeddings_np = np.array(new_embeddings).astype('float32')
            faiss.normalize_L2(new_embeddings_np)
            self.index.add(new_embeddings_np)
        else:
            # Append to existing embeddings
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        self.corpus.extend(new_texts)
        logger.info(f"Added {len(new_texts)} new documents to index")

    def remove_from_index(self, text: str):
        """Remove a text from the index (requires rebuilding)."""
        try:
            if text in self.corpus:
                idx = self.corpus.index(text)
                self.corpus.pop(idx)
                
                # Rebuild index
                if len(self.corpus) > 0:
                    self.build_index(self.corpus)
                else:
                    self.index = None
                    self.embeddings = None
                    
        except Exception as e:
            logger.error(f"Error removing from index: {e}")
            raise 