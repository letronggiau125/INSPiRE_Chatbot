import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from functools import lru_cache
from utils.logger import setup_logger
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import os
import pickle
from chromadb.utils import embedding_functions

# Set seed for consistent language detection
DetectorFactory.seed = 0
logger = setup_logger()

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

class EmbeddingModel:
    """Enhanced embedding model with sophisticated multilingual support and caching."""
    
    def __init__(
        self,
        cache_path: str = 'embedding_cache.pkl'
    ):
        logger.info("Initializing OpenAI embedding model")
        
        # Initialize OpenAI embedding function
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        self.cache_path = cache_path
        self.cache = self._load_cache()
        
        # Language detection patterns
        self.vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        self.vietnamese_patterns = [
            r'\b(của|và|hoặc|trong|với|các|những|này|khi|làm|được|không|cho|tại|về|từ|có|đã|sẽ)\b',
            r'\b(thì|là|để|bị|bởi|rằng|nhưng|mà|hay|còn|đang|nên|theo|tới|vào|ra|lên|xuống)\b'
        ]
    
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    @lru_cache(maxsize=1000)
    def _is_vietnamese(self, text: str) -> bool:
        """Simple Vietnamese detection based on common characters."""
        return any(char in self.vietnamese_chars for char in text)

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Advanced language detection with confidence score."""
        try:
            # Clean text
            text = text.strip()
            if not text:
                return "unknown", 0.0
            
            # Check Vietnamese characteristics
            text_chars = set(text.lower())
            vi_char_ratio = len(text_chars.intersection(self.vietnamese_chars)) / len(text)
            
            # Check Vietnamese word patterns
            vi_word_count = sum(len(re.findall(pattern, text)) for pattern in self.vietnamese_patterns)
            text_words = len(text.split())
            vi_word_ratio = vi_word_count / text_words if text_words > 0 else 0
            
            # Use langdetect as additional signal
            try:
                detected_lang = detect(text)
                is_vi_detected = detected_lang == 'vi'
            except LangDetectException:
                is_vi_detected = False
            
            # Combine signals
            vi_confidence = (vi_char_ratio * 0.4 + vi_word_ratio * 0.4 + float(is_vi_detected) * 0.2)
            
            if vi_confidence > 0.5:
                return "vi", vi_confidence
            return "other", 1 - vi_confidence
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "unknown", 0.0
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing."""
        try:
            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Normalize Vietnamese characters
            text = re.sub(r'òóọỏõôồốộổỗơờớợởỡ', 'o', text)
            text = re.sub(r'ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ', 'O', text)
            text = re.sub(r'èéẹẻẽêềếệểễ', 'e', text)
            text = re.sub(r'ÈÉẸẺẼÊỀẾỆỂỄ', 'E', text)
            text = re.sub(r'àáạảãâầấậẩẫăằắặẳẵ', 'a', text)
            text = re.sub(r'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ', 'A', text)
            text = re.sub(r'ìíịỉĩ', 'i', text)
            text = re.sub(r'ÌÍỊỈĨ', 'I', text)
            text = re.sub(r'ùúụủũưừứựửữ', 'u', text)
            text = re.sub(r'ÙÚỤỦŨƯỪỨỰỬỮ', 'U', text)
            text = re.sub(r'ỳýỵỷỹ', 'y', text)
            text = re.sub(r'ỲÝỴỶỸ', 'Y', text)
            text = re.sub(r'đ', 'd', text)
            text = re.sub(r'Đ', 'D', text)
            
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if text in self.cache:
            return self.cache[text]

        try:
            # Get embedding from OpenAI
            embedding = self.ef([text])[0]
            normalized = l2_normalize(np.array(embedding))
            
            # Cache the result
            self.cache[text] = normalized
            self._save_cache()
            
            return normalized
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(1536)  # Return zero vector as fallback (OpenAI dimension)

    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts efficiently."""
        results = []
        to_compute = []
        
        for text in texts:
            if text in self.cache:
                results.append(self.cache[text])
            else:
                to_compute.append(text)
        
        if to_compute:
            try:
                # Process in batches
                batch_size = 32
                for i in range(0, len(to_compute), batch_size):
                    batch = to_compute[i:i + batch_size]
                    batch_embs = self.ef(batch)
                    normalized_embs = [l2_normalize(np.array(emb)) for emb in batch_embs]
                    results.extend(normalized_embs)
                    
                    # Update cache
                    for text, emb in zip(batch, normalized_embs):
                        self.cache[text] = emb
                    self._save_cache()
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Fill with zeros for failed computations
                results.extend([np.zeros(1536)] * len(to_compute))
        
        return np.vstack(results)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            return float(np.dot(emb1, emb2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

# Initialize a singleton instance
embedding_model = EmbeddingModel()
