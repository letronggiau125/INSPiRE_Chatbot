import os
import numpy as np
import pickle
import re
from typing import List, Tuple, Dict, Any
from functools import lru_cache
from utils.logger import setup_logger
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from openai import OpenAI

# Set seed for consistent language detection
DetectorFactory.seed = 0
logger = setup_logger()

# Utility: L2 normalization
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

class EmbeddingModel:
    """Enhanced embedding model with multilingual support and caching using OpenAI static API."""
    def __init__(self, cache_path: str = 'embedding_cache.pkl'):
        logger.info("Initializing OpenAI embedding model")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self.model = "text-embedding-ada-002"
        self.cache_path = cache_path
        self.cache = self._load_cache()
        
        # Enhanced Vietnamese detection
        self.vietnamese_chars = set(
            'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        )
        # Expanded Vietnamese patterns
        self.vietnamese_patterns = [
            r"\b(của|và|hoặc|trong|với|các|những|này|khi|làm|được|không|cho|tại|về|từ|có|đã|sẽ)\b",
            r"\b(thì|là|để|bị|bởi|rằng|nhưng|mà|hay|còn|đang|nên|theo|tới|vào|ra|lên|xuống)\b",
            r"\b(cần|phải|muốn|thích|yêu|ghét|thấy|biết|hiểu|nghĩ|nhớ|quên|học|dạy|giúp)\b",
            r"\b(xin|cảm|ơn|chào|tạm|biệt|gặp|lại|khỏe|vui|buồn|giận|sợ|thương|nhớ)\b"
        ]

    def _load_cache(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    @lru_cache(maxsize=1000)
    def _is_vietnamese(self, text: str) -> bool:
        return any(char in self.vietnamese_chars for char in text)

    def detect_language(self, text: str) -> Tuple[str, float]:
        try:
            text = text.strip()
            if not text:
                return "unknown", 0.0
            chars = set(text.lower())
            vi_char_ratio = len(chars & self.vietnamese_chars) / len(text)
            vi_word_count = sum(len(re.findall(p, text)) for p in self.vietnamese_patterns)
            word_count = len(text.split())
            vi_word_ratio = vi_word_count / word_count if word_count else 0.0
            try:
                detected = detect(text)
                is_vi = detected == 'vi'
            except LangDetectException:
                is_vi = False
            vi_conf = vi_char_ratio * 0.4 + vi_word_ratio * 0.4 + float(is_vi) * 0.2
            return ("vi", vi_conf) if vi_conf > 0.5 else ("other", 1 - vi_conf)
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "unknown", 0.0

    def preprocess_text(self, text: str) -> str:
        try:
            # Basic cleaning
            text = text.strip()
            text = re.sub(r"\s+", ' ', text)
            
            # Remove URLs and special characters
            text = re.sub(r"http\S+|www\S+", '', text)
            text = re.sub(r'[^\w\s\d,.!?]', '', text)
            
            # Normalize Vietnamese characters
            text = self._normalize_vietnamese(text)
            
            # Remove redundant punctuation
            text = re.sub(r'([,.!?])\1+', r'\1', text)
            
            # Ensure proper spacing around punctuation
            text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text

    def _normalize_vietnamese(self, text: str) -> str:
        replacements = [
            ('òóọỏõôồốộổỗơờớợởỡ','o'), ('ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ','O'),
            ('èéẹẻẽêềếệểễ','e'), ('ÈÉẸẺẼÊỀẾỆỂỄ','E'),
            ('àáạảãâầấậẩẫăằắặẳẵ','a'), ('ÀÁẠẢÃÂẦẤẬẨỪẰẮẶẲẴ','A'),
            ('ìíịỉĩ','i'), ('ÌÍỊỈĨ','I'),
            ('ùúụủũưừứựửữ','u'), ('ÙÚỤỦŨƯỪỨỰỬỮ','U'),
            ('ỳýỵỷỹ','y'), ('ỲÝỴỶỸ','Y'),
            ('đ','d'), ('Đ','D')
        ]
        for chars, repl in replacements:
            text = re.sub(f'[{chars}]', repl, text)
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        # Preprocess text before checking cache
        processed_text = self.preprocess_text(text)
        
        if processed_text in self.cache:
            return self.cache[processed_text]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=processed_text
            )
            emb = np.array(response.data[0].embedding)
            norm = l2_normalize(emb)
            
            # Cache the result
            self.cache[processed_text] = norm
            self._save_cache()
            
            return norm
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros((1536,), dtype=float)

    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        # Preprocess all texts
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        results, to_compute = [], []
        to_compute_indices = []
        
        # Check cache for each processed text
        for i, pt in enumerate(processed_texts):
            if pt in self.cache:
                results.append(self.cache[pt])
            else:
                to_compute.append(pt)
                to_compute_indices.append(i)
        
        if to_compute:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=to_compute
                )
                
                # Process and cache new embeddings
                computed_embeddings = []
                for item in response.data:
                    emb = np.array(item.embedding, dtype=float)
                    norm = l2_normalize(emb)
                    self.cache[to_compute[item.index]] = norm
                    computed_embeddings.append(norm)
                
                # Insert computed embeddings in correct positions
                for idx, emb in zip(to_compute_indices, computed_embeddings):
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = emb
                
                self._save_cache()
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                for idx in to_compute_indices:
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = np.zeros((1536,), dtype=float)
        
        # Fill any remaining None values with zeros
        results = [r if r is not None else np.zeros((1536,), dtype=float) for r in results]
        return np.vstack(results)

    def compute_similarity(self, text1: str, text2: str) -> float:
        try:
            # Preprocess both texts
            proc_text1 = self.preprocess_text(text1)
            proc_text2 = self.preprocess_text(text2)
            
            emb1 = self.get_embedding(proc_text1)
            emb2 = self.get_embedding(proc_text2)
            
            return float(np.dot(emb1, emb2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

# Export singleton
embedding_model = EmbeddingModel()

