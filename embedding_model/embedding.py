import os
import numpy as np
import re
import pickle
from typing import List, Tuple, Dict, Any
from functools import lru_cache
from utils.logger import setup_logger
from openai import OpenAI, OpenAIError
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0
logger = setup_logger()

# Utility: L2 normalization

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

class EmbeddingModel:
    """Enhanced embedding model with multilingual support and caching using OpenAI v1.x API."""
    def __init__(self, cache_path: str = 'embedding_cache.pkl'):
        logger.info("Initializing OpenAI embedding model (v1.x)")
        self.client = OpenAI()
        self.model = "text-embedding-ada-002"
        self.cache_path = cache_path
        self.cache = self._load_cache()

        # Vietnamese detection
        self.vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        self.vietnamese_patterns = [
            r'\b(của|và|hoặc|trong|với|các|những|này|khi|làm|được|không|cho|tại|về|từ|có|đã|sẽ)\b',
            r'\b(thì|là|để|bị|bởi|rằng|nhưng|mà|hay|còn|đang|nên|theo|tới|vào|ra|lên|xuống)\b'
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
            char_set = set(text.lower())
            vi_char_ratio = len(char_set.intersection(self.vietnamese_chars)) / len(text)

            vi_word_count = sum(len(re.findall(p, text)) for p in self.vietnamese_patterns)
            word_count = len(text.split())
            vi_word_ratio = vi_word_count / word_count if word_count > 0 else 0.0

            try:
                detected = detect(text)
                is_vi = detected == 'vi'
            except LangDetectException:
                is_vi = False

            confidence = vi_char_ratio * 0.4 + vi_word_ratio * 0.4 + float(is_vi) * 0.2
            if confidence > 0.5:
                return "vi", confidence
            return "other", 1 - confidence
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "unknown", 0.0

    def preprocess_text(self, text: str) -> str:
        try:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'http\S+|www\S+', '', text)
            replacements = [
                ('òóọỏõôồốộổỗơờớợởỡ','o'),('ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ','O'),
                ('èéẹẻẽêềếệểễ','e'),('ÈÉẸẺẼÊỀẾỆỂỄ','E'),
                ('àáạảãâầấậẩẫăằắặẳẵ','a'),('ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ','A'),
                ('ìíịỉĩ','i'),('ÌÍỊỈĨ','I'),
                ('ùúụủũưừứựửữ','u'),('ÙÚỤỦŨƯỪỨỰỬỮ','U'),
                ('ỳýỵỷỹ','y'),('ỲÝỴỶỸ','Y'),('đ','d'),('Đ','D')
            ]
            for chars, repl in replacements:
                text = re.sub(f'[{chars}]', repl, text)
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text

    def get_embedding(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            emb = np.array(resp.data[0].embedding, dtype=float)
            norm = l2_normalize(emb)
            self.cache[text] = norm
            self._save_cache()
            return norm
        except OpenAIError as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(1536, dtype=float)

    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        results: List[np.ndarray] = []
        to_compute: List[str] = []
        for t in texts:
            if t in self.cache:
                results.append(self.cache[t])
            else:
                to_compute.append(t)
        if to_compute:
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=to_compute
                )
                for i, datum in enumerate(resp.data):
                    emb = np.array(datum.embedding, dtype=float)
                    norm = l2_normalize(emb)
                    self.cache[to_compute[i]] = norm
                    results.append(norm)
                self._save_cache()
            except OpenAIError as e:
                logger.error(f"Error in batch processing: {e}")
                results.extend([np.zeros(1536, dtype=float)] * len(to_compute))
        return np.vstack(results)

    def compute_similarity(self, text1: str, text2: str) -> float:
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            return float(np.dot(emb1, emb2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

# Export singleton instance
embedding_model = EmbeddingModel()
