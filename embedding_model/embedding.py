from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from functools import lru_cache
import torch
from utils.logger import setup_logger
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0
logger = setup_logger()

class EmbeddingModel:
    """Enhanced embedding model with sophisticated multilingual support and caching."""
    
    def __init__(self):
        """Initialize embedding models for better multilingual support."""
        try:
            # Primary model for Vietnamese
            self.vi_model = SentenceTransformer("keepitreal/vietnamese-sbert")
            # Multilingual model for better cross-lingual understanding
            self.multi_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            # Device configuration
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vi_model.to(self.device)
            self.multi_model.to(self.device)
            
            # Language detection patterns
            self.vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
            self.vietnamese_patterns = [
                r'\b(của|và|hoặc|trong|với|các|những|này|khi|làm|được|không|cho|tại|về|từ|có|đã|sẽ)\b',
                r'\b(thì|là|để|bị|bởi|rằng|nhưng|mà|hay|còn|đang|nên|theo|tới|vào|ra|lên|xuống)\b'
            ]
            
            logger.info(f"Embedding models initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            raise
    
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
    
    @lru_cache(maxsize=10000)
    def get_embedding(self, text: str) -> List[List[float]]:
        """Convert text to embedding vector with enhanced processing."""
        try:
            # Input validation and preprocessing
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            text = self.preprocess_text(text)
            if not text:
                return np.zeros((1, 768)).tolist()
            
            # Detect language and get confidence
            lang, confidence = self.detect_language(text)
            
            # Get embeddings from both models
            with torch.no_grad():
                vi_embedding = self.vi_model.encode([text], convert_to_tensor=True)
                multi_embedding = self.multi_model.encode([text], convert_to_tensor=True)
                
                # Dynamic weighting based on language confidence
                if lang == "vi":
                    vi_weight = 0.5 + (confidence * 0.3)  # Max 0.8
                    multi_weight = 1 - vi_weight
                else:
                    multi_weight = 0.5 + (confidence * 0.3)  # Max 0.8
                    vi_weight = 1 - multi_weight
                
                combined_embedding = (
                    vi_weight * vi_embedding + multi_weight * multi_embedding
                ).cpu().numpy()
                
                return combined_embedding.tolist()
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros((1, 768)).tolist()
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts efficiently."""
        try:
            if not texts:
                return []
            
            # Preprocess all texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_embeddings = [self.get_embedding(text)[0] for text in batch]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return [np.zeros(768).tolist()] * len(texts)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            # Preprocess texts
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)
            
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            # Convert to numpy arrays
            emb1_np = np.array(emb1)
            emb2_np = np.array(emb2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_np.flatten(), emb2_np.flatten()) / (
                np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

# Initialize a singleton instance
embedding_model = EmbeddingModel()
