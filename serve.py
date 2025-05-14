import os
import json
import re
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from reflection import Reflection
from semantic_router.semantic import SemanticSearch
from embedding_model.embedding import EmbeddingModel
from config import Config
from utils.logger import setup_logger
from utils.rate_limiter import rate_limit
from utils.validators import MessageValidator
from utils.response import ChatResponse
import numpy as np
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_distances
from rag.rag_integration import RAG
from rag.ingest_faq import ingest_all_faqs

# Initialize embedding model
embedding_model = EmbeddingModel()

# Lazy loading model to reduce memory footprint
@lru_cache(maxsize=1)
def get_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )

# Load environment variables
load_dotenv()
db_path = os.getenv('DB_PATH', './chroma_db')

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static',
    template_folder='templates'
)
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

logger = setup_logger()

class FAQMatcher:
    def __init__(self):
        self.ef = get_embedding_function()
        self.tech_terms = {
            'máy tính': ['computer', 'laptop', 'pc'],
            'máy in': ['printer'],
            'máy quét': ['scanner'],
            'photo': ['copy'],
            'wifi': ['internet'],
            'tải': ['download', 'upload'],
            'đăng nhập': ['login'],
            'tài khoản': ['account'],
            'mật khẩu': ['password']
        }
        self.abbrev_map = {
            'tdt': 'trường đại học tôn đức thắng',
            'tv': 'thư viện',
            'sv': 'sinh viên',
            'gv': 'giảng viên',
            'cn': 'chủ nhật',
            't2': 'thứ hai',
            't3': 'thứ ba',
            't4': 'thứ tư',
            't5': 'thứ năm',
            't6': 'thứ sáu',
            't7': 'thứ bảy',
            'p/s': 'phòng sinh viên',
            'p/gv': 'phòng giảng viên',
            'p/tv': 'phòng thư viện',
            'p/hc': 'phòng hành chính',
            'p/kt': 'phòng kỹ thuật',
            'p/tt': 'phòng thông tin',
            'p/ql': 'phòng quản lý',
            'p/nv': 'phòng nhân viên',
            'p/bs': 'phòng bảo vệ',
            'p/vs': 'phòng vệ sinh',
            'p/đx': 'phòng đọc xuất',
            'p/mt': 'phòng mượn trả',
            'p/tk': 'phòng tra cứu'
        }
        self.keyword_map = {
            'printer': 'dichvu', 'scanner': 'huongdan', 'wifi': 'dichvu',
            'download': 'huongdan', 'upload': 'huongdan', 'login': 'huongdan'
        }
        self.question_patterns = self.generate_question_patterns()
        self.faq_data = self.load_faq_data()
        self.faq_map = self.build_faq_map()
        self.corpus_embeddings = self.encode_corpus()

    def generate_question_patterns(self) -> Dict[str, List[str]]:
        """Generate regex patterns for different question categories."""
        return {
            'huongdan': [
                r'(làm thế nào|làm sao|how do i|how can i).*',
                r'cách .*',
                r'where .*',
                r'ở đâu .*'
            ],
            'quydinh': [
                r'quy định .*',
                r'(có được|được phép|rules).*'
            ],
            'dichvu': [
                r'dịch vụ .*',
                r'(phí|giá|services).*'
            ],
            'muon_tra': [
                r'(mượn|trả|gia hạn|borrow|return|renew).*'
            ],
            'lienhe': [
                r'(liên hệ|contact|gặp ai).*'
            ],
            'chitchat': [
                r'(xin chào|hi|hello|tạm biệt|bye).*'
            ]
        }

    def load_faq_data(self) -> List[Dict[str, Any]]:
        """Load FAQ data from JSON file."""
        try:
            with open("faq_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    if 'embedding' not in entry:
                        entry['embedding'] = self.ef([entry['question']])[0]
                return data
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            return []

    def build_faq_map(self) -> Dict[str, Dict[str, Any]]:
        """Build hash map for direct question/alias lookups."""
        faq_map = {}
        for entry in self.faq_data:
            question = entry['question'].strip().lower()
            faq_map[question] = entry
            for alias in entry.get('aliases', []):
                alias_norm = alias.strip().lower()
                faq_map[alias_norm] = entry
        return faq_map

    def encode_corpus(self) -> np.ndarray:
        """Encode all FAQ questions for semantic search."""
        return np.array(self.ef([e['question'] for e in self.faq_data]))

    def normalize_abbreviations(self, text: str) -> str:
        """Normalize text by expanding abbreviations."""
        q = text.lower().strip()
        for abbr, full in self.abbrev_map.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            q = re.sub(pattern, full, q)
        return q

    def find_best_match(self, user_question: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Find best matching FAQ entry using direct lookup and semantic search."""
        # Normalize the query
        norm_query = self.normalize_abbreviations(user_question).strip().lower()
        
        # 1. Direct lookup in FAQ map
        if norm_query in self.faq_map:
            return {**self.faq_map[norm_query], 'confidence': 1.0}
        
        # 2. Check keyword matches
        for keyword, cat in self.keyword_map.items():
            if keyword in norm_query:
                return {'question': user_question, 'category': cat, 'confidence': 1.0}
        
        # 3. Semantic search fallback
        try:
            query_embedding = np.array(self.ef([user_question]))
            similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= threshold:
                return {**self.faq_data[best_idx], 'confidence': float(best_score)}
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        
        # 4. No match found - return fallback response
        return {
            'question': user_question,
            'category': 'unknown',
            'confidence': 0.0
        }

    def get_best_response(self, user_message: str) -> Tuple[Optional[str], Optional[str], float]:
        match = self.find_best_match(user_message)
        return (match.get('answer'), match.get('category'), match.get('confidence')) if match else (None, None, 0.0)

faq_matcher = FAQMatcher()

chatbot = Reflection(db_path=Config.DB_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@rate_limit

def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default_session")
        if not user_message:
            return jsonify({"error": Config.MESSAGES['empty_message']}), 400
        response_text, category, confidence = faq_matcher.get_best_response(user_message)
        if not response_text or confidence < Config.CHAT_SETTINGS['min_confidence_score']:
            user_embedding = embedding_model.get_embedding(user_message)
            if not user_embedding:
                return jsonify(ChatResponse(
                    message=Config.MESSAGES['unknown_question'],
                    confidence=0.0
                ).to_dict())
            response_text = chatbot.chat(session_id, user_message)
        return jsonify(ChatResponse(message=response_text, category=category, confidence=confidence).to_dict())
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": Config.MESSAGES['server_error']}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    return jsonify({"error": Config.MESSAGES['server_error']}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=Config.DEBUG_MODE)
