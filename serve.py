import os
import json
import re
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from reflection import Reflection
from semantic_router.routers.semantic import SemanticRouter
from semantic_router.routers.route import Route
from embedding_model.embedding import embedding_model
from config import Config
from utils.logger import setup_logger
from utils.rate_limiter import rate_limit
from utils.validators import MessageValidator
from utils.response import ChatResponse
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# 🌟 Load environment variables
load_dotenv()
db_path = os.getenv('DB_PATH', './chroma_db')

# 🏗️ Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = setup_logger()

# Initialize sentence transformer model for semantic similarity
sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

class FAQMatcher:
    """Optimized FAQ Matcher."""

    def __init__(self):
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
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
        self.keyword_map = {
            'printer': 'dichvu', 'scanner': 'huongdan', 'wifi': 'dichvu',
            'download': 'huongdan', 'upload': 'huongdan', 'login': 'huongdan'
        }

        self.question_patterns = self.generate_question_patterns()
        self.faq_data = self.load_faq_data()
        self.variation_lookup = self.build_variation_lookup()


    def load_faq_data(self) -> List[Dict[str, Any]]:
        try:
            with open("faq_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    # Use precomputed embedding if available, else compute
                    if 'embedding' in entry:
                        entry['embedding'] = torch.tensor(entry['embedding'])
                    else:
                        entry['embedding'] = self.sentence_model.encode(entry['question'], convert_to_tensor=True)
                    entry['variations'] = self.generate_question_variations(entry['question'])
                return data
        except Exception as e:
            print(f"Error loading FAQ data: {e}")
            return []

    def build_variation_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        for entry in self.faq_data:
            for variation in entry['variations']:
                lookup[variation.lower()] = entry
        return lookup

    def generate_question_patterns(self) -> Dict[str, List[str]]:
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

    def generate_question_variations(self, question: str) -> List[str]:
        variations = [question.lower().rstrip('?')]
        base = variations[0]
        for vn_term, eng_terms in self.tech_terms.items():
            if vn_term in base:
                for eng in eng_terms:
                    variations.append(base.replace(vn_term, eng))
            for eng in eng_terms:
                if eng in base:
                    variations.append(base.replace(eng, vn_term))
        return list(set(variations))

    @lru_cache(maxsize=1000)
    def match_question_pattern(self, question: str) -> Optional[str]:
        question = question.lower()
        for category, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return category
        return None

    def find_best_match(self, user_question: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        user_question_clean = user_question.lower().rstrip('?')

        for keyword, cat in self.keyword_map.items():
            if keyword in user_question_clean:
                return {'question': user_question, 'category': cat, 'confidence': 1.0}

        if user_question_clean in self.variation_lookup:
            matched_entry = self.variation_lookup[user_question_clean]
            return {
                'question': matched_entry['question'],
                'answer': matched_entry['answer'],
                'category': matched_entry['category'],
                'confidence': 1.0
            }

        try:
            question_embedding = self.sentence_model.encode(user_question, convert_to_tensor=True)
            similar = []
            for entry in self.faq_data:
                # Use cosine similarity for 1D tensors
                score = float(util.pytorch_cos_sim(entry['embedding'], question_embedding)[0][0])
                if score > threshold:
                    similar.append((score, entry))

            if similar:
                similar.sort(key=lambda x: x[0], reverse=True)
                best_score, best_entry = similar[0]
                return {
                    'question': best_entry['question'],
                    'answer': best_entry['answer'],
                    'category': best_entry['category'],
                    'confidence': best_score
                }
        except Exception as e:
            print(f"Error in semantic matching: {e}")

        return None

    def get_best_response(self, user_message: str) -> Tuple[Optional[str], Optional[str], float]:
        """Compatible method for legacy calls."""
        match = self.find_best_match(user_message)
        if match:
            return match.get('answer'), match.get('category'), match.get('confidence')
        else:
            return None, None, 0.0

# Initialize FAQ matcher
faq_matcher = FAQMatcher()

def get_samples_by_category(category):
    """Get sample questions for each category."""
    samples = [entry["question"] for entry in faq_matcher.faq_data if entry.get("category") == category]
    if not samples:
        logger.warning(f"No data found for category: {category}")
    return samples

# Initialize Routes for FAQ
routes = [
    Route(name=f"faq_{category}", samples=get_samples_by_category(category))
    for category in Config.CATEGORIES.keys()
]
semantic_router = SemanticRouter(routes=routes)

# Initialize chatbot with ChromaDB
chatbot = Reflection(db_path=Config.DB_PATH)

def get_response(session_id: str, user_message: str) -> ChatResponse:
    """Process user message and return appropriate response."""
    logger.info(f"Processing message: {user_message}", session_id=session_id)
    
    # Validate and sanitize input
    user_message = MessageValidator.sanitize_message(user_message)
    is_valid, error = MessageValidator.validate_message(user_message)
    if not is_valid:
        return ChatResponse(message=error)
    
    try:
        # Get best response using enhanced matching
        response, category, confidence = faq_matcher.get_best_response(user_message)
        
        if response and confidence >= Config.CHAT_SETTINGS['min_confidence_score']:
            return ChatResponse(
                message=response,
                category=category,
                confidence=confidence
            )
        
        # If no good match found, try the chatbot
        user_embedding = embedding_model.get_embedding(user_message)
        if not user_embedding or not isinstance(user_embedding, list) or len(user_embedding) == 0:
            logger.warning(f"Empty embedding for message: {user_message}", session_id=session_id)
            return ChatResponse(
                message="Xin lỗi, hiện tại tôi chưa thể trả lời câu hỏi của bạn, vui lòng đặt câu hỏi trên fanpage thư viện để có thể giải đáp thắc mắc. Đây là địa chỉ ạ (https://www.facebook.com/tvdhtdt)",
                category=None,
                confidence=0.0
            )
        
        # The following semantic_router.route logic is removed because the method does not exist
        # Fallback to chatbot
        response = chatbot.chat(session_id, user_message)
        return ChatResponse(
            message="Xin lỗi, hiện tại tôi chưa thể trả lời câu hỏi của bạn, vui lòng đặt câu hỏi trên fanpage thư viện để có thể giải đáp thắc mắc. Đây là địa chỉ ạ (https://www.facebook.com/tvdhtdt)",
            category=None,
            confidence=0.0
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True, session_id=session_id)
        return ChatResponse(message=Config.MESSAGES['server_error'])

@app.route("/")
def home():
    """Serve the main chat interface."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default_session")
        
        if not user_message:
            return jsonify({"error": Config.MESSAGES['empty_message']}), 400
        
        response = get_response(session_id, user_message)
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": Config.MESSAGES['server_error']}), 500

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
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
    logger.info(f"Starting server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG_MODE
    )
