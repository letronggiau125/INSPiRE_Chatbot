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
import numpy as np
import torch

# Lazy loading model to reduce memory footprint
@lru_cache(maxsize=1)
def get_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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
        self.sentence_model = get_sentence_model()
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
                    entry['embedding'] = torch.tensor(entry['embedding']) if 'embedding' in entry else self.sentence_model.encode(entry['question'], convert_to_tensor=True)
                    entry['variations'] = self.generate_question_variations(entry['question'])
                return data
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            return []

    def build_variation_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        for entry in self.faq_data:
            for variation in entry['variations']:
                lookup[variation.lower()] = entry
        return lookup

    def generate_question_patterns(self) -> Dict[str, List[str]]:
        return {
            'huongdan': [r'(làm thế nào|làm sao|how do i|how can i).*', r'cách .*', r'where .*', r'ở đâu .*'],
            'quydinh': [r'quy định .*', r'(có được|được phép|rules).*'],
            'dichvu': [r'dịch vụ .*', r'(phí|giá|services).*'],
            'muon_tra': [r'(mượn|trả|gia hạn|borrow|return|renew).*'],
            'lienhe': [r'(liên hệ|contact|gặp ai).*'],
            'chitchat': [r'(xin chào|hi|hello|tạm biệt|bye).*']
        }

    def generate_question_variations(self, question: str) -> List[str]:
        base = question.lower().rstrip('?')
        variations = [base]
        for vn_term, eng_terms in self.tech_terms.items():
            if vn_term in base:
                variations.extend(base.replace(vn_term, eng) for eng in eng_terms)
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
            return {**matched_entry, 'confidence': 1.0}
        try:
            question_embedding = self.sentence_model.encode(user_question, convert_to_tensor=True)
            similar = [(float(torch.nn.functional.cosine_similarity(entry['embedding'], question_embedding, dim=0)), entry) for entry in self.faq_data]
            similar = [pair for pair in similar if pair[0] > threshold]
            if similar:
                similar.sort(key=lambda x: x[0], reverse=True)
                best_score, best_entry = similar[0]
                return {**best_entry, 'confidence': best_score}
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        return None

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
