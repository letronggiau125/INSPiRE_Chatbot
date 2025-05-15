import os
import json
import re
from typing import Optional, Tuple, List, Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from rag.ingest_faq import ingest_all_faqs
from reflection import Reflection
from semantic_router.semantic import SemanticSearch
from embedding_model.embedding import EmbeddingModel
from config import Config
from utils.logger import setup_logger
from utils.rate_limiter import rate_limit
from utils.validators import MessageValidator
from utils.response import ChatResponse

# Load environment variables
load_dotenv()
default_db = './chroma_db'

# Ingest all FAQ data into ChromaDB before starting the service
print("🔄 Ingesting all FAQ data into ChromaDB...")
ingest_all_faqs()
print("✅ FAQ ingest completed.")

# Initialize paths and services
db_path = os.getenv('DB_PATH', default_db)

# Initialize Flask app
app = Flask(
    __name__,
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

# Setup logging
logger = setup_logger()

# Initialize embedding and semantic search models
embedding_model = EmbeddingModel()
semantic_search = SemanticSearch()

class FAQMatcher:
    def __init__(self):
        # Abbreviation and keyword maps
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
            # ... thêm các viết tắt khác
        }
        self.keyword_map = {
            'printer': 'dichvu',
            'scanner': 'huongdan',
            'wifi': 'dichvu',
            'download': 'huongdan',
            'upload': 'huongdan',
            'login': 'huongdan'
        }

        # Load FAQ data and build hash map for direct lookup
        self.faq_data = self.load_faq_data()
        self.faq_map = self.build_faq_map()

        # Build semantic index on FAQ questions
        questions = [entry['question'] for entry in self.faq_data]
        semantic_search.build_index(questions)

    def load_faq_data(self) -> List[Dict[str, Any]]:
        try:
            with open("faq_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            return []

    def build_faq_map(self) -> Dict[str, Dict[str, Any]]:
        faq_map: Dict[str, Dict[str, Any]] = {}
        for entry in self.faq_data:
            key = entry['question'].strip().lower()
            faq_map[key] = entry
            for alias in entry.get('aliases', []):
                faq_map[alias.strip().lower()] = entry
        return faq_map

    def normalize_abbreviations(self, text: str) -> str:
        q = text.lower().strip()
        for abbr, full in self.abbrev_map.items():
            q = re.sub(rf"\b{re.escape(abbr)}\b", full, q)
        return q

    def find_best_match(self, user_question: str, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        norm = self.normalize_abbreviations(user_question)
        # 1. Direct lookup
        entry = self.faq_map.get(norm.lower())
        if entry:
            return {**entry, 'confidence': 1.0}
        # 2. Keyword map
        for kw, cat in self.keyword_map.items():
            if kw in norm:
                return {'question': user_question, 'category': cat, 'confidence': 1.0}
        # 3. Semantic fallback via SemanticSearch
        try:
            results = semantic_search.query(user_question, top_k=1, threshold=threshold)
            if results:
                idx = results[0]['index']
                score = float(results[0]['score'])
                matched = self.faq_data[idx].copy()
                matched['confidence'] = score
                return matched
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        # 4. No match
        return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

    def get_best_response(self, user_message: str) -> Tuple[Optional[str], Optional[str], float]:
        match = self.find_best_match(user_message)
        if match and match['confidence'] >= Config.CHAT_SETTINGS['min_confidence_score']:
            return match.get('answer'), match.get('category'), match.get('confidence')
        # Fallback to chatbot
        emb = embedding_model.get_embedding(user_message)
        if emb:
            answer = Reflection(db_path=db_path).chat(None, user_message)
            return answer, 'chatbot', 0.0
        return Config.MESSAGES['unknown_question'], 'unknown', 0.0

# Instantiate matcher and chatbot
faq_matcher = FAQMatcher()
chatbot = Reflection(db_path=Config.DB_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    try:
        data = request.get_json() or {}
        msg = data.get("message", "").strip()
        session_id = data.get("session_id", "default_session")
        if not msg:
            return jsonify({"error": Config.MESSAGES['empty_message']}), 400

        answer, category, confidence = faq_matcher.get_best_response(msg)
        return jsonify(ChatResponse(
            message=answer,
            category=category,
            confidence=confidence
        ).to_dict())
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
