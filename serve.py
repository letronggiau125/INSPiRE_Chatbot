import os
import json
import re
import unicodedata
from typing import Optional, Tuple, List, Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from difflib import SequenceMatcher

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
        # Blacklist phrases that should be filtered out immediately
        self.no_answer_phrases = [
            'thời tiết','chính trị','thể thao','kinh tế vĩ mô','đảng phái','livestream','3D','government policy','chính sách nhà nước','tourism','du lịch','hotel','khách sạn',
            'việt nam có bao nhiêu tỉnh','toán học','vật lý','cờ bạc','đua xe','hàng giả','LGBT','how many provinces does vietnam have','việt nam có bao nhiêu tỉnh','weather','thời tiết','climate change','biến đổi khí hậu',
            'hóa học','sinh học','lịch sử','địa lý','văn học','kim tiêm','học phí','giá cao','population of vietnam','dân số việt nam','flight','chuyến bay','cảng hàng không',
            'tên bạn là gì','bạn bao nhiêu tuổi', 'bạn sống ở đâu','làm tiền','bê đê','quốc phòng','stock market', 'chỉ số chứng khoán','bus schedule', 'lịch xe buýt', 'giao thông',
            'bạn có người yêu chưa','bạn đã kết hôn chưa','cách mạng','covid','quân sự','biển','đảo','inflation rate', 'tỷ lệ lạm phát','music festival', 'lễ hội âm nhạc','sports','thể thao','football', 'bóng đá', 'basketball',
            'bạn hoạt động như thế nào', 'thuật toán của bạn là gì','đảo chính','cầm tù','hoàng xa','trường xa','exchange rate', 'tỷ giá ngoại tệ','công thức nấu ăn'
            'bạn học như thế nào','bạn là model gì', 'dữ liệu huấn luyện của bạn là gì','đánh','khai chiến','concert', 'buổi hòa nhạc', 'movie', 'phim','health', 'sức khỏe', 'y tế',
            'nhà hàng','căng tin','ký túc xá', 'nhà ở','cồn','chất kích thích','mại dâm','xung đột','quốc gia', 'dân tộc','how do you work', 'bot hoạt động như thế nào',
            'giao thông','lịch xe buýt','bãi đậu xe','lịch thi','bia','politics', 'chính trị', 'đảng phái', 'kinh tế vĩ mô','exercise', 'thể dục', 'gym',
       
        ]

        # Library domain keywords
        self.LIBRARY_KEYWORDS = {
            'library', 'book', 'borrow', 'return', 'renew',
            'resources', 'wifi', 'computer', 'printer',
            'find', 'lost item', 'library card',
            'contact', 'rules', 'services', 'guidelines',
            'thư viện', 'sách', 'mượn', 'trả', 'đọc', 'tài liệu',
            'sinh viên', 'giảng viên', 'trường', 'đại học',
            'tdtu', 'thẻ', 'phí', 'phạt', 'gia hạn', 'đăng ký',
            'tài khoản', 'cơ sở', 'phòng', 'giờ', 'mở cửa',
            'đóng cửa', 'dịch vụ', 'nghiên cứu', 'học tập',
            'thi', 'kỳ thi', 'khoa', 'ngành'
        }

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
        }
        self.keyword_map = {
            'printer': 'dichvu',
            'scanner': 'huongdan',
            'wifi': 'dichvu',
            'download': 'huongdan',
            'upload': 'huongdan',
            'login': 'huongdan',
            'lost': 'lienhe',
            'mất': 'lienhe',
            'thất lạc': 'lienhe',
            'quên': 'lienhe',
            'forgot': 'lienhe',
            'missing': 'lienhe'
        }

        # Regex patterns to "predict" categories from user queries
        self.question_patterns = {
            'instructions': [r'(how|how do i|how can i)', r'cach .*', r'where .*', r'where .*'],
            'quydinh': [r'quy định .*', r'(co có|dức được|dức được|rules).*'],
            'dichvu': [r'điều sĩ .*', r'(phì|giá|services).*'],
            'muon_tra': [r'(mượn|tuần| xuân|borrow|return|renew).*'],
            'lianhe': [r'(liễn liên|contact|gải ai).*'],
            'chitchat': [r'(hello|hi|hello|goodbye|bye).*']
        }
        
        # Strict chitchat patterns - only basic greetings and thanks
        self.chitchat_patterns = {
            r'\b(xin chào|hello|hi|hey|chào)\b': 'greeting',
            r'\b(cảm ơn|thank|thanks|cám ơn)\b': 'thanks',
            r'\b(tạm biệt|bye|goodbye)\b': 'goodbye'
        }

        # Load FAQ data and build hash map for direct lookup
        self.faq_data = self.load_faq_data()
        self.faq_map = self.build_faq_map()

        # Build semantic index on FAQ questions
        questions = [entry['question'] for entry in self.faq_data]
        semantic_search.build_index(questions)

        # Define allowed specialized categories for semantic search
        self.allowed_specialized = {'huongdan', 'quydinh', 'muon_tra', 'dichvu', 'lienhe'}

    def in_library_domain(self, text: str) -> bool:
        """Check if the text is related to library domain."""
        text = text.lower()
        return any(kw in text for kw in self.LIBRARY_KEYWORDS)

    def load_faq_data(self) -> List[Dict[str, Any]]:
        try:
            with open("faq_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            return []

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing accents, punctuation, and extra whitespace."""
        # 1) NFKD for accent separation
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if not unicodedata.combining(ch))
        # 2) Lower, strip punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # remove .,;?…
        text = re.sub(r'\s+', ' ', text).strip()  # collect whitespace
        return text

    def build_faq_map(self) -> Dict[str, Dict[str, Any]]:
        faq_map = {}
        for entry in self.faq_data:
            # Normalize main question
            raw = entry['question']
            norm = self.normalize_text(raw)
            faq_map[norm] = entry
            
            # Normalize aliases too
            for alias in entry.get('aliases', []):
                norm_alias = self.normalize_text(alias)
                faq_map[norm_alias] = entry
        return faq_map

    def normalize_abbreviations(self, text: str) -> str:
        q = text.lower().strip()
        # Remove trailing punctuation
        q = re.sub(r'[?!.]+$', '', q)
        # Expand abbreviations
        for abbr, full in self.abbrev_map.items():
            q = re.sub(rf"\b{re.escape(abbr)}\b", full, q)
        return q

    def find_best_match(self, user_question: str, threshold: float = 0.9) -> Dict[str, Any]:
        norm = self.normalize_text(user_question)

        # 1) Direct lookup / alias
        if norm in self.faq_map:
            return {**self.faq_map[norm], 'confidence': 1.0}

        # 2) Fuzzy-match alias
        best_ratio, best_entry = 0.0, None
        for key, ent in self.faq_map.items():
            r = SequenceMatcher(None, norm, key).ratio()
            if r > best_ratio:
                best_ratio, best_entry = r, ent
        if best_ratio >= 0.85:
            return {**best_entry, 'confidence': best_ratio}

        # 3) Keyword map
        for kw, cat in self.keyword_map.items():
            if kw in norm:
                return {'question': user_question, 'category': cat, 'confidence': 1.0}

        # 4) Domain filter: if not in library domain, ignore semantics
        if not self.in_library_domain(norm):
            return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

        # 5) Semantic fallback only for allowed categories
        try:
            results = semantic_search.query(user_question, top_k=1, threshold=threshold)
            if results:
                idx, score = results[0]['index'], float(results[0]['score'])
                matched = self.faq_data[idx].copy()
                if matched.get('category') in self.allowed_specialized and score >= threshold:
                    matched['confidence'] = score
                    return matched
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")

        # 6) No match
        return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

    def get_best_response(self, user_message: str) -> Tuple[str, str, float]:
        match = self.find_best_match(user_message)
        answer = match.get('answer')
        category = match.get('category')
        confidence = match.get('confidence', 0.0)

        # 1) Direct/alias or keyword match with given answer
        if confidence == 1.0 and answer:
            return answer, category, confidence

        # 2) Only chatbot dialogue when truly chit-chat
        if category == 'chitchat':
            conv = Reflection(db_path=db_path).chat(None, user_message)
            return conv, 'chatbot', 0.0

        # 3) Fuzzy or semantic success
        if confidence > 0.0:
            return answer or "", category, confidence

        # 4) Default: not understood
        return (
            "Xin lỗi, tôi chưa hiểu câu hỏi. Vui lòng thử lại hoặc đặt câu hỏi khác nhé.",
            'unknown',
            0.0
        )

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
