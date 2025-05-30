import os
import json
import re
import unicodedata
from typing import Optional, Tuple, List, Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from difflib import SequenceMatcher
from rapidfuzz import fuzz, process

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
            'thời tiết','chính trị','thể thao','kinh tế vĩ mô','đảng phái','livestream','3D','government policy','chính sách nhà nước','tourism','du lịch','hotel','khách sạn','cá độ','đặt cược'
            'việt nam có bao nhiêu tỉnh','toán học','vật lý','cờ bạc','đua xe','hàng giả','LGBT','how many provinces does vietnam have','việt nam có bao nhiêu tỉnh','weather','thời tiết','climate change','biến đổi khí hậu',
            'hóa học','sinh học','lịch sử','địa lý','văn học','kim tiêm','học phí','giá cao','population of vietnam','dân số việt nam','flight','chuyến bay','cảng hàng không','cá cược','tử sát'
            'tên bạn là gì','bạn bao nhiêu tuổi', 'bạn sống ở đâu','làm tiền','bê đê','quốc phòng','stock market', 'chỉ số chứng khoán','bus schedule', 'lịch xe buýt', 'giao thông','tự tử'
            'bạn có người yêu chưa','bạn đã kết hôn chưa','cách mạng','covid','quân sự','biển','đảo','inflation rate', 'tỷ lệ lạm phát','music festival', 'lễ hội âm nhạc','sports','thể thao','football', 'bóng đá', 'basketball',
            'bạn hoạt động như thế nào', 'thuật toán của bạn là gì','đảo chính','cầm tù','hoàng xa','trường xa','exchange rate', 'tỷ giá ngoại tệ','công thức nấu ăn','bê đê','thiêu sống'
            'bạn học như thế nào','bạn là model gì', 'dữ liệu huấn luyện của bạn là gì','đánh','khai chiến','concert', 'buổi hòa nhạc', 'movie', 'phim','health', 'sức khỏe', 'y tế',
            'nhà hàng','căng tin','ký túc xá', 'nhà ở','cồn','chất kích thích','mại dâm','xung đột','quốc gia', 'dân tộc','how do you work', 'bot hoạt động như thế nào','giết'
            'giao thông','lịch xe buýt','bãi đậu xe','lịch thi','bia','politics', 'chính trị', 'đảng phái', 'kinh tế vĩ mô','exercise', 'thể dục', 'gym','nhạy cảm','chém','đâm'
        ]

        # Common typos mapping
        self.common_typos = {
            'thời tiết': ['thoi tiet', 'thời tiết', 'thời tiêt'],
            'chính trị': ['chinh tri', 'chính trị', 'chính trị'],
            'thể thao': ['the thao', 'thể thao', 'thể thao'],
            'du lịch': ['du lich', 'du lịch', 'du lịch'],
            'khách sạn': ['khach san', 'khách sạn', 'khách sạn'],
            'thể dục': ['the duc', 'thể dục', 'thể dục'],
            'giao thông': ['giao thong', 'giao thông', 'giao thông'],
            'chính sách': ['chinh sach', 'chính sách', 'chính sách'],
            'quốc gia': ['quoc gia', 'quốc gia', 'quốc gia'],
            'dân tộc': ['dan toc', 'dân tộc', 'dân tộc'],
            'Thư viện': ['thu vien', 'thư viện','thư viện inspire',"INSPiRE",'thư viện INSPiRE','Thư viện Truyền cảm hứng'],
            'Trường Đại học Tôn Đức Thắng': ['tdtu', 'trường đại học tôn đức thắng', 'TDTU','trường tdt','Trường Ton Duc Thang','đại học Ton Duc Thang'],
            'Đại học Tôn Đức Thắng': ['tdtu', 'trường đại học tôn đức thắng', 'TDTU'],
        }   

        # Ignored patterns for super normalization
        self.IGNORED_PATTERNS = [
            r'\bở thư lý\b', r'\bài thư lý\b', r'\btrong thư lý\b',
            r'\btoi muon\b', r'\btoi can\b', r'\bgiu\b', r'\bcho em hỏi\b', r'\bxin hỏi\b', r'\blàm xin\b', r'\bxin\b',
            r'\bbai là\b', r'\bai đang\b', r'\bai rồi\b', r'\bai đi\b', r'\bai đi chủ\b',
            r'\bnay về\b', r'\bchi ​​​​tiết\b', r'\b thông tin về\b',
            r'\bvui lòng\b', r'\blàm ơn\b', r'\bgiúp\b', r'\bgiúp tôi\b',
            r'\bcho tôi biết\b', r'\bcho tôi hỏi\b', r'\bcho hỏi\b',
            r'\bcó thể\b', r'\bcó được không\b', r'\bcó phải\b',
            r'\bđược không\b', r'\bphải không\b', r'\bđúng không\b'
        ]

        # Synonym mapping for better matching
        self.SYNONYM_MAP = {
            'mượn': ['vay', 'lấy', 'nhận', 'đọc'],
            'trả': ['hoàn trả', 'nộp lại', 'gửi lại'],
            'tài liệu': ['sách', 'sách báo', 'giáo trình', 'tài liệu học tập'],
            'giảng viên': ['thầy cô', 'giáo viên', 'giảng viên'],
            'sinh viên': ['học viên', 'sinh viên'],
            'thư viện': ['thư viện trường', 'thư viện tdtu'],
            'máy tính': ['máy vi tính', 'computer', 'pc'],
            'máy in': ['printer', 'máy photocopy'],
            'wifi': ['internet', 'mạng', 'kết nối mạng'],
            'đăng nhập': ['login', 'đăng ký', 'truy cập'],
            'tài khoản': ['account', 'user'],
            'mật khẩu': ['password', 'pass'],
            'phí': ['giá', 'chi phí', 'tiền'],
            'gia hạn': ['renew', 'kéo dài', 'thêm thời gian'],
            'quy định': ['nội quy', 'luật', 'quy tắc'],
            'dịch vụ': ['service', 'tiện ích'],
            'hướng dẫn': ['guide', 'manual', 'tutorial'],
            'liên hệ': ['contact', 'gặp', 'gặp mặt'],
            'mất': ['thất lạc', 'quên', 'không tìm thấy'],
            'tìm': ['search', 'tìm kiếm', 'locate']
        }

        # Library domain keywords
        self.LIBRARY_KEYWORDS = {
            'library', 'book', 'borrow', 'return', 'renew','phòng chức năng',
            'resources', 'wifi', 'computer', 'printer','inspire','INSPiRE',
            'find', 'lost item', 'library card','Thư viện Truyền cảm hứng',
            'contact', 'rules', 'services', 'guidelines',
            'thư viện', 'sách', 'mượn', 'trả', 'đọc', 'tài liệu',
            'sinh viên', 'giảng viên', 'trường', 'đại học',
            'tdtu', 'thẻ', 'phí', 'phạt', 'gia hạn', 'đăng ký',
            'tài khoản', 'cơ sở', 'phòng', 'giờ', 'mở cửa',
            'đóng cửa', 'dịch vụ', 'nghiên cứu', 'học tập',
            'thi', 'kỳ thi', 'khoa', 'ngành','TVĐHTĐT'
        }

        # Expanded tech terms mapping
        self.tech_terms = {
            'máy tính': ['computer', 'laptop', 'pc'],
            'máy in': ['printer'],
            'máy quét': ['scanner'],
            'photo': ['copy'],
            'wifi': ['internet'],
            'tải': ['download', 'upload'],
            'đăng nhập': ['login'],
            'tài khoản': ['account'],
            'mật khẩu': ['password'],
            # Reference management software
            'endnote': 'huongdan',
            'zotero': 'huongdan',
            'mendeley': 'huongdan',
            'reference manager': 'huongdan',
            'citation': 'huongdan',
            'trích dẫn': 'huongdan',
            'tài liệu tham khảo': 'huongdan',
            # Database access
            'database': 'huongdan',
            'cơ sở dữ liệu': 'huongdan',
            'proquest': 'huongdan',
            'ebsco': 'huongdan',
            'ieee': 'huongdan',
            'science direct': 'huongdan',
            'springer': 'huongdan',
            # Research tools
            'turnitin': 'huongdan',
            'plagiarism': 'huongdan',
            'đạo văn': 'huongdan',
            'research': 'huongdan',
            'nghiên cứu': 'huongdan'
        }
        self.abbrev_map = {
            'tdt': 'trường đại học tôn đức thắng',
            'tv': 'thư viện',
            'sv': 'sinh viên',
            'gv': 'Giảng viên',
            'GV-VC': 'Giảng viên-Viên chức',
            'cựu sv': 'Cựu Sinh viên',
            'hvch': 'Học viên cao học',
            'ncs' : 'Ngiên cứu sinh',
            'ht' : 'Hiệu trưởng',
            'csdl': 'Cơ sở dữ liệu',
            'CSDL' : 'Cơ sở dữ liệu',
            'thẻ sv': 'Thẻ sinh viên',
            'thẻ tv': 'Thẻ thư viện',
            'database': 'Cơ sở dữ liệu',
            'copyright': 'Bản quyền',
            'Thư viện đại học Tôn Đức Thắng': 'TVĐHTĐT'
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
        
        # Build normalized questions and aliases for fuzzy matching
        self.norm_questions = [
            (self.super_normalize(self.expand_synonyms(q['question'])), i)
            for i, q in enumerate(self.faq_data)
        ]
        self.norm_aliases = [
            (self.super_normalize(self.expand_synonyms(alias)), i)
            for i, q in enumerate(self.faq_data)
            for alias in q.get('aliases', [])
        ]

        # Build semantic index on FAQ questions
        questions = [entry['question'] for entry in self.faq_data]
        semantic_search.build_index(questions)

        # Define allowed specialized categories for semantic search
        self.allowed_specialized = {'huongdan', 'quydinh', 'muon_tra', 'dichvu', 'lienhe'}

    def remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents from text."""
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def is_blocked_phrase(self, user_message: str, threshold: int = 80) -> bool:
        """Check if user message contains any blocked phrases using fuzzy matching."""
        if not user_message:
            return False
            
        user_message_lower = user_message.lower().strip()
        user_message_no_accents = self.remove_accents(user_message_lower)
        
        # List of library-related terms that should never be blocked
        library_terms = {
            'sách', 'sach', 'book', 'books',
            'phòng', 'phong', 'room', 'rooms',
            'thư viện', 'thu vien', 'library',
            'mượn', 'muon', 'borrow',
            'trả', 'tra', 'return',
            'đọc', 'doc', 'read',
            'tài liệu', 'tai lieu', 'document',
            'giáo trình', 'giao trinh', 'textbook',
            'tạp chí', 'tap chi', 'journal',
            'báo', 'bao', 'newspaper',
            'máy tính', 'may tinh', 'computer',
            'máy in', 'may in', 'printer',
            'wifi', 'internet', 'mạng',
            'đăng ký', 'dang ky', 'register',
            'đăng nhập', 'dang nhap', 'login',
            'tài khoản', 'tai khoan', 'account',
            'mật khẩu', 'mat khau', 'password',
            'thẻ', 'the', 'card',
            'phí', 'phi', 'fee',
            'gia hạn', 'gia han', 'renew',
            'quy định', 'quy dinh', 'rule',
            'dịch vụ', 'dich vu', 'service',
            'hướng dẫn', 'huong dan', 'guide',
            'liên hệ', 'lien he', 'contact'
        }

        # List of sensitive terms that should always be blocked
        sensitive_terms = {
            'nhạy cảm', 'nhay cam', 'sensitive',
            'chính trị', 'chinh tri', 'politics',
            'đảng phái', 'dang phai', 'political party',
            'kinh tế vĩ mô', 'kinh te vi mo', 'macroeconomic',
            'thể dục', 'the duc', 'exercise',
            'gym', 'fitness', 
            'giao thông', 'giao thong', 'traffic',
            'lịch xe buýt', 'lich xe buyt', 'bus schedule',
            'bãi đậu xe', 'bai dau xe', 'parking',
            'lịch thi', 'lich thi', 'exam schedule',
            'bia', 'beer','cầm đồ'
            'nhà hàng', 'nha hang', 'restaurant',
            'căng tin', 'cang tin', 'canteen',
            'ký túc xá', 'ky tuc xa', 'dormitory',
            'nhà ở', 'nha o', 'housing',
            'cồn', 'con', 'alcohol',
            'chất kích thích', 'chat kich thich', 'stimulant',
            'mại dâm', 'mai dam', 'prostitution',
            'xung đột', 'xung dot', 'conflict',
            'quốc gia', 'quoc gia', 'nation',
            'dân tộc', 'dan toc', 'ethnicity'
        }
        
        # First check for sensitive terms - these should always be blocked
        for term in sensitive_terms:
            if term in user_message_lower or term in user_message_no_accents:
                logger.info(f"Blocked sensitive term: '{term}'")
                return True
                
        # Then check exact matches for other blocked phrases
        for phrase in self.no_answer_phrases:
            if phrase in user_message_lower:
                # Skip if the phrase is part of a library term and not a sensitive term
                if any(term in user_message_lower for term in library_terms) and not any(term in user_message_lower for term in sensitive_terms):
                    continue
                return True
                
        # Then check fuzzy matches with higher threshold for longer phrases
        for phrase in self.no_answer_phrases:
            # Skip short phrases to avoid false positives
            if len(phrase) < 4:
                continue
                
            # Skip if the phrase is part of a library term and not a sensitive term
            if any(term in phrase for term in library_terms) and not any(term in phrase for term in sensitive_terms):
                continue
                
            # Calculate threshold based on phrase length
            phrase_threshold = min(threshold, 85) if len(phrase) > 10 else threshold
            
            # Check original phrase
            ratio = fuzz.partial_ratio(phrase, user_message_lower)
            if ratio >= phrase_threshold:
                # Double check that it's not a library term or is a sensitive term
                if not any(term in user_message_lower for term in library_terms) or any(term in user_message_lower for term in sensitive_terms):
                    logger.info(f"Blocked phrase match: '{phrase}' with ratio {ratio}")
                    return True

            # Check phrase without accents
            phrase_no_accents = self.remove_accents(phrase)
            ratio = fuzz.partial_ratio(phrase_no_accents, user_message_no_accents)
            if ratio >= phrase_threshold:
                # Double check that it's not a library term or is a sensitive term
                if not any(term in user_message_no_accents for term in library_terms) or any(term in user_message_no_accents for term in sensitive_terms):
                    logger.info(f"Blocked phrase match (no accents): '{phrase}' with ratio {ratio}")
                    return True

            # Check common typos
            if phrase in self.common_typos:
                for typo in self.common_typos[phrase]:
                    ratio = fuzz.partial_ratio(typo, user_message_lower)
                    if ratio >= phrase_threshold:
                        # Double check that it's not a library term or is a sensitive term
                        if not any(term in user_message_lower for term in library_terms) or any(term in user_message_lower for term in sensitive_terms):
                            logger.info(f"Blocked phrase match (typo): '{phrase}' with ratio {ratio}")
                            return True

            # Only check individual words for longer phrases
            if len(phrase.split()) > 1:
                words = phrase.split()
                for word in words:
                    if len(word) < 4:  # Skip short words
                        continue
                    # Skip if the word is a library term and not a sensitive term
                    if word in library_terms and word not in sensitive_terms:
                        continue
                    ratio = fuzz.partial_ratio(word, user_message_lower)
                    if ratio >= 90:  # Higher threshold for word matches
                        # Double check that it's not part of a library term or is a sensitive term
                        if not any(term in user_message_lower for term in library_terms) or any(term in user_message_lower for term in sensitive_terms):
                            logger.info(f"Blocked word match: '{word}' with ratio {ratio}")
                            return True

        return False

    def super_normalize(self, text: str) -> str:
        """Super normalize text by removing auxiliary phrases and normalizing."""
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', ' ', t)
        for p in self.IGNORED_PATTERNS:
            t = re.sub(p, ' ', t)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    def expand_synonyms(self, text: str) -> str:
        """Expand synonyms in text to their canonical form."""
        t = text.lower()
        for canonical, synonyms in self.SYNONYM_MAP.items():
            for syn in synonyms:
                t = t.replace(syn, canonical)
        return t

    def fuzzy_lookup(self, norm_user: str, norm_questions: List[Tuple[str, int]], threshold: int = 85) -> Optional[int]:
        """Find best fuzzy match using rapidfuzz."""
        best_ratio = 0
        best_idx = None
        
        for norm, idx in norm_questions:
            ratio = fuzz.ratio(norm_user, norm)
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_idx = idx
                
        return best_idx

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

    def find_best_match(self, user_question: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Find the best matching FAQ entry for a user question."""
        # 1. Check for blocked phrases using fuzzy matching
        if self.is_blocked_phrase(user_question):
            logger.info(f"Question blocked: {user_question}")
            return {'question': user_question, 'category': 'unknown', 'confidence': 0.0, '_no_answer': True}

        # 2. Super normalize and expand synonyms
        user_norm = self.super_normalize(self.expand_synonyms(user_question))
        logger.info(f"Normalized question: {user_norm}")

        # 3. Direct matches (exact match after normalization)
        for norm, idx in self.norm_questions + self.norm_aliases:
            if user_norm == norm:
                entry = self.faq_data[idx].copy()
                entry['confidence'] = 1.0
                logger.info(f"Found exact match: {entry['question']}")
                return entry

        # 4. Fuzzy matches
        idx = self.fuzzy_lookup(user_norm, self.norm_questions + self.norm_aliases)
        if idx is not None:
            entry = self.faq_data[idx].copy()
            entry['confidence'] = 0.95
            logger.info(f"Found fuzzy match: {entry['question']}")
            return entry

        # 5. Domain filter: if not in library domain, ignore semantics
        if not self.in_library_domain(user_norm):
            logger.info(f"Question not in library domain: {user_norm}")
            return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

        # 6. Semantic match only for allowed categories
        try:
            results = semantic_search.query(user_question, top_k=1, threshold=threshold)
            logger.info(f"Semantic search results for '{user_question}': {results}")
            
            if results and len(results) > 0:
                idx, score = results[0]['index'], float(results[0]['score'])
                matched = self.faq_data[idx].copy()
                logger.info(f"Top semantic match: idx={idx}, score={score}, category={matched.get('category')}")
                
                # Only accept if category is allowed and score is high enough
                if matched.get('category') in self.allowed_specialized and score >= threshold:
                    matched['confidence'] = score
                    return matched
                else:
                    logger.info(f"Semantic match rejected: category={matched.get('category')}, score={score}")
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}", exc_info=True)

        # 7. No match found
        logger.info(f"No match found for: {user_question}")
        return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

    def get_best_response(self, user_message: str) -> Tuple[str, str, float]:
        match = self.find_best_match(user_message)
        
        # If caught no-answer at yes/no step
        if match.get('_no_answer'):
            return "Xin lỗi, tôi chưa có câu trả lời cho câu hỏi này.", 'unknown', 0.0

        answer = match.get('answer')
        category = match.get('category')
        conf = match.get('confidence', 0.0)

        # 1) FAQ exact/alias
        if conf == 1.0 and answer:
            return answer, category, conf

        # 2) FAQ category keyword but no answer
        if conf == 1.0 and not answer:
            return "Xin lỗi, tôi chưa có câu trả lời cho câu hỏi này.", 'unknown', 0.0

        # 3) Semantic expertise
        allowed = {'huongdan', 'quydinh', 'muon_tra', 'dichvu', 'lienhe'}
        sem_thr = Config.CHAT_SETTINGS.get('semantic_confidence_score', 0.9)
        if category in allowed and conf >= sem_thr:
            return answer or "", category, conf

        # 4) Chat
        if category == 'chitchat':
            ans = Reflection(db_path=db_path).chat(None, user_message)
            return ans, 'chatbot', 0.0

        # 5) Fallback cuối
        return "Xin lỗi, tôi chưa hiểu câu hỏi. Vui lòng thử lại hoặc đặt câu hỏi tại fanpage thư viện (https://www.facebook.com/tvdhtdt) để được giải đáp nhé", 'unknown', 0.0

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
