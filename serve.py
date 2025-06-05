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
import numpy as np

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
CORS(app)  # Allow all origins during development

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
            'nhà hàng','căng tin','ký túc xá', 'nhà ở','cồn','chất kích thích','mại dâm','xung đột','quốc gia', 'dân tộc','how do you work', 'bot hoạt động như thế nào','giết','sách cấm'
            'giao thông','lịch xe buýt','bãi đậu xe','lịch thi','bia','politics', 'chính trị', 'đảng phái', 'kinh tế vĩ mô','exercise', 'thể dục', 'gym','nhạy cảm','chém','đâm','cách mạng'
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
            'tìm': ['search', 'tìm kiếm', 'locate'],
            'học qua đêm': ['học ban đêm', 'học khuya', 'học tối', 'học đêm', 'học xuyên đêm'],
            'khu vực học qua đêm': ['khu vực học ban đêm', 'khu vực học khuya', 'phòng học qua đêm', 'phòng học ban đêm', 'phòng học khuya'],
            'đăng ký học qua đêm': ['đăng ký học ban đêm', 'đăng ký học khuya', 'đặt phòng học qua đêm', 'đặt phòng học ban đêm', 'đặt phòng học khuya']
        }

        # Library domain keywords
        self.LIBRARY_KEYWORDS = {
            # Basic library terms
            'thư viện', 'library', 'thu vien',
            'sách', 'book', 'books', 'sach',
            'tài liệu', 'tai lieu', 'document', 'documents',
            'giáo trình', 'giao trinh', 'textbook', 'textbooks',
            'tạp chí', 'tap chi', 'journal', 'journals',
            'báo', 'bao', 'newspaper', 'newspapers',
            
            # Library services
            'mượn', 'muon', 'borrow', 'borrowing',
            'trả', 'tra', 'return', 'returning',
            'đọc', 'doc', 'read', 'reading',
            'wifi', 'internet', 'mạng', 'mang',
            'máy tính', 'may tinh', 'computer', 'computers',
            'máy in', 'may in', 'printer', 'printers',
            'máy quét', 'may quet', 'scanner', 'scanners',
            'photo', 'photocopy', 'copy', 'copying',
            
            # Library operations
            'giờ mở cửa', 'gio mo cua', 'opening hours', 'opening time',
            'giờ đóng cửa', 'gio dong cua', 'closing hours', 'closing time',
            'đăng ký', 'dang ky', 'register', 'registration',
            'đăng nhập', 'dang nhap', 'login', 'sign in',
            'tài khoản', 'tai khoan', 'account', 'accounts',
            'mật khẩu', 'mat khau', 'password', 'passwords',
            'thẻ', 'the', 'card', 'cards',
            'phí', 'phi', 'fee', 'fees',
            'gia hạn', 'gia han', 'renew', 'renewal',
            
            # Library spaces
            'phòng', 'phong', 'room', 'rooms',
            'khu vực', 'khu vuc', 'area', 'areas',
            'tầng', 'tang', 'floor', 'floors',
            'tòa nhà', 'toa nha', 'building', 'buildings',
            
            # Library users
            'sinh viên', 'sinh vien', 'student', 'students',
            'giảng viên', 'giang vien', 'lecturer', 'lecturers',
            'cán bộ', 'can bo', 'staff', 'staffs',
            
            # Library rules
            'quy định', 'quy dinh', 'rule', 'rules',
            'nội quy', 'noi quy', 'regulation', 'regulations',
            'hướng dẫn', 'huong dan', 'guide', 'guidelines',
            
            # Library resources
            'cơ sở dữ liệu', 'co so du lieu', 'database', 'databases',
            'tài nguyên', 'tai nguyen', 'resource', 'resources',
            'tài liệu số', 'tai lieu so', 'digital resource', 'digital resources',
            
            # Library services
            'dịch vụ', 'dich vu', 'service', 'services',
            'hỗ trợ', 'ho tro', 'support', 'assistance',
            'tư vấn', 'tu van', 'consultation', 'advice',
            
            # Library locations
            'cơ sở', 'co so', 'campus', 'campuses',
            'chi nhánh', 'chi nhanh', 'branch', 'branches',
            
            # Library operations
            'mở cửa', 'mo cua', 'open', 'opening',
            'đóng cửa', 'dong cua', 'close', 'closing',
            'nghỉ', 'nghi', 'closed', 'holiday',
            'làm việc', 'lam viec', 'working', 'operating',
            
            # Library specific
            'INSPiRE', 'inspire', 'thư viện truyền cảm hứng',
            'TVĐHTĐT', 'thư viện đại học tôn đức thắng',
            'TDTU', 'tdtu', 'ton duc thang'
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
            'chitchat': [r'(hello|hi|hello|goodbye|bye).*'],
            'plagiarism': [
                r'(kiểm tra|check).*(trùng lặp|đạo văn|sao chép)',
                r'(có|hỗ trợ).*(kiểm tra|check).*(trùng lặp|đạo văn)',
                r'(cách|làm sao).*(kiểm tra|check).*(trùng lặp|đạo văn)'
            ],
            'publication': [
                r'(công bố|bài báo).*(quốc tế|international)',
                r'(kiểm tra|check).*(công bố|bài báo)',
                r'(hỗ trợ|support).*(công bố|publication)'
            ]
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
        
        # List of university-related terms that should never be blocked
        university_terms = {
            'tdtu', 'ton duc thang', 'trường đại học tôn đức thắng',
            'fibaa', 'hcéres', 'aacsb', 'abest', 'acbsp',
            'chứng nhận', 'chung nhan', 'certification',
            'kiểm định', 'kiem dinh', 'accreditation',
            'xếp hạng', 'xep hang', 'ranking',
            'đánh giá', 'danh gia', 'assessment',
            'chất lượng', 'chat luong', 'quality',
            'đào tạo', 'dao tao', 'education',
            'giảng dạy', 'giang day', 'teaching',
            'nghiên cứu', 'nghien cuu', 'research',
            'khoa học', 'khoa hoc', 'science',
            'quốc tế', 'quoc te', 'international'
        }

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
            'liên hệ', 'lien he', 'contact',
            'công nghệ', 'cong nghe', 'technology',
            'hệ thống', 'he thong', 'system',
            'phần mềm', 'phan mem', 'software',
            'ứng dụng', 'ung dung', 'application',
            'còn', 'con', 'still', 'continue',
            'mở cửa', 'mo cua', 'open',
            'đóng cửa', 'dong cua', 'close',
            'giờ', 'gio', 'time', 'hour',
            'thời gian', 'thoi gian', 'time'
        }

        # Combine all protected terms
        protected_terms = library_terms.union(university_terms)
        
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
            'bia', 'beer', 'cầm đồ',
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
            # Use word boundary check to prevent partial matches
            # Add Vietnamese word boundary check
            if re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', user_message_lower) or \
               re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', user_message_no_accents):
                # Double check that it's not a protected term
                if not any(prot_term in user_message_lower for prot_term in protected_terms):
                    logger.info(f"Blocked sensitive term: '{term}'")
                    return True
                
        # Then check exact matches for other blocked phrases
        for phrase in self.no_answer_phrases:
            # Skip if the phrase is part of a protected term
            if any(term in user_message_lower for term in protected_terms):
                continue
                
            # Use word boundary check to prevent partial matches
            # Add Vietnamese word boundary check
            if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', user_message_lower):
                logger.info(f"Blocked phrase match: '{phrase}'")
                return True
                
        # Then check fuzzy matches with higher threshold for longer phrases
        for phrase in self.no_answer_phrases:
            # Skip short phrases to avoid false positives
            if len(phrase) < 4:
                continue
                
            # Skip if the phrase is part of a protected term
            if any(term in phrase for term in protected_terms):
                continue
                
            # Calculate threshold based on phrase length and content
            phrase_threshold = min(threshold, 85) if len(phrase) > 10 else threshold
            
            # Increase threshold for questions containing special characters or acronyms
            if re.search(r'[A-Z]', user_message) or re.search(r'[^a-z0-9\s]', user_message):
                phrase_threshold = min(phrase_threshold + 5, 95)
            
            # Check original phrase
            ratio = fuzz.partial_ratio(phrase, user_message_lower)
            if ratio >= phrase_threshold:
                # Double check that it's not a protected term
                if not any(term in user_message_lower for term in protected_terms):
                    logger.info(f"Blocked phrase fuzzy match: '{phrase}' with ratio {ratio}")
                    return True
                    
            # Check phrase without accents
            phrase_no_accents = self.remove_accents(phrase)
            ratio = fuzz.partial_ratio(phrase_no_accents, user_message_no_accents)
            if ratio >= phrase_threshold:
                # Double check that it's not a protected term
                if not any(term in user_message_no_accents for term in protected_terms):
                    logger.info(f"Blocked phrase fuzzy match (no accents): '{phrase}' with ratio {ratio}")
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
        # First expand using existing synonym map
        for canonical, synonyms in self.SYNONYM_MAP.items():
            for syn in synonyms:
                t = t.replace(syn, canonical)
        
        # Then expand using new synonym mappings
        new_synonyms = {
            "mượn": ["borrow", "take", "checkout"],
            "trả": ["return", "give back"],
            "sách": ["book", "tài liệu"]
        }
        
        for word, syns in new_synonyms.items():
            for syn in syns:
                if syn in t:
                    t = t.replace(syn, word)
                    
        return t

    def fuzzy_lookup(self, norm_user: str, norm_questions: List[Tuple[str, int]], threshold: int = 85) -> Optional[int]:
        """Find best fuzzy match using rapidfuzz with semantic relevance check."""
        candidates = [q[0] for q in norm_questions]
        
        # First try exact match
        for norm, idx in norm_questions:
            if norm == norm_user:
                return idx
                
        # Then try token set ratio for partial matches
        result = process.extractOne(
            norm_user, 
            candidates, 
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold
        )
        
        if result:
            matched_text, score, _ = result
            
            # Additional semantic relevance check
            if score >= threshold:
                # Extract key terms from both texts
                user_terms = set(norm_user.split())
                matched_terms = set(matched_text.split())
                
                # Check for key term overlap
                common_terms = user_terms.intersection(matched_terms)
                
                # Define topic-specific terms with more specific keywords
                topic_keywords = {
                    'visit': {
                        'tham quan', 'thăm', 'visit', 'tour', 'khách', 'người ngoài', 'outsider',
                        'khách ngoài trường', 'khách thăm', 'tour guide', 'hướng dẫn viên',
                        'đăng ký tham quan', 'đặt lịch tham quan', 'lịch tham quan'
                    },
                    'wifi': {
                        'wifi', 'internet', 'mạng', 'mật khẩu', 'password', 'kết nối', 'connect',
                        'truy cập wifi', 'mật khẩu wifi', 'kết nối mạng', 'internet access'
                    },
                    'borrow': {
                        'mượn', 'trả', 'sách', 'tài liệu', 'borrow', 'return', 'book',
                        'đem về', 'mang về', 'take home', 'bring home'
                    },
                    'access': {
                        'thẻ', 'card', 'truy cập', 'access', 'vào', 'sử dụng', 'use',
                        'thẻ sinh viên', 'thẻ thư viện', 'library card', 'student card'
                    },
                    'facility': {
                        'phòng', 'room', 'khu vực', 'area', 'cơ sở vật chất', 'facility',
                        'phòng học', 'phòng nghiên cứu', 'phòng thuyết trình'
                    },
                    'service': {
                        'dịch vụ', 'service', 'hỗ trợ', 'support', 'tư vấn', 'consultation',
                        'scan', 'photocopy', 'copy', 'in ấn', 'printing'
                    },
                    'leadership': {
                        'hiệu trưởng', 'hiệu phó', 'trưởng', 'phó', 'chủ tịch', 'giám đốc',
                        'dean', 'principal', 'president', 'director'
                    },
                    'night_study': {
                        'học qua đêm', 'học ban đêm', 'học khuya', 'học tối', 'học đêm',
                        'khu vực học qua đêm', 'khu vực học ban đêm', 'khu vực học khuya',
                        'phòng học qua đêm', 'phòng học ban đêm', 'phòng học khuya',
                        'đăng ký học qua đêm', 'đăng ký học ban đêm', 'đăng ký học khuya',
                        'thời gian học qua đêm', 'thời gian học ban đêm', 'thời gian học khuya'
                    },
                    'plagiarism': {
                        'kiểm tra trùng lặp', 'kiểm tra đạo văn', 'check đạo văn', 
                        'turnitin', 'ithenticate', 'plagiarism', 'trùng lặp',
                        'sao chép', 'copy', 'duplicate', 'check trùng lặp',
                        'kiểm tra sao chép', 'kiểm tra bài báo', 'check bài báo'
                    },
                    'publication': {
                        'công bố', 'bài báo', 'tạp chí', 'journal', 'publication',
                        'nghiên cứu', 'research', 'paper', 'article', 'bài viết',
                        'công bố quốc tế', 'international publication'
                    }
                }
                
                # Check which topics are present in user question
                user_topics = set()
                for topic, terms in topic_keywords.items():
                    if any(term in user_terms for term in terms):
                        user_topics.add(topic)
                
                # Check which topics are present in matched question
                matched_topics = set()
                for topic, terms in topic_keywords.items():
                    if any(term in matched_terms for term in terms):
                        matched_topics.add(topic)
                
                # Require at least one common topic
                if not user_topics.intersection(matched_topics):
                    logger.info(f"Rejected fuzzy match due to topic mismatch. User topics: {user_topics}, Matched topics: {matched_topics}")
                    return None
                
                # Special handling for specific topics
                if 'visit' in user_topics:
                    if 'visit' not in matched_topics:
                        logger.info("Rejected fuzzy match for visit question - no visit-related terms")
                        return None
                    # For visit questions, require higher threshold and more specific terms
                    if score < 90:
                        logger.info("Rejected fuzzy match for visit question - score too low")
                        return None
                    # Additional check for visit-specific terms
                    visit_specific_terms = {'khách ngoài trường', 'người ngoài', 'outsider', 'tour guide'}
                    if any(term in user_terms for term in visit_specific_terms):
                        if not any(term in matched_terms for term in visit_specific_terms):
                            logger.info("Rejected fuzzy match for outsider visit question - no specific terms")
                            return None
                
                if 'wifi' in user_topics:
                    if 'wifi' not in matched_topics:
                        logger.info("Rejected fuzzy match for wifi question - no wifi-related terms")
                        return None
                    # For wifi questions, require more specific terms
                    wifi_specific_terms = {'mật khẩu', 'password', 'truy cập', 'access'}
                    if any(term in user_terms for term in wifi_specific_terms):
                        if not any(term in matched_terms for term in wifi_specific_terms):
                            logger.info("Rejected fuzzy match for wifi access question - no specific terms")
                            return None
                
                if 'borrow' in user_topics:
                    if 'borrow' not in matched_topics:
                        logger.info("Rejected fuzzy match for borrowing question - no borrow-related terms")
                        return None
                
                if 'leadership' in user_topics:
                    if 'leadership' not in matched_topics:
                        logger.info("Rejected fuzzy match for leadership question - no leadership-related terms")
                        return None
                    # For leadership questions, require higher threshold
                    if score < 90:
                        logger.info("Rejected fuzzy match for leadership question - score too low")
                        return None
                
                # Require more common terms for better matching
                if len(common_terms) >= 2:  # Require at least 2 common terms
                    # Find the index of the matched question
                    for norm, idx in norm_questions:
                        if norm == matched_text:
                            return idx
                            
        return None

    def in_library_domain(self, text: str) -> bool:
        """Check if the text is related to library domain using comprehensive keyword matching."""
        # Normalize the text
        text = self.normalize_text(text)
        
        # Check for exact keyword matches
        for kw in self.LIBRARY_KEYWORDS:
            if kw in text:
                return True
                
        # Check for compound terms (e.g., "thư viện trường", "thư viện tdtu")
        compound_terms = [
            'thư viện trường', 'thư viện tdtu', 'thư viện đại học',
            'library tdtu', 'tdtu library', 'university library'
        ]
        for term in compound_terms:
            if term in text:
                return True
                
        # Check for common library-related phrases
        library_phrases = [
            'mượn sách', 'trả sách', 'đọc sách',
            'thẻ thư viện', 'thẻ sinh viên',
            'giờ mở cửa', 'giờ đóng cửa',
            'đăng ký thẻ', 'gia hạn sách'
        ]
        for phrase in library_phrases:
            if phrase in text:
                return True
                
        return False

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
        # 1) Strip and lowercase
        text = text.strip().lower()
        
        # 2) Normalize unicode composition
        text = unicodedata.normalize('NFC', text)
        
        # 3) Remove Vietnamese accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                      if unicodedata.category(c) != 'Mn')
        
        # 4) Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 5) Expand abbreviations using existing map
        for abbr, full in self.abbrev_map.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
            
        # 6) Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def build_faq_map(self) -> Dict[str, Dict[str, Any]]:
        """Create lookup table for questions and aliases with normalized keys."""
        faq_map = {}
        for entry in self.faq_data:
            # Get all questions including aliases
            all_questions = [entry['question']] + entry.get('aliases', [])
            
            # Add each question and its normalized form to the map
            for q in all_questions:
                # Store original question
                faq_map[q.lower()] = entry
                
                # Store normalized version
                norm_q = self.normalize_text(q)
                faq_map[norm_q] = entry
                
                # Generate and store paraphrases
                paraphrases = self.generate_paraphrases(q)
                for p in paraphrases:
                    faq_map[p.lower()] = entry
                    faq_map[self.normalize_text(p)] = entry
                    
        return faq_map

    def generate_paraphrases(self, question: str) -> List[str]:
        """Generate paraphrases for a question to improve matching."""
        paraphrases = []
        
        # Basic template-based paraphrases
        templates = [
            "Tôi muốn biết {q}",
            "Cho tôi biết {q}",
            "Bạn có thể cho tôi biết {q}",
            "Xin hỏi {q}",
            "Tôi cần biết {q}",
            "Làm sao để {q}",
            "Cách {q}",
            "Hướng dẫn {q}",
            "Giải thích {q}",
            "Thông tin về {q}"
        ]
        
        # Add template-based paraphrases
        for template in templates:
            paraphrases.append(template.format(q=question.lower()))
            
        # Add common Vietnamese variations
        variations = {
            "làm thế nào": ["cách nào", "làm sao", "làm gì", "phải làm gì"],
            "tôi muốn": ["tôi cần", "tôi muốn biết", "tôi cần biết"],
            "cho tôi": ["cho em", "cho mình", "cho tôi biết"], 
            "xin hỏi": ["cho hỏi", "cho tôi hỏi", "tôi muốn hỏi"]
        }
        
        # Add variation-based paraphrases
        for original, replacements in variations.items():
            if original in question.lower():
                for replacement in replacements:
                    paraphrases.append(question.lower().replace(original, replacement))
                    
        return list(set(paraphrases))  # Remove duplicates

    def normalize_abbreviations(self, text: str) -> str:
        q = text.lower().strip()
        # Remove trailing punctuation
        q = re.sub(r'[?!.]+$', '', q)
        # Expand abbreviations
        for abbr, full in self.abbrev_map.items():
            q = re.sub(rf"\b{re.escape(abbr)}\b", full, q)
        return q

    def semantic_search(self, user_question: str, threshold: float = 0.75) -> Optional[Tuple[int, float]]:
        """Find closest question based on semantic similarity (cosine)."""
        try:
            # Add specific handling for plagiarism and publication questions
            if any(term in user_question.lower() for term in ['trùng lặp', 'đạo văn', 'sao chép', 'plagiarism']):
                # Increase threshold for plagiarism-related questions
                threshold = 0.85
                
            if any(term in user_question.lower() for term in ['công bố', 'bài báo', 'publication']):
                # Increase threshold for publication-related questions
                threshold = 0.85
            
            # Get user question embedding
            user_emb = embedding_model.get_embedding(self.normalize_text(user_question))
            
            # Get all question embeddings
            question_embeddings = []
            for entry in self.faq_data:
                question_emb = embedding_model.get_embedding(self.normalize_text(entry['question']))
                question_embeddings.append(question_emb)
                
            # Calculate similarities
            scores = []
            for emb in question_embeddings:
                # Calculate cosine similarity
                dot_product = np.dot(user_emb, emb)
                norm_user = np.linalg.norm(user_emb)
                norm_emb = np.linalg.norm(emb)
                similarity = dot_product / (norm_user * norm_emb)
                scores.append(similarity)
                
            # Find best match
            best_idx = int(np.argmax(scores))
            
            # Additional relevance check
            if scores[best_idx] >= threshold:
                # Check for specific question types
                user_text = self.normalize_text(user_question)
                matched_text = self.normalize_text(self.faq_data[best_idx]['question'])
                
                # Define topic keywords with more specific terms
                topic_keywords = {
                    'hours': {'giờ', 'thời gian', 'mở cửa', 'đóng cửa', 'hoạt động', 'cuối tuần', 'weekend'},
                    'wifi': {'wifi', 'internet', 'mạng', 'mật khẩu', 'password'},
                    'borrow': {
                        'mượn', 'trả', 'sách', 'tài liệu', 'borrow', 'return', 'book',
                        'đem về', 'mang về', 'take home', 'bring home',
                        'gia hạn', 'renew', 'extension',
                        'thời hạn', 'deadline', 'due date',
                        'số lượng', 'quantity', 'limit',
                        'phí', 'fee', 'charge'
                    },
                    'access': {'thẻ', 'card', 'truy cập', 'access', 'vào', 'sử dụng', 'use'},
                    'student': {'sinh viên', 'student', 'tân sinh viên', 'new student', 'học viên'},
                    'facility': {'phòng', 'room', 'khu vực', 'area', 'cơ sở vật chất', 'facility'},
                    'service': {
                        'dịch vụ', 'service', 'hỗ trợ', 'support', 'tư vấn', 'consultation',
                        'scan', 'photocopy', 'copy', 'in ấn', 'printing',
                        'máy in', 'printer', 'máy quét', 'scanner'
                    },
                    'leadership': {
                        'hiệu trưởng', 'hiệu phó', 'trưởng', 'phó', 'chủ tịch', 'giám đốc',
                        'dean', 'principal', 'president', 'director'
                    },
                    'night_study': {
                        'học qua đêm', 'học ban đêm', 'học khuya', 'học tối', 'học đêm',
                        'khu vực học qua đêm', 'khu vực học ban đêm', 'khu vực học khuya',
                        'phòng học qua đêm', 'phòng học ban đêm', 'phòng học khuya',
                        'đăng ký học qua đêm', 'đăng ký học ban đêm', 'đăng ký học khuya',
                        'thời gian học qua đêm', 'thời gian học ban đêm', 'thời gian học khuya'
                    },
                    'plagiarism': {
                        'kiểm tra trùng lặp', 'kiểm tra đạo văn', 'check đạo văn', 
                        'turnitin', 'ithenticate', 'plagiarism', 'trùng lặp',
                        'sao chép', 'copy', 'duplicate', 'check trùng lặp',
                        'kiểm tra sao chép', 'kiểm tra bài báo', 'check bài báo'
                    },
                    'publication': {
                        'công bố', 'bài báo', 'tạp chí', 'journal', 'publication',
                        'nghiên cứu', 'research', 'paper', 'article', 'bài viết',
                        'công bố quốc tế', 'international publication'
                    }
                }
                
                # Check which topics are present in user question
                user_topics = set()
                for topic, keywords in topic_keywords.items():
                    if any(kw in user_text for kw in keywords):
                        user_topics.add(topic)
                
                # Check which topics are present in matched answer
                matched_topics = set()
                for topic, keywords in topic_keywords.items():
                    if any(kw in matched_text for kw in keywords):
                        matched_topics.add(topic)
                
                # Require at least one common topic
                if not user_topics.intersection(matched_topics):
                    logger.info(f"Rejected match due to topic mismatch. User topics: {user_topics}, Matched topics: {matched_topics}")
                    return None, None
                
                # Special handling for borrowing questions
                if 'borrow' in user_topics:
                    if 'borrow' not in matched_topics:
                        logger.info("Rejected match for borrowing question - no borrow-related topics")
                        return None, None
                    # Additional check for take-home related terms
                    take_home_terms = {'đem về', 'mang về', 'take home', 'bring home'}
                    if any(term in user_text for term in take_home_terms):
                        if not any(term in matched_text for term in take_home_terms):
                            logger.info("Rejected match for take-home question - no take-home related terms")
                            return None, None
                
                # Special handling for student access questions
                if 'student' in user_topics and 'access' in user_topics:
                    if not ('student' in matched_topics or 'access' in matched_topics):
                        logger.info("Rejected match for student access question - no relevant topics")
                        return None, None
                
                logger.info(f"Found semantic match with score {scores[best_idx]}: {self.faq_data[best_idx]['question']}")
                logger.info(f"Topics - User: {user_topics}, Matched: {matched_topics}")
                return best_idx, scores[best_idx]
                
            return None, None
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}", exc_info=True)
            return None, None

    def _find_publication_plagiarism_match(self, user_question: str) -> Dict[str, Any]:
        """Find best match for publication plagiarism checking questions."""
        # Normalize the question
        user_norm = self.super_normalize(self.expand_synonyms(user_question))
        
        # Define specific keywords for publication plagiarism
        plagiarism_keywords = {
            'trùng lặp', 'đạo văn', 'sao chép', 'plagiarism',
            'kiểm tra', 'check', 'verify', 'xác minh'
        }
        publication_keywords = {
            'công bố', 'bài báo', 'publication', 'paper',
            'quốc tế', 'international', 'journal', 'article'
        }
        
        # Check if question contains both plagiarism and publication terms
        has_plagiarism = any(kw in user_norm for kw in plagiarism_keywords)
        has_publication = any(kw in user_norm for kw in publication_keywords)
        
        if not (has_plagiarism and has_publication):
            # If missing either term, return no match
            return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}
            
        # Search for exact matches first
        for norm, idx in self.norm_questions + self.norm_aliases:
            if norm == user_norm:
                entry = self.faq_data[idx].copy()
                entry['confidence'] = 1.0
                return entry
                
        # Try fuzzy matching with higher threshold
        idx = self.fuzzy_lookup(user_norm, self.norm_questions + self.norm_aliases, threshold=90)
        if idx is not None:
            entry = self.faq_data[idx].copy()
            entry['confidence'] = 0.95
            return entry
            
        # Try semantic search with higher threshold
        try:
            idx, score = self.semantic_search(user_question, threshold=0.85)
            if idx is not None:
                matched = self.faq_data[idx].copy()
                matched['confidence'] = score
                return matched
        except Exception as e:
            logger.error(f"Error in semantic search for publication plagiarism: {e}")
            
        # No match found
        return {'question': user_question, 'category': 'unknown', 'confidence': 0.0}

    def find_best_match(self, user_question: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Find the best matching FAQ entry for a user question."""
        # Add context awareness for plagiarism and publication questions
        if any(term in user_question.lower() for term in ['trùng lặp', 'đạo văn', 'sao chép', 'plagiarism']):
            # Check if it's related to publications
            if any(term in user_question.lower() for term in ['công bố', 'bài báo', 'publication']):
                # Look for specific plagiarism checking for publications
                return self._find_publication_plagiarism_match(user_question)
            
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

        # 4. Fuzzy matches with semantic relevance check
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

        # 6. Semantic match using cosine similarity with topic relevance check
        try:
            idx, score = self.semantic_search(user_question, threshold=threshold)
            if idx is not None:
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
        
        # Nếu ở bước yes/no mà không có câu trả lời
        if match.get('_no_answer'):
            return "Xin lỗi, tôi chưa có câu trả lời cho câu hỏi này.", 'unknown', 0.0

        answer = match.get('answer')
        category = match.get('category')
        conf = match.get('confidence', 0.0)

        # 1) FAQ exact/alias hoặc fuzzy match với answer
        if (conf >= 0.95 or conf == 1.0) and answer:
            return answer, category, conf

        # 2) FAQ category keyword nhưng không có answer
        if conf >= 0.95 and not answer:
            fallback_md = (
                "Xin lỗi, tôi chưa hoàn toàn hiểu câu hỏi của bạn, "
                "Bạn có thể trình bày câu hỏi một cách rõ ràng hơn không "
                "hoặc để lại câu hỏi trên live chat "
                "[Fanpage thư viện](https://www.facebook.com/tvdhtdt)"
                "để nhận được phản hồi nhé"
            )
            return fallback_md, 'unknown', 0.0

        # 3) Semantic expertise cho các category liên quan thư viện
        allowed = {'huongdan', 'quydinh', 'muon_tra', 'dichvu', 'lienhe'}
        sem_thr = Config.CHAT_SETTINGS.get('semantic_confidence_score', 0.9)
        if category in allowed and conf >= sem_thr:
            return answer or "", category, conf

        # 4) Chat cho câu hỏi thông thường (chitchat)
        if category == 'chitchat':
            return answer or "", category, conf

        # 5) Fallback cuối
        fallback_universal = (
            "Xin lỗi, tôi chưa hiểu câu hỏi. Vui lòng thử lại hoặc "
            "đặt câu hỏi tại [Fanpage thư viện](https://www.facebook.com/tvdhtdt) "
            "để được giải đáp nhé"
        )
        return fallback_universal, 'unknown', 0.0

    def validate_match(self, user_question: str, matched_question: str, score: float) -> bool:
        # Add specific validation for plagiarism and publication questions
        if any(term in user_question.lower() for term in ['trùng lặp', 'đạo văn', 'sao chép', 'plagiarism']):
            # Require both plagiarism and publication terms for better accuracy
            has_plagiarism = any(term in user_question.lower() for term in ['trùng lặp', 'đạo văn', 'sao chép', 'plagiarism'])
            has_publication = any(term in user_question.lower() for term in ['công bố', 'bài báo', 'publication'])
            
            if not (has_plagiarism and has_publication):
                return False
            
        return True

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
        # Ignore session_id since we don't want to maintain chat history
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
