import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    
    # Server settings
    FLASK_HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT: int = int(os.getenv('FLASK_PORT', 5001))
    DEBUG_MODE: bool = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    # Database settings
    DB_PATH: str = os.getenv('DB_PATH', './chroma_db')
    
    # Rate limiting settings
    RATE_LIMIT: Dict[str, Any] = {
        'requests_per_minute': int(os.getenv('RATE_LIMIT_RPM', 60)),
        'burst_limit': int(os.getenv('RATE_LIMIT_BURST', 10))
    }
    
    # Chat settings
    CHAT_SETTINGS: Dict[str, Any] = {
        'max_message_length': 500,
        'min_confidence_score': 0.6,
        'semantic_confidence_score': 0.95,
        'fuzzy_confidence_score': 0.85,
        'keyword_match_threshold': 0.8,
        'domain_context_threshold': 0.7,
        'default_session_timeout': 3600
    }
    
    # FAQ Categories with descriptions
    CATEGORIES: Dict[str, str] = {
        'muon_tra': 'Mượn trả sách - Thông tin về quy trình mượn và trả sách',
        'quydinh': 'Quy định - Các quy định và nội quy thư viện',
        'huongdan': 'Hướng dẫn - Hướng dẫn sử dụng dịch vụ và tiện ích',
        'dichvu': 'Dịch vụ - Thông tin về các dịch vụ thư viện',
        'lienhe': 'Liên hệ - Thông tin liên hệ và hỗ trợ',
        'chitchat': 'Trò chuyện - Các câu hỏi thông thường'
    }
    
    # Response messages with formatting options
    MESSAGES: Dict[str, str] = {
        'empty_message': 'Tin nhắn không được để trống',
        'unknown_question': ('Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể thử:\n'
                           '- Đặt câu hỏi rõ ràng hơn\n'
                           '- Sử dụng từ khóa liên quan đến thư viện\n'
                           '- Chọn một trong các chủ đề gợi ý'),
        'server_error': 'Xin lỗi, tôi chưa thể đưa ra câu trả lời ngay bây giờ. Vui lòng thử lại sau.',
        'processing_error': 'Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại.',
        'rate_limit_exceeded': 'Quá nhiều yêu cầu. Vui lòng thử lại sau {timeout} giây.',
        'invalid_input': 'Tin nhắn không hợp lệ. {reason}'
    }
    
    # Logging settings
    LOGGING: Dict[str, Any] = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'file_path': 'logs/chatbot_{date}.log'
    }
    
    @classmethod
    def get_category_description(cls, category: str) -> str:
        """Get the description for a given category."""
        return cls.CATEGORIES.get(category, 'Không xác định')
    
    @classmethod
    def format_error_message(cls, message_key: str, **kwargs) -> str:
        """Format an error message with provided parameters."""
        return cls.MESSAGES[message_key].format(**kwargs)
