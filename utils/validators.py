import re
from typing import Optional, Tuple

class MessageValidator:
    @staticmethod
    def sanitize_message(message: str) -> str:
        """Remove any HTML tags and normalize whitespace."""
        # Remove any HTML tags
        message = re.sub(r'<[^>]+>', '', message)
        # Remove multiple spaces
        message = ' '.join(message.split())
        return message.strip()
    
    @staticmethod
    def validate_message(message: str) -> Tuple[bool, Optional[str]]:
        """Validate the message content."""
        if not message:
            return False, "Tin nhắn không được để trống"
        if len(message) > 500:
            return False, "Tin nhắn quá dài (tối đa 500 ký tự)"
        return True, None 