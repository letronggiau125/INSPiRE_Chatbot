from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ChatResponse:
    message: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'response': self.message,
            'category': self.category,
            'confidence': round(self.confidence, 2) if self.confidence else None,
            'timestamp': self.timestamp
        } 