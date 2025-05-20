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
    
    # def to_dict(self):
    #     return {
    #         'response': self.message,
    #         'category': self.category,
    #         'confidence': round(self.confidence, 2) if self.confidence else None,
    #         'timestamp': self.timestamp
    #     } 
    def to_dict(self):
          return {
             'response':  self.message,
             'category':  self.category,
            # Nếu self.confidence là None (không gán), giữ None,
            # còn nếu là số (kể cả 0.0) thì làm tròn 2 chữ số
             'confidence': round(self.confidence, 2) if self.confidence is not None else None,
             'timestamp': self.timestamp
    }