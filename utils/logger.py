import logging
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
from config import Config

class ChatbotLogger:
    """Enhanced logger for the chatbot application."""
    
    def __init__(self):
        self.logger = None
        self.setup_logger()
    
    def setup_logger(self) -> None:
        """Configure and initialize the logger."""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Get current date for log file name
        current_date = datetime.now().strftime("%Y%m%d")
        log_file = Config.LOGGING['file_path'].format(date=current_date)
        
        # Create logger instance
        logger = logging.getLogger('chatbot')
        logger.setLevel(getattr(logging, Config.LOGGING['level'].upper()))
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            fmt=Config.LOGGING['format'],
            datefmt=Config.LOGGING['date_format']
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def _log_with_context(self, level: int, message: str, 
                         session_id: Optional[str] = None, 
                         **kwargs) -> None:
        """Log a message with additional context."""
        extra = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or 'no_session'
        }
        extra.update(kwargs)
        
        context_message = f"[Session: {extra['session_id']}] {message}"
        if kwargs:
            context_message += f" | Additional context: {kwargs}"
            
        self.logger.log(level, context_message)
    
    def info(self, message: str, session_id: Optional[str] = None, **kwargs) -> None:
        """Log an info message."""
        self._log_with_context(logging.INFO, message, session_id, **kwargs)
    
    def error(self, message: str, session_id: Optional[str] = None, **kwargs) -> None:
        """Log an error message."""
        self._log_with_context(logging.ERROR, message, session_id, **kwargs)
    
    def warning(self, message: str, session_id: Optional[str] = None, **kwargs) -> None:
        """Log a warning message."""
        self._log_with_context(logging.WARNING, message, session_id, **kwargs)
    
    def debug(self, message: str, session_id: Optional[str] = None, **kwargs) -> None:
        """Log a debug message."""
        self._log_with_context(logging.DEBUG, message, session_id, **kwargs)

# Create a global logger instance
chatbot_logger = ChatbotLogger()

def setup_logger():
    """Return the global logger instance."""
    return chatbot_logger 