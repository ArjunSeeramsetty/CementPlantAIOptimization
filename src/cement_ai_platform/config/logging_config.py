# src/cement_ai_platform/config/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

def setup_logging(name: str = __name__):
    """
    Setup centralized logging configuration with file rotation and console output.
    
    Args:
        name (str): Logger name, defaults to module name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from environment variable
    log_level = os.getenv("CEMENT_LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Format string
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    
    # Custom formatter that handles Unicode properly
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            # Convert the record to string and handle Unicode
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Fallback: encode problematic characters
                msg = super().format(record)
                return msg.encode('ascii', 'replace').decode('ascii')
    
    formatter = UnicodeFormatter(fmt)
    
    # File handler with rotation (10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        "logs/app.log", 
        maxBytes=10_485_760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Set encoding to UTF-8 to handle Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Fallback if reconfigure is not available
    
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = __name__):
    """
    Get a logger instance with centralized configuration.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logging(name)
