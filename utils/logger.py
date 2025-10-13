import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    # Create logs directory if needed
    os.makedirs('logs', exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create main application loggers
scraping_logger = setup_logger('scraping', 'scraping.log')
processing_logger = setup_logger('processing', 'processing.log')
model_logger = setup_logger('model', 'model.log')
main_logger = setup_logger('main', 'main.log')
