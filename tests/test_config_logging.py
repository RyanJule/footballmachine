import pytest
import os
import tempfile
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestConfiguration:
    """Test configuration loading and validation"""
    
    def test_config_module_exists(self):
        """Config module should be importable"""
        import config
        assert config is not None
    
    def test_database_url_default(self):
        """Database should default to SQLite if no env var"""
        from config.database import DATABASE_URL
        assert 'sqlite' in DATABASE_URL.lower()
    
    def test_database_url_from_env(self):
        """Database URL should be overridable via environment"""
        os.environ['DATABASE_URL'] = 'postgresql://test/db'
        # Need to reimport to pick up new env var
        import importlib
        import config.database
        importlib.reload(config.database)
        
        assert config.database.DATABASE_URL == 'postgresql://test/db'
        
        # Cleanup
        del os.environ['DATABASE_URL']
        importlib.reload(config.database)
    
    def test_scraping_config_has_required_fields(self):
        """Scraping config should have all required fields"""
        from config.scraping import BASE_URL, SELENIUM_CONFIG, MIN_REQUEST_DELAY
        
        assert BASE_URL == 'https://www.pro-football-reference.com'
        assert 'headless' in SELENIUM_CONFIG
        assert MIN_REQUEST_DELAY > 0
    
    def test_model_config_has_required_fields(self):
        """Model config should match specification"""
        from config.model import LSTM_CONFIG
        
        assert LSTM_CONFIG['player_features'] == 670
        assert LSTM_CONFIG['roster_size'] == 64
        assert LSTM_CONFIG['sequence_length'] == 20


class TestLogging:
    """Test logging functionality - Windows compatible"""
    
    def test_logger_setup_function_exists(self):
        """Logger setup function should be importable"""
        from utils.logger import setup_logger
        assert callable(setup_logger)
    
    def test_logger_creates_log_file(self):
        """Logger should create log files in logs directory"""
        from utils.logger import setup_logger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                logger = setup_logger('test_create', 'test_create.log')
                logger.info("Test message")
                
                # Flush and close handlers to release file
                for handler in logger.handlers[:]:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                
                # Check log file was created
                assert Path('logs/test_create.log').exists()
                
                # Check content was written
                with open('logs/test_create.log', 'r') as f:
                    content = f.read()
                    assert 'Test message' in content
                    
            finally:
                # Cleanup logger
                logging.getLogger('test_create').handlers.clear()
                os.chdir(original_cwd)
    
    def test_logger_has_file_and_console_handlers(self):
        """Logger should output to both file and console"""
        from utils.logger import setup_logger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                logger = setup_logger('test_handlers', 'test_handlers.log')
                
                # Check handlers
                handlers = logger.handlers
                assert len(handlers) == 2
                
                handler_types = [type(h).__name__ for h in handlers]
                assert 'RotatingFileHandler' in handler_types
                assert 'StreamHandler' in handler_types
                
                # Cleanup
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
                    
            finally:
                logging.getLogger('test_handlers').handlers.clear()
                os.chdir(original_cwd)
    
    def test_main_loggers_exist(self):
        """All main application loggers should exist"""
        # Import in a way that doesn't create files in test directory
        import sys
        import importlib
        
        # Temporarily modify the logger module to not create files
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            try:
                # Reload logger module in temp directory
                if 'utils.logger' in sys.modules:
                    importlib.reload(sys.modules['utils.logger'])
                
                from utils.logger import (
                    scraping_logger,
                    processing_logger,
                    model_logger,
                    main_logger
                )
                
                assert scraping_logger is not None
                assert processing_logger is not None
                assert model_logger is not None
                assert main_logger is not None
                
                # Cleanup all handlers
                for logger in [scraping_logger, processing_logger, model_logger, main_logger]:
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
                        
            finally:
                os.chdir(original_cwd)
    
    def test_logger_log_levels(self):
        """Logger should respect log levels"""
        from utils.logger import setup_logger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                logger = setup_logger('test_levels', 'test_levels.log', level=logging.WARNING)
                
                logger.debug("Debug message")
                logger.info("Info message")
                logger.warning("Warning message")
                
                # Flush and close before reading
                for handler in logger.handlers[:]:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                
                with open('logs/test_levels.log', 'r') as f:
                    content = f.read()
                    assert 'Debug message' not in content
                    assert 'Info message' not in content
                    assert 'Warning message' in content
                    
            finally:
                logging.getLogger('test_levels').handlers.clear()
                os.chdir(original_cwd)


class TestErrorHandling:
    """Test error handling utilities"""
    
    def test_custom_exceptions_exist(self):
        """Custom exception classes should be defined"""
        from utils.error_handler import (
            DataValidationError,
            ScrapingError,
            ModelError
        )
        
        assert issubclass(DataValidationError, Exception)
        assert issubclass(ScrapingError, Exception)
        assert issubclass(ModelError, Exception)
    
    def test_custom_exceptions_can_be_raised(self):
        """Custom exceptions should be raiseable"""
        from utils.error_handler import DataValidationError, ScrapingError, ModelError
        
        with pytest.raises(DataValidationError):
            raise DataValidationError("Test error")
        
        with pytest.raises(ScrapingError):
            raise ScrapingError("Test error")
        
        with pytest.raises(ModelError):
            raise ModelError("Test error")
    
    def test_retry_decorator_exists(self):
        """Retry decorator should be importable"""
        from utils.error_handler import retry_with_backoff
        assert callable(retry_with_backoff)
    
    def test_retry_decorator_retries_on_failure(self):
        """Retry decorator should retry failed functions"""
        from utils.error_handler import retry_with_backoff
        
        call_count = {'count': 0}
        
        @retry_with_backoff(max_retries=3, backoff_factor=0.01)  # Fast backoff for testing
        def failing_function():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise ValueError("Temporary failure")
            return "Success"
        
        result = failing_function()
        
        assert result == "Success"
        assert call_count['count'] == 3
    
    def test_retry_decorator_raises_after_max_retries(self):
        """Retry decorator should raise after max attempts"""
        from utils.error_handler import retry_with_backoff
        
        @retry_with_backoff(max_retries=2, backoff_factor=0.01)
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_failing_function()
    
    def test_retry_decorator_succeeds_immediately(self):
        """Retry decorator should not retry if function succeeds"""
        from utils.error_handler import retry_with_backoff
        
        call_count = {'count': 0}
        
        @retry_with_backoff(max_retries=3)
        def success_function():
            call_count['count'] += 1
            return "Success"
        
        result = success_function()
        
        assert result == "Success"
        assert call_count['count'] == 1  # Only called once


# ============================================================================
# UPDATED IMPLEMENTATION: utils/logger.py
# ============================================================================

"""
utils/logger.py
Updated to handle Windows file locking issues better
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Log file name (will be created in logs/ directory)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if needed
    os.makedirs('logs', exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Explicitly set encoding for Windows
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Create main application loggers
scraping_logger = setup_logger('scraping', 'scraping.log')
processing_logger = setup_logger('processing', 'processing.log')
model_logger = setup_logger('model', 'model.log')
main_logger = setup_logger('main', 'main.log')


# ============================================================================
# UPDATED IMPLEMENTATION: utils/error_handler.py
# ============================================================================

"""
utils/error_handler.py
No changes needed, but included for completeness
"""

import functools
import time


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    pass


class ModelError(Exception):
    """Custom exception for model errors"""
    pass


def retry_with_backoff(max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    """Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between retries
        exceptions: Tuple of exception types to catch
    
    Returns:
        Decorated function that retries on failure
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # Import here to avoid circular dependency during testing
                        try:
                            from utils.logger import scraping_logger
                            scraping_logger.error(
                                f"Function {func.__name__} failed after {max_retries} attempts: {str(e)}"
                            )
                        except:
                            pass  # Logger may not be available during testing
                        raise e
                    
                    wait_time = backoff_factor ** attempt
                    
                    try:
                        from utils.logger import scraping_logger
                        scraping_logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {wait_time}s..."
                        )
                    except:
                        pass  # Logger may not be available during testing
                    
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator
