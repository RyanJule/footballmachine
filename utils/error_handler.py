import functools
import time
from utils.logger import scraping_logger


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
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        scraping_logger.error(
                            f"Function {func.__name__} failed after {max_retries} attempts: {str(e)}"
                        )
                        raise e
                    
                    wait_time = backoff_factor ** attempt
                    scraping_logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator
