import random

# Scraping configuration
BASE_URL = 'https://www.pro-football-reference.com'

SELENIUM_CONFIG = {
    'headless': True,
    'window_size': (1920, 1080),
    'implicit_wait': 10,
    'page_load_timeout': 30
}

# Rate limiting
MIN_REQUEST_DELAY = 1.0  # seconds
MAX_REQUEST_DELAY = 3.0  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2


def get_request_delay():
    return random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)