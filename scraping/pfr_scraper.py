import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from fake_useragent import UserAgent
from io import StringIO

from config.scraping import SELENIUM_CONFIG, BASE_URL, get_request_delay
from utils.logger import scraping_logger
from utils.error_handler import retry_with_backoff, ScrapingError


class PFRScraper:
    """Base scraper for Pro Football Reference"""
    
    def __init__(self):
        """Initialize scraper with Selenium driver"""
        self.driver = None
        self.session = requests.Session()
        self.setup_selenium()
    
    def setup_selenium(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()
            
            if SELENIUM_CONFIG['headless']:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument(f"--window-size={SELENIUM_CONFIG['window_size'][0]},{SELENIUM_CONFIG['window_size'][1]}")
            chrome_options.add_argument(f"--user-agent={UserAgent().random}")
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(SELENIUM_CONFIG['implicit_wait'])
            self.driver.set_page_load_timeout(SELENIUM_CONFIG['page_load_timeout'])
            
            # Hide webdriver
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            scraping_logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            scraping_logger.error(f"Failed to initialize Selenium: {str(e)}")
            raise ScrapingError(f"Selenium setup failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2)
    def get_page_with_selenium(self, url):
        """Get page using Selenium for JavaScript rendering"""
        try:
            self.driver.get(url)
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                scraping_logger.warning(f"Timeout waiting for page: {url}")
            
            time.sleep(get_request_delay())
            
            page_source = self.driver.page_source
            scraping_logger.info(f"Successfully scraped page: {url}")
            
            return page_source
            
        except Exception as e:
            scraping_logger.error(f"Failed to scrape page {url}: {str(e)}")
            raise ScrapingError(f"Failed to scrape {url}: {str(e)}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2)
    def get_page_with_requests(self, url):
        """Get page using requests library"""
        try:
            headers = {
                'User-Agent': UserAgent().random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            time.sleep(get_request_delay())
            
            scraping_logger.info(f"Successfully requested page: {url}")
            return response.content
            
        except Exception as e:
            scraping_logger.error(f"Failed to request page {url}: {str(e)}")
            raise ScrapingError(f"Failed to request {url}: {str(e)}")
    
    def parse_table(self, html_content, table_id=None):
        """Parse HTML table, handling commented tables"""
        try:
            if not html_content:
                return pd.DataFrame()
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Try to find uncommented table
            if table_id:
                table = soup.find('table', {'id': table_id})
            else:
                table = soup.find('table')
            
            if table:
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    return df
                except:
                    return pd.DataFrame()
            
            # Look for commented tables
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            
            for comment in comments:
                comment_str = str(comment)
                if '<table' in comment_str and (not table_id or table_id in comment_str):
                    comment_soup = BeautifulSoup(comment_str, 'lxml')
                    table = comment_soup.find('table')
                    if table:
                        try:
                            df = pd.read_html(str(table))[0]
                            return df
                        except:
                            continue
            
            scraping_logger.warning(f"Table {table_id} not found")
            return pd.DataFrame()
            
        except Exception as e:
            scraping_logger.error(f"Failed to parse table {table_id}: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
        self.session.close()
        scraping_logger.info("PFRScraper closed successfully")
