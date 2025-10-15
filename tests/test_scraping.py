import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPFRScraperSetup:
    """Test PFR scraper initialization"""
    
    def test_pfr_scraper_imports(self):
        """PFRScraper class should be importable"""
        from scraping.pfr_scraper import PFRScraper
        assert PFRScraper is not None
    
    def test_pfr_scraper_initialization(self):
        """Should initialize without errors"""
        from scraping.pfr_scraper import PFRScraper
        
        # Mock Selenium to avoid needing Chrome driver
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PFRScraper()
            assert scraper is not None
    
    def test_pfr_scraper_has_required_methods(self):
        """PFRScraper should have all required methods"""
        from scraping.pfr_scraper import PFRScraper
        
        required_methods = [
            'get_page_with_selenium',
            'get_page_with_requests',
            'parse_table',
            'close'
        ]
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PFRScraper()
            
            for method in required_methods:
                assert hasattr(scraper, method)
                assert callable(getattr(scraper, method))
    
    def test_base_url_configured(self):
        """Base URL should be set correctly"""
        from config.scraping import BASE_URL
        
        assert BASE_URL == 'https://www.pro-football-reference.com'


class TestPlayerScraper:
    """Test player data scraping"""
    
    def test_player_scraper_imports(self):
        """PlayerScraper should be importable"""
        from scraping.player_scraper import PlayerScraper
        assert PlayerScraper is not None
    
    def test_player_scraper_initialization(self):
        """Should initialize as subclass of PFRScraper"""
        from scraping.player_scraper import PlayerScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PlayerScraper()
            assert scraper is not None
    
    def test_player_scraper_has_required_methods(self):
        """PlayerScraper should have specific methods"""
        from scraping.player_scraper import PlayerScraper
        
        required_methods = [
            'get_player_links_from_team',
            'scrape_player_data',
            '_extract_combine_data',
            '_extract_college_data',
            '_extract_nfl_career_data'
        ]
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PlayerScraper()
            
            for method in required_methods:
                assert hasattr(scraper, method)
                assert callable(getattr(scraper, method))


class TestGameScraper:
    """Test game data scraping"""
    
    def test_game_scraper_imports(self):
        """GameScraper should be importable"""
        from scraping.game_scraper import GameScraper
        assert GameScraper is not None
    
    def test_game_scraper_initialization(self):
        """Should initialize as subclass of PFRScraper"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            assert scraper is not None
    
    def test_game_scraper_has_required_methods(self):
        """GameScraper should have specific methods"""
        from scraping.game_scraper import GameScraper
        
        required_methods = [
            'scrape_game_data',
            'get_week_games',
            '_extract_play_by_play',
            '_parse_play_description'
        ]
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            for method in required_methods:
                assert hasattr(scraper, method)
                assert callable(getattr(scraper, method))


class TestTableParsing:
    """Test HTML table parsing"""
    
    def test_parse_empty_html(self):
        """Should handle empty HTML gracefully"""
        from scraping.pfr_scraper import PFRScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PFRScraper()
            
            result = scraper.parse_table("<html></html>")
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_parse_table_with_data(self):
        """Should parse valid HTML table"""
        from scraping.pfr_scraper import PFRScraper
        
        html = """
        <table>
            <tr><th>Name</th><th>Position</th></tr>
            <tr><td>Tom Brady</td><td>QB</td></tr>
            <tr><td>Rob Gronkowski</td><td>TE</td></tr>
        </table>
        """
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PFRScraper()
            result = scraper.parse_table(html)
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'Name' in result.columns or 0 in result.columns


class TestPlayParsing:
    """Test play description parsing"""
    
    def test_parse_pass_play(self):
        """Should identify pass plays"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            desc = "T.Brady pass complete to R.Gronkowski for 12 yards"
            result = scraper._parse_play_description(desc)
            
            assert result['play_type'] == 'pass'
    
    def test_parse_run_play(self):
        """Should identify run plays"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            desc = "L.Henry rush for 8 yards"
            result = scraper._parse_play_description(desc)
            
            assert result['play_type'] == 'run'
    
    def test_parse_touchdown(self):
        """Should identify touchdowns"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            desc = "T.Brady pass complete to R.Gronkowski for 12 yards, touchdown"
            result = scraper._parse_play_description(desc)
            
            assert result['touchdown'] is True
    
    def test_parse_yardage(self):
        """Should extract yardage from description"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            desc = "T.Brady pass complete to R.Gronkowski for 45 yards"
            result = scraper._parse_play_description(desc)
            
            assert result['yards_gained'] == 45
    
    def test_parse_interception(self):
        """Should identify interceptions"""
        from scraping.game_scraper import GameScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = GameScraper()
            
            desc = "T.Brady pass intercepted by S.Gilmore"
            result = scraper._parse_play_description(desc)
            
            assert result['interception'] is True


class TestErrorHandling:
    """Test error handling in scraping"""
    
    def test_scraper_returns_empty_on_error(self):
        """Should return empty DataFrame on parsing error"""
        from scraping.pfr_scraper import PFRScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PFRScraper()
            
            result = scraper.parse_table(None)
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_player_scraper_returns_dict_on_error(self):
        """Should return empty dict on player scrape error"""
        from scraping.player_scraper import PlayerScraper
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            scraper = PlayerScraper()
            
            result = scraper.scrape_player_data("invalid_url", {
                'name': 'Test',
                'position': 'QB',
                'pfr_id': 'test'
            })
            
            # Should return dict with required keys even on error
            assert isinstance(result, dict)
            assert 'player_id' in result or result == {}
