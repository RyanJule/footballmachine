import re
import pandas as pd
from bs4 import BeautifulSoup

from scraping.pfr_scraper import PFRScraper
from config.scraping import BASE_URL
from utils.logger import scraping_logger


class GameScraper(PFRScraper):
    """Scrape game and play-by-play data"""
    
    def __init__(self):
        super().__init__()
    
    def scrape_game_data(self, game_url, season, week):
        """Scrape detailed game data"""
        try:
            html = self.get_page_with_selenium(game_url)
            soup = BeautifulSoup(html, 'lxml')
            
            game_data = {
                'season': season,
                'week': week,
                'url': game_url,
                'game_id': game_url.split('/')[-1].replace('.htm', ''),
                'plays': []
            }
            
            self._extract_play_by_play(html, game_data)
            
            scraping_logger.info(f"Successfully scraped game: {game_url}")
            return game_data
            
        except Exception as e:
            scraping_logger.error(f"Failed to scrape game: {str(e)}")
            return {'plays': [], 'game_id': ''}
    
    def get_week_games(self, season, week):
        """Get all game URLs for a specific week"""
        url = f"{BASE_URL}/years/{season}/week_{week}.htm"
        
        try:
            html = self.get_page_with_selenium(url)
            games_df = self.parse_table(html, 'games')
            
            games = []
            if not games_df.empty:
                for _, row in games_df.iterrows():
                    # Look for boxscore link
                    for col in row.index:
                        cell_str = str(row[col])
                        if 'boxscores' in cell_str and '<a' in cell_str:
                            soup = BeautifulSoup(cell_str, 'html.parser')
                            link = soup.find('a')
                            if link and link.get('href'):
                                games.append(BASE_URL + link['href'])
                                break
            
            scraping_logger.info(f"Found {len(games)} games for week {week}")
            return games
            
        except Exception as e:
            scraping_logger.error(f"Failed to get week games: {str(e)}")
            return []
    
    def _extract_play_by_play(self, html, game_data):
        """Extract play-by-play data"""
        try:
            pbp_df = self.parse_table(html, 'pbp')
            
            if pbp_df.empty:
                return
            
            plays = []
            for _, row in pbp_df.iterrows():
                desc = str(row.get('Description', '') if 'Description' in row.index else row.get(7, ''))
                
                play = {
                    'quarter': int(str(row.get('Quarter', 0)).replace('Q', '')) if 'Quarter' in row.index else 0,
                    'description': desc,
                }
                
                # Parse play details
                play.update(self._parse_play_description(desc))
                
                plays.append(play)
            
            game_data['plays'] = plays
            
        except Exception as e:
            scraping_logger.warning(f"Failed to extract plays: {str(e)}")
            game_data['plays'] = []
    
    def _parse_play_description(self, description):
        """Parse play description to extract details"""
        desc = str(description).lower() if description else ""
        
        result = {
            'play_type': 'unknown',
            'yards_gained': 0,
            'touchdown': False,
            'field_goal': False,
            'interception': False,
            'fumble': False
        }
        
        # Determine play type
        if any(word in desc for word in ['pass', 'sacked', 'threw', 'completion', 'incomplete']):
            result['play_type'] = 'pass'
        elif any(word in desc for word in ['rush', 'run', 'carried', 'scramble']):
            result['play_type'] = 'run'
        elif 'punt' in desc:
            result['play_type'] = 'punt'
        elif any(word in desc for word in ['field goal', 'extra point', 'kick']):
            result['play_type'] = 'kick'
        
        # Check for scoring
        result['touchdown'] = 'touchdown' in desc
        result['field_goal'] = 'field goal' in desc and 'good' in desc
        
        # Check for turnovers
        result['interception'] = 'interception' in desc or 'intercepted' in desc
        result['fumble'] = 'fumble' in desc
        
        # Extract yardage
        yard_match = re.search(r'(\d+)\s*yard', desc)
        if yard_match:
            result['yards_gained'] = int(yard_match.group(1))
        
        return result