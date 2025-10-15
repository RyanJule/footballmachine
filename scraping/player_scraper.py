import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from scraping.pfr_scraper import PFRScraper
from config.scraping import BASE_URL
from utils.logger import scraping_logger
from utils.error_handler import retry_with_backoff, ScrapingError


class PlayerScraper(PFRScraper):
    """Scrape individual player data"""
    
    def __init__(self):
        super().__init__()
    
    def get_player_links_from_team(self, team_url, season):
        """Extract player links from team roster"""
        roster_url = f"{team_url}/{season}_roster.htm"
        
        try:
            html = self.get_page_with_selenium(roster_url)
            soup = BeautifulSoup(html, 'lxml')
            
            player_links = {}
            
            roster_table = soup.find('table', {'id': 'roster'})
            if not roster_table:
                scraping_logger.warning(f"Roster table not found for {roster_url}")
                return player_links
            
            for row in roster_table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all('td')
                if not cells:
                    continue
                
                player_link = row.find('a')
                if player_link and '/players/' in player_link.get('href', ''):
                    player_name = player_link.text.strip()
                    player_url = BASE_URL + player_link['href']
                    player_id = player_link['href'].split('/')[-1].replace('.htm', '')
                    position = cells[1].text.strip() if len(cells) > 1 else 'Unknown'
                    
                    player_links[player_id] = {
                        'name': player_name,
                        'url': player_url,
                        'position': position,
                        'pfr_id': player_id
                    }
            
            scraping_logger.info(f"Found {len(player_links)} players for {roster_url}")
            return player_links
            
        except Exception as e:
            scraping_logger.error(f"Failed to get player links: {str(e)}")
            return {}
    
    @retry_with_backoff(max_retries=3, backoff_factor=2)
    def scrape_player_data(self, player_url, player_info):
        """Scrape comprehensive player data"""
        try:
            html = self.get_page_with_selenium(player_url)
            soup = BeautifulSoup(html, 'lxml')
            
            player_data = {
                'player_id': player_info['pfr_id'],
                'name': player_info['name'],
                'position': player_info['position'],
                'pfr_url': player_url
            }
            
            # Extract various data sections
            self._extract_combine_data(html, player_data)
            self._extract_college_data(html, player_data)
            self._extract_nfl_career_data(html, player_data)
            
            scraping_logger.info(f"Successfully scraped: {player_info['name']}")
            return player_data
            
        except Exception as e:
            scraping_logger.error(f"Failed to scrape player: {str(e)}")
            return {
                'player_id': player_info['pfr_id'],
                'name': player_info['name'],
                'position': player_info['position'],
                'combine_stats': {},
                'college_stats': {},
                'nfl_career_stats': {}
            }
    
    def _extract_combine_data(self, html, player_data):
        """Extract combine stats"""
        try:
            combine_df = self.parse_table(html, 'combine')
            
            combine_stats = {}
            if not combine_df.empty:
                for col in combine_df.columns:
                    col_lower = str(col).lower()
                    if 'height' in col_lower:
                        combine_stats['height'] = self._safe_float(combine_df[col].iloc[0])
                    elif 'weight' in col_lower:
                        combine_stats['weight'] = self._safe_float(combine_df[col].iloc[0])
                    elif '40' in col_lower or 'forty' in col_lower:
                        combine_stats['forty_yard'] = self._safe_float(combine_df[col].iloc[0])
                    elif 'bench' in col_lower:
                        combine_stats['bench'] = self._safe_float(combine_df[col].iloc[0])
                    elif 'broad' in col_lower or 'jump' in col_lower:
                        combine_stats['broad_jump'] = self._safe_float(combine_df[col].iloc[0])
                    elif 'shuttle' in col_lower:
                        combine_stats['shuttle'] = self._safe_float(combine_df[col].iloc[0])
                    elif '3cone' in col_lower or 'cone' in col_lower:
                        combine_stats['three_cone'] = self._safe_float(combine_df[col].iloc[0])
                    elif 'vertical' in col_lower:
                        combine_stats['vertical'] = self._safe_float(combine_df[col].iloc[0])
            
            player_data['combine_stats'] = combine_stats
            
        except Exception as e:
            scraping_logger.warning(f"Failed to extract combine data: {str(e)}")
            player_data['combine_stats'] = {}
    
    def _extract_college_data(self, html, player_data):
        """Extract college career stats"""
        try:
            college_df = self.parse_table(html, 'college_stats')
            
            college_stats = {
                'passing': {},
                'rushing': {},
                'receiving': {},
                'defense': {},
                'kicking': {}
            }
            
            if not college_df.empty:
                # Parse totals from dataframe
                pass
            
            player_data['college_stats'] = college_stats
            
        except Exception as e:
            scraping_logger.warning(f"Failed to extract college data: {str(e)}")
            player_data['college_stats'] = {}
    
    def _extract_nfl_career_data(self, html, player_data):
        """Extract NFL career stats"""
        try:
            nfl_stats = {}
            
            player_data['nfl_career_stats'] = nfl_stats
            
        except Exception as e:
            scraping_logger.warning(f"Failed to extract NFL data: {str(e)}")
            player_data['nfl_career_stats'] = {}
    
    @staticmethod
    def _safe_float(value, default=0.0):
        """Safely convert to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default