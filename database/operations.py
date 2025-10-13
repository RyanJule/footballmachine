from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime

from config.database import SessionLocal
from database.models import Team, Player, Season, Game, Play
from utils.logger import processing_logger


class DatabaseOperations:
    """Context manager for database operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def create_or_update_team(self, team_data: Dict) -> Team:
        """Create or update team record"""
        try:
            team = self.db.query(Team).filter(Team.pfr_id == team_data['pfr_id']).first()
            
            if team:
                # Update existing
                for key, value in team_data.items():
                    if hasattr(team, key):
                        setattr(team, key, value)
            else:
                # Create new
                team = Team(**team_data)
                self.db.add(team)
            
            self.db.commit()
            self.db.refresh(team)
            return team
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to create/update team: {str(e)}")
            raise
    
    def create_or_get_season(self, year: int) -> Season:
        """Create or get season record"""
        try:
            season = self.db.query(Season).filter(Season.year == year).first()
            
            if not season:
                season = Season(year=year)
                self.db.add(season)
                self.db.commit()
                self.db.refresh(season)
            
            return season
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to create/get season: {str(e)}")
            raise
    
    def create_or_update_game(self, game_data: Dict) -> Game:
        """Create or update game record"""
        try:
            game = self.db.query(Game).filter(Game.pfr_game_id == game_data['pfr_game_id']).first()
            
            if game:
                for key, value in game_data.items():
                    if hasattr(game, key):
                        setattr(game, key, value)
            else:
                game = Game(**game_data)
                self.db.add(game)
            
            self.db.commit()
            self.db.refresh(game)
            return game
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to create/update game: {str(e)}")
            raise
    
    def bulk_create_plays(self, plays_data: List[Dict]) -> int:
        """Bulk create play records"""
        try:
            plays = [Play(**play_data) for play_data in plays_data]
            self.db.bulk_save_objects(plays)
            self.db.commit()
            
            processing_logger.info(f"Created {len(plays)} play records")
            return len(plays)
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to bulk create plays: {str(e)}")
            raise