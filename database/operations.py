from sqlalchemy.orm import Session
from typing import Dict, List
from datetime import datetime

from config.database import SessionLocal
from database.models import Team, Player, Season, Game, Play, PlayerSeason
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

    def create_or_update_player(self, player_data: Dict) -> Player:
        """Create or update player record"""
        try:
            player = self.db.query(Player).filter(Player.pfr_id == player_data['pfr_id']).first()
            
            if player:
                # Update existing
                for key, value in player_data.items():
                    if hasattr(player, key):
                        setattr(player, key, value)
            else:
                # Create new
                player = Player(**player_data)
                self.db.add(player)
            
            self.db.commit()
            self.db.refresh(player)
            return player
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to create/update player: {str(e)}")
            raise
    
    def create_or_update_player_season(self, player_season_data: Dict) -> Player:
        """Create or update player record"""
        try:
            player_season = self.db.query(PlayerSeason).filter(PlayerSeason.player_id == player_season_data['player_id']).first()
            
            if player_season:
                # Update existing
                for key, value in player_season_data.items():
                    if hasattr(player_season, key):
                        setattr(player_season, key, value)
            else:
                # Create new
                player_season = PlayerSeason(**player_season_data)
                self.db.add(player_season)
            
            self.db.commit()
            self.db.refresh(player_season)
            return player_season
            
        except Exception as e:
            self.db.rollback()
            processing_logger.error(f"Failed to create/update player: {str(e)}")
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
    
    def get_players_by_team_season(self, team_id: int, season_id: int):
        try:
            from database.models import Player, PlayerSeason
            
            players = self.db.query(Player).join(PlayerSeason).filter(
                PlayerSeason.team_id == team_id,
                PlayerSeason.season_id == season_id
            ).all()
            
            return players
            
        except Exception as e:
            processing_logger.error(f"Failed to get players: {str(e)}")
            return []