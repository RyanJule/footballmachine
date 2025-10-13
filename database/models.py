from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.types import TypeDecorator
from datetime import datetime, timezone
import json as json_lib

from config.database import Base


# Custom JSON type that works with SQLite
class JSONType(TypeDecorator):
    """Platform-independent JSON type"""
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json_lib.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json_lib.loads(value)
        return None

def utc_now():
    return datetime.now(timezone.utc)

class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    abbreviation = Column(String(3), unique=True)
    pfr_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=utc_now())
    
    # Relationships
    seasons = relationship("TeamSeason", back_populates="team")
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")


class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    pfr_id = Column(String, unique=True, index=True)
    position = Column(String)
    draft_year = Column(Integer)
    draft_team = Column(String)
    draft_pick = Column(Integer)
    birth_date = Column(DateTime)
    college = Column(String)
    height = Column(Integer)
    weight = Column(Integer)
    created_at = Column(DateTime, default=utc_now())
    updated_at = Column(DateTime, default=utc_now(), onupdate=utc_now())
    
    # JSON fields for flexible data storage
    combine_stats = Column(JSONType)
    college_stats = Column(JSONType)
    
    # Relationships
    seasons = relationship("PlayerSeason", back_populates="player")


class Season(Base):
    __tablename__ = "seasons"
    
    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, unique=True, index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_complete = Column(Boolean, default=False)
    
    # Relationships
    games = relationship("Game", back_populates="season")
    player_seasons = relationship("PlayerSeason", back_populates="season")
    team_seasons = relationship("TeamSeason", back_populates="season")


class TeamSeason(Base):
    __tablename__ = "team_seasons"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    season_id = Column(Integer, ForeignKey("seasons.id"))
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    ties = Column(Integer, default=0)
    
    # JSON fields for stats
    offensive_stats = Column(JSONType)
    defensive_stats = Column(JSONType)
    
    # Relationships
    team = relationship("Team", back_populates="seasons")
    season = relationship("Season", back_populates="team_seasons")


class PlayerSeason(Base):
    __tablename__ = "player_seasons"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    season_id = Column(Integer, ForeignKey("seasons.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    games_played = Column(Integer)
    games_started = Column(Integer)
    
    # JSON fields for stats
    individual_stats = Column(JSONType)
    team_performance_stats = Column(JSONType)
    opponent_performance_stats = Column(JSONType)
    
    # Relationships
    player = relationship("Player", back_populates="seasons")
    season = relationship("Season", back_populates="player_seasons")


class Game(Base):
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True, index=True)
    season_id = Column(Integer, ForeignKey("seasons.id"))
    week = Column(Integer)
    game_date = Column(DateTime)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    home_score = Column(Integer)
    away_score = Column(Integer)
    is_complete = Column(Boolean, default=False)
    pfr_game_id = Column(String, unique=True, index=True)
    
    # JSON field for game info
    game_info = Column(JSONType)
    
    # Relationships
    season = relationship("Season", back_populates="games")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    plays = relationship("Play", back_populates="game")


class Play(Base):
    __tablename__ = "plays"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    play_number = Column(Integer)
    quarter = Column(Integer)
    time_remaining = Column(String)
    down = Column(Integer)
    yards_to_go = Column(Integer)
    yard_line = Column(Integer)
    
    # Play outcome
    play_type = Column(String)
    yards_gained = Column(Integer)
    touchdown = Column(Boolean, default=False)
    field_goal = Column(Boolean, default=False)
    safety = Column(Boolean, default=False)
    
    # JSON field for tensor data
    play_state_tensor = Column(JSONType)
    
    # Relationship
    game = relationship("Game", back_populates="plays")


class ModelWeights(Base):
    __tablename__ = "model_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    version = Column(String)
    trained_date = Column(DateTime, default=utc_now())
    training_seasons = Column(JSONType)
    hyperparameters = Column(JSONType)
    metrics = Column(JSONType)
    weights_file_path = Column(String)
    is_active = Column(Boolean, default=False)


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String)
    status = Column(String)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    results = Column(JSONType)