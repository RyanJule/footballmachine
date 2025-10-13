import pytest
import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='function')
def test_db():
    """Create a fresh test database for each test"""
    from config.database import Base, engine
    from database import models
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Drop all tables after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope='function')
def db_session(test_db):
    """Provide a database session for testing"""
    from config.database import SessionLocal
    
    session = SessionLocal()
    yield session
    session.close()


class TestDatabaseSetup:
    """Test database connection and table creation"""
    
    def test_database_imports(self):
        """Database configuration should be importable"""
        from config.database import Base, engine, SessionLocal
        assert Base is not None
        assert engine is not None
        assert SessionLocal is not None
    
    def test_create_tables_function_exists(self):
        """Should have function to create all tables"""
        from config.database import create_tables
        assert callable(create_tables)
    
    def test_create_tables_works(self, test_db):
        """Should create all tables without errors"""
        from config.database import Base
        
        # Get all table names
        table_names = Base.metadata.tables.keys()
        
        # Should have our core tables
        expected_tables = {'teams', 'players', 'seasons', 'games', 'plays'}
        assert expected_tables.issubset(table_names)


class TestTeamModel:
    """Test Team database model"""
    
    def test_team_model_exists(self):
        """Team model should be defined"""
        from database.models import Team
        assert Team is not None
    
    def test_team_has_required_fields(self):
        """Team should have all required fields"""
        from database.models import Team
        
        required_fields = ['id', 'name', 'abbreviation', 'pfr_id', 'created_at']
        for field in required_fields:
            assert hasattr(Team, field)
    
    def test_create_team(self, db_session):
        """Should be able to create a team"""
        from database.models import Team
        
        team = Team(
            name='Buffalo Bills',
            abbreviation='BUF',
            pfr_id='buf'
        )
        
        db_session.add(team)
        db_session.commit()
        
        assert team.id is not None
        assert team.created_at is not None
    
    def test_team_pfr_id_unique(self, db_session):
        """Team pfr_id should be unique"""
        from database.models import Team
        from sqlalchemy.exc import IntegrityError
        
        team1 = Team(name='Bills', abbreviation='BUF', pfr_id='buf')
        db_session.add(team1)
        db_session.commit()
        
        team2 = Team(name='Bills Copy', abbreviation='BUF', pfr_id='buf')
        db_session.add(team2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestPlayerModel:
    """Test Player database model"""
    
    def test_player_model_exists(self):
        """Player model should be defined"""
        from database.models import Player
        assert Player is not None
    
    def test_player_has_required_fields(self):
        """Player should have all required fields"""
        from database.models import Player
        
        required_fields = ['id', 'name', 'pfr_id', 'position', 'combine_stats', 'college_stats']
        for field in required_fields:
            assert hasattr(Player, field)
    
    def test_create_player(self, db_session):
        """Should be able to create a player"""
        from database.models import Player
        
        player = Player(
            name='Tom Brady',
            pfr_id='BradTo00',
            position='QB',
            combine_stats={'height': 76, 'weight': 225},
            college_stats={'passing': {'yards': 10000}}
        )
        
        db_session.add(player)
        db_session.commit()
        
        assert player.id is not None
        assert player.combine_stats['height'] == 76
    
    def test_player_json_fields(self, db_session):
        """JSON fields should store complex data"""
        from database.models import Player
        
        complex_stats = {
            'passing': {'completions': 100, 'attempts': 150},
            'rushing': {'attempts': 50, 'yards': 200}
        }
        
        player = Player(
            name='Test Player',
            pfr_id='TestP00',
            position='QB',
            college_stats=complex_stats
        )
        
        db_session.add(player)
        db_session.commit()
        
        # Retrieve and verify
        retrieved = db_session.query(Player).filter_by(pfr_id='TestP00').first()
        assert retrieved.college_stats['passing']['completions'] == 100


class TestSeasonModel:
    """Test Season database model"""
    
    def test_season_model_exists(self):
        """Season model should be defined"""
        from database.models import Season
        assert Season is not None
    
    def test_create_season(self, db_session):
        """Should be able to create a season"""
        from database.models import Season
        
        season = Season(year=2024, is_complete=False)
        db_session.add(season)
        db_session.commit()
        
        assert season.id is not None
        assert season.year == 2024


class TestGameModel:
    """Test Game database model"""
    
    def test_game_model_exists(self):
        """Game model should be defined"""
        from database.models import Game
        assert Game is not None
    
    def test_create_game_with_relationships(self, db_session):
        """Should create game with team relationships"""
        from database.models import Team, Season, Game
        
        # Create prerequisites
        season = Season(year=2024)
        home_team = Team(name='Bills', abbreviation='BUF', pfr_id='buf')
        away_team = Team(name='Chiefs', abbreviation='KC', pfr_id='kan')
        
        db_session.add_all([season, home_team, away_team])
        db_session.commit()
        
        # Create game
        game = Game(
            season_id=season.id,
            week=1,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_score=24,
            away_score=21,
            pfr_game_id='202409050buf',
            is_complete=True
        )
        
        db_session.add(game)
        db_session.commit()
        
        assert game.id is not None
        assert game.home_team.name == 'Bills'
        assert game.away_team.name == 'Chiefs'


class TestPlayModel:
    """Test Play database model"""
    
    def test_play_model_exists(self):
        """Play model should be defined"""
        from database.models import Play
        assert Play is not None
    
    def test_create_play(self, db_session):
        """Should be able to create a play"""
        from database.models import Team, Season, Game, Play
        
        # Create prerequisites
        season = Season(year=2024)
        home_team = Team(name='Bills', abbreviation='BUF', pfr_id='buf')
        away_team = Team(name='Chiefs', abbreviation='KC', pfr_id='kan')
        db_session.add_all([season, home_team, away_team])
        db_session.commit()
        
        game = Game(
            season_id=season.id,
            week=1,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            pfr_game_id='202409050buf'
        )
        db_session.add(game)
        db_session.commit()
        
        # Create play
        play = Play(
            game_id=game.id,
            play_number=1,
            quarter=1,
            down=1,
            yards_to_go=10,
            yard_line=25,
            play_type='pass',
            yards_gained=5,
            play_state_tensor={'data': [1, 2, 3]}
        )
        
        db_session.add(play)
        db_session.commit()
        
        assert play.id is not None
        assert play.play_type == 'pass'


class TestDatabaseOperations:
    """Test database operations helper class"""
    
    def test_operations_class_exists(self):
        """DatabaseOperations class should exist"""
        from database.operations import DatabaseOperations
        assert DatabaseOperations is not None
    
    def test_create_or_update_team(self, db_session, test_db):
        """Should create or update teams"""
        from database.operations import DatabaseOperations
        
        with DatabaseOperations() as db_ops:
            team_data = {
                'name': 'Buffalo Bills',
                'abbreviation': 'BUF',
                'pfr_id': 'buf'
            }
            
            # Create
            team1 = db_ops.create_or_update_team(team_data)
            assert team1.id is not None
            
            # Update (same pfr_id)
            team_data['name'] = 'Buffalo Bills Updated'
            team2 = db_ops.create_or_update_team(team_data)
            
            assert team1.id == team2.id  # Same team
            assert team2.name == 'Buffalo Bills Updated'
    
    def test_create_or_get_season(self, db_session, test_db):
        """Should create season or return existing"""
        from database.operations import DatabaseOperations
        
        with DatabaseOperations() as db_ops:
            season1 = db_ops.create_or_get_season(2024)
            season2 = db_ops.create_or_get_season(2024)
            
            assert season1.id == season2.id  # Same season
    
    def test_bulk_create_plays(self, db_session, test_db):
        """Should bulk insert plays efficiently"""
        from database.operations import DatabaseOperations
        from database.models import Team, Season, Game
        
        # Setup
        with DatabaseOperations() as db_ops:
            season = db_ops.create_or_get_season(2024)
            
            home_team = db_ops.create_or_update_team({
                'name': 'Bills', 'abbreviation': 'BUF', 'pfr_id': 'buf'
            })
            away_team = db_ops.create_or_update_team({
                'name': 'Chiefs', 'abbreviation': 'KC', 'pfr_id': 'kan'
            })
            
            game = db_ops.create_or_update_game({
                'season_id': season.id,
                'week': 1,
                'home_team_id': home_team.id,
                'away_team_id': away_team.id,
                'pfr_game_id': '202409050buf'
            })
            
            # Bulk create plays
            plays_data = [
                {
                    'game_id': game.id,
                    'play_number': i,
                    'quarter': 1,
                    'down': 1,
                    'yards_to_go': 10,
                    'yard_line': 25,
                    'play_type': 'pass',
                    'yards_gained': 5
                }
                for i in range(50)  # 50 plays
            ]
            
            count = db_ops.bulk_create_plays(plays_data)
            assert count == 50