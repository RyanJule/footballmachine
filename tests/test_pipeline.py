import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='function')
def test_db():
    """Create test database"""
    from config.database import Base, engine
    from database import models
    
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope='function')
def db_ops(test_db):
    """Provide database operations"""
    from database.operations import DatabaseOperations
    
    with DatabaseOperations() as ops:
        yield ops


class TestDataPipelineSetup:
    """Test pipeline initialization"""
    
    def test_pipeline_class_imports(self):
        """DataPipeline class should be importable"""
        from data_processing.pipeline import DataPipeline
        assert DataPipeline is not None
    
    def test_pipeline_initialization(self):
        """Should initialize with all components"""
        from data_processing.pipeline import DataPipeline
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            assert pipeline.tensor_builder is not None
            assert pipeline.db_ops is not None
    
    def test_pipeline_has_required_methods(self):
        """Pipeline should have all processing methods"""
        from data_processing.pipeline import DataPipeline
        
        required_methods = [
            'process_scraped_player',
            'process_scraped_game',
            'process_team_roster',
            'build_game_tensors'
        ]
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            for method in required_methods:
                assert hasattr(pipeline, method)
                assert callable(getattr(pipeline, method))


class TestPlayerProcessing:
    """Test player data processing"""
    
    def test_process_player_to_database(self, db_ops):
        """Should save scraped player to database"""
        from data_processing.pipeline import DataPipeline
        
        scraped_player = {
            'player_id': 'TestP00',
            'name': 'Test Player',
            'position': 'QB',
            'pfr_url': 'https://pfr.com/players/T/TestP00.htm',
            'combine_stats': {'height': 76, 'weight': 225},
            'college_stats': {'passing': {'yards': 10000}},
            'nfl_career_stats': {'passing': {'yards': 50000}}
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # Process player
            db_player = pipeline.process_scraped_player(scraped_player)
            
            assert db_player is not None
            assert db_player.pfr_id == 'TestP00'
            assert db_player.name == 'Test Player'
            assert db_player.combine_stats['height'] == 76
    
    def test_process_player_creates_tensor(self, db_ops):
        """Should create 670-feature tensor for player"""
        from data_processing.pipeline import DataPipeline
        
        scraped_player = {
            'player_id': 'TestP01',
            'name': 'Test Player 2',
            'position': 'RB',
            'combine_stats': {},
            'college_stats': {},
            'nfl_career_stats': {}
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # Get tensor for player
            tensor = pipeline.tensor_builder.build_player_tensor(scraped_player)
            
            assert tensor.shape == (670,)
    
    def test_process_duplicate_player_updates(self, db_ops):
        """Should update existing player instead of creating duplicate"""
        from data_processing.pipeline import DataPipeline
        
        player_v1 = {
            'player_id': 'DupP00',
            'name': 'Duplicate Player',
            'position': 'QB',
            'combine_stats': {'height': 75},
            'college_stats': {},
            'nfl_career_stats': {}
        }
        
        player_v2 = {
            'player_id': 'DupP00',
            'name': 'Duplicate Player Updated',
            'position': 'QB',
            'combine_stats': {'height': 76},
            'college_stats': {},
            'nfl_career_stats': {}
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # First insert
            db_player1 = pipeline.process_scraped_player(player_v1)
            id1 = db_player1.id
            
            # Second insert (should update)
            db_player2 = pipeline.process_scraped_player(player_v2)
            id2 = db_player2.id
            
            # Should be same player
            assert id1 == id2
            assert db_player2.combine_stats['height'] == 76


class TestGameProcessing:
    """Test game data processing"""
    
    def test_process_game_to_database(self, db_ops):
        """Should save scraped game to database"""
        from data_processing.pipeline import DataPipeline
        from database.models import Team, Season
        
        # Create prerequisites
        season = db_ops.create_or_get_season(2024)
        home_team = db_ops.create_or_update_team({
            'name': 'Bills', 'abbreviation': 'BUF', 'pfr_id': 'buf'
        })
        away_team = db_ops.create_or_update_team({
            'name': 'Chiefs', 'abbreviation': 'KC', 'pfr_id': 'kan'
        })
        
        scraped_game = {
            'game_id': '202409050buf',
            'season': 2024,
            'week': 1,
            'home_team': 'buf',
            'away_team': 'kan',
            'home_score': 24,
            'away_score': 21,
            'plays': [
                {
                    'quarter': 1,
                    'description': 'J.Allen pass to S.Diggs for 15 yards',
                    'play_type': 'pass',
                    'yards_gained': 15
                }
            ]
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # Process game
            db_game = pipeline.process_scraped_game(scraped_game)
            
            assert db_game is not None
            assert db_game.pfr_game_id == '202409050buf'
            assert db_game.week == 1
    
    def test_process_game_creates_plays(self, db_ops):
        """Should save play-by-play data"""
        from data_processing.pipeline import DataPipeline
        from database.models import Play
        
        season = db_ops.create_or_get_season(2024)
        home_team = db_ops.create_or_update_team({
            'name': 'Bills', 'abbreviation': 'BUF', 'pfr_id': 'buf'
        })
        away_team = db_ops.create_or_update_team({
            'name': 'Chiefs', 'abbreviation': 'KC', 'pfr_id': 'kan'
        })
        
        scraped_game = {
            'game_id': '202409050buf2',
            'season': 2024,
            'week': 1,
            'home_team': 'buf',
            'away_team': 'kan',
            'home_score': 24,
            'away_score': 21,
            'plays': [
                {'quarter': 1, 'play_type': 'pass', 'yards_gained': 15},
                {'quarter': 1, 'play_type': 'run', 'yards_gained': 5},
                {'quarter': 1, 'play_type': 'pass', 'yards_gained': 20}
            ]
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            db_game = pipeline.process_scraped_game(scraped_game)
            
            # Check plays were created
            plays_count = db_ops.db.query(Play).filter_by(game_id=db_game.id).count()
            assert plays_count == 3


class TestRosterProcessing:
    """Test roster assembly and tensor building"""
    
    def test_build_roster_tensor_from_db(self, db_ops):
        """Should build roster tensor from database players"""
        from data_processing.pipeline import DataPipeline
        
        season = db_ops.create_or_get_season(2024)
        team = db_ops.create_or_update_team({
            'name': 'Bills', 'abbreviation': 'BUF', 'pfr_id': 'buf'
        })
        
        # Create some players
        for i in range(5):
            player_data = {
                'name': f'Player {i}',
                'pfr_id': f'Play{i:02d}',
                'position': 'QB' if i == 0 else 'RB',
                'combine_stats': {},
                'college_stats': {}
            }
            db_player = db_ops.create_or_update_player(player_data)
            
            # Create player season
            db_ops.create_or_update_player_season({
                'player_id': db_player.id,
                'season_id': season.id,
                'team_id': team.id,
                'games_played': 16,
                'individual_stats': {}
            })
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # Build roster tensor for team
            roster_tensor = pipeline.process_team_roster(team.id, season.id)
            
            # Should be flattened 64*670
            assert roster_tensor.shape == (64 * 670,)
    
    def test_build_game_tensor(self, db_ops):
        """Should build complete game tensor"""
        from data_processing.pipeline import DataPipeline
        
        season = db_ops.create_or_get_season(2024)
        
        # Create teams
        home_team = db_ops.create_or_update_team({
            'name': 'Bills', 'abbreviation': 'BUF', 'pfr_id': 'buf'
        })
        away_team = db_ops.create_or_update_team({
            'name': 'Chiefs', 'abbreviation': 'KC', 'pfr_id': 'kan'
        })
        
        # Create game
        game = db_ops.create_or_update_game({
            'season_id': season.id,
            'week': 1,
            'home_team_id': home_team.id,
            'away_team_id': away_team.id,
            'pfr_game_id': '202409050buf3',
            'home_score': 24,
            'away_score': 21
        })
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            # Build game tensor
            game_tensor = pipeline.build_game_tensors(game.id)
            
            # Should be home(64*670) + away(64*670) + game_info(50)
            expected_size = (64 * 670 * 2) + 50
            assert game_tensor.shape == (expected_size,)


class TestDataValidation:
    """Test data validation and cleaning"""
    
    def test_validates_player_position(self):
        """Should validate player position"""
        from data_processing.pipeline import DataPipeline
        
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P']
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            for pos in valid_positions:
                assert pipeline._validate_position(pos) is True
            
            assert pipeline._validate_position('INVALID') is False
    
    def test_cleans_player_data(self, db_ops):
        """Should clean and normalize player data"""
        from data_processing.pipeline import DataPipeline
        
        messy_player = {
            'player_id': 'Messy00',
            'name': '  Tom Brady  ',  # Extra spaces
            'position': 'qb',  # Lowercase
            'combine_stats': {'height': '76'},  # String instead of int
            'college_stats': {},
            'nfl_career_stats': {}
        }
        
        with patch('scraping.pfr_scraper.webdriver.Chrome'):
            pipeline = DataPipeline()
            
            cleaned = pipeline._clean_player_data(messy_player)
            
            assert cleaned['name'] == 'Tom Brady'  # Trimmed
            assert cleaned['position'] == 'QB'  # Uppercase
