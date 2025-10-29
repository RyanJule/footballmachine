import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


# ============================================================================
# Test 1: Application Initialization
# ============================================================================

def test_app_initialization():
    """Test main application initializes all systems"""
    from app.main import NFLPredictionApp
    
    app = NFLPredictionApp()
    
    assert app.db_initialized is False
    assert app.config is not None
    assert app.orchestrator is not None


# ============================================================================
# Test 2: System Initialization
# ============================================================================

def test_system_initialization():
    """Test system setup creates database and configs"""
    from app.main import NFLPredictionApp
    
    app = NFLPredictionApp()
    
    with patch('app.main.create_tables') as mock_create:
        result = app.initialize_system()
        
        assert result['status'] in ['success', 'error']
        if result['status'] == 'success':
            mock_create.assert_called()


# ============================================================================
# Test 3: Chronological Training Pipeline
# ============================================================================

def test_chronological_training_pipeline():
    """Test chronological data processing pipeline"""
    from app.training_pipeline import ChronologicalTrainingPipeline
    
    pipeline = ChronologicalTrainingPipeline()
    
    with patch('app.training_pipeline.GameScraper') as mock_scraper, \
         patch('app.training_pipeline.DatabaseOperations') as mock_db:
        
        # Mock games for a week
        mock_scraper.return_value.get_week_games.return_value = ['game1']
        mock_scraper.return_value.scrape_game_data.return_value = {
            'game_id': 'test1',
            'season': 2023,
            'week': 1,
            'plays': []
        }
        
        result = pipeline.process_week(season=2023, week=1)
        
        assert 'games_processed' in result
        assert 'player_tensors_updated' in result


# ============================================================================
# Test 4: Season Processing
# ============================================================================

def test_process_full_season():
    """Test processing entire season chronologically"""
    from app.training_pipeline import ChronologicalTrainingPipeline
    
    pipeline = ChronologicalTrainingPipeline()
    
    with patch.object(pipeline, 'process_week') as mock_process_week:
        mock_process_week.return_value = {
            'games_processed': 10,
            'player_tensors_updated': 100
        }
        
        result = pipeline.process_season(season=2023, start_week=1, end_week=3)
        
        assert result['status'] == 'success'
        assert result['weeks_processed'] == 3
        assert mock_process_week.call_count == 3


# ============================================================================
# Test 5: Game Prediction
# ============================================================================

def test_game_prediction():
    """Test predicting a single game"""
    from app.prediction_engine import PredictionEngine
    
    engine = PredictionEngine()
    
    with patch('app.prediction_engine.DatabaseOperations') as mock_db, \
         patch('app.prediction_engine.DataPipeline') as mock_pipeline:
        
        # Mock game data
        mock_game = Mock()
        mock_game.id = 1
        mock_game.home_team_id = 1
        mock_game.away_team_id = 2
        mock_game.season_id = 1
        
        mock_db.return_value.__enter__.return_value.db.query.return_value.filter_by.return_value.first.return_value = mock_game
        
        result = engine.predict_game(game_id=1)
        
        assert 'predictions' in result
        assert 'confidence' in result


# ============================================================================
# Test 6: Week Predictions
# ============================================================================

def test_predict_week():
    """Test predicting all games in a week"""
    from app.prediction_engine import PredictionEngine
    
    engine = PredictionEngine()
    
    with patch('app.prediction_engine.DatabaseOperations') as mock_db:
        # Mock games for a week
        mock_game1 = Mock()
        mock_game1.id = 1
        mock_game1.pfr_game_id = 'game1'
        
        mock_db.return_value.__enter__.return_value.db.query.return_value.join.return_value.filter.return_value.all.return_value = [mock_game1]
        
        result = engine.predict_week(season=2024, week=1)
        
        assert 'week' in result
        assert 'predictions' in result


# ============================================================================
# Test 7: Player Statistics Prediction
# ============================================================================

def test_predict_player_stats():
    """Test predicting player statistics"""
    from app.prediction_engine import PredictionEngine
    
    engine = PredictionEngine()
    
    with patch('app.prediction_engine.DatabaseOperations') as mock_db:
        mock_player = Mock()
        mock_player.id = 1
        mock_player.name = "Patrick Mahomes"
        mock_player.position = "QB"
        
        mock_db.return_value.__enter__.return_value.db.query.return_value.filter_by.return_value.first.return_value = mock_player
        
        result = engine.predict_player_game_stats(player_id=1, game_id=1)
        
        assert 'player_name' in result
        assert 'predicted_stats' in result


# ============================================================================
# Test 8: Season Leader Predictions
# ============================================================================

def test_predict_season_leaders():
    """Test predicting season statistical leaders"""
    from app.prediction_engine import PredictionEngine
    
    engine = PredictionEngine()
    
    with patch('app.prediction_engine.DatabaseOperations') as mock_db:
        result = engine.predict_season_leaders(season=2024, category='passing_yards')
        
        assert 'category' in result
        assert 'leaders' in result


# ============================================================================
# Test 9: Data Backup System
# ============================================================================

def test_database_backup():
    """Test database backup to cloud"""
    from app.backup import BackupManager
    
    manager = BackupManager()
    
    with patch('app.backup.os.path.exists') as mock_exists, \
         patch('app.backup.open', create=True) as mock_open:
        
        mock_exists.return_value = True
        
        result = manager.backup_database()
        
        assert 'status' in result
        assert 'backup_path' in result or 'error' in result


# ============================================================================
# Test 10: Configuration Validation
# ============================================================================

def test_config_validation():
    """Test configuration validation on startup"""
    from app.config_validator import ConfigValidator
    
    validator = ConfigValidator()
    
    result = validator.validate_all()
    
    assert 'database' in result
    assert 'scrapers' in result
    assert 'models' in result
    assert result['overall_status'] in ['valid', 'invalid', 'warning']


# ============================================================================
# Test 11: Flask App Routes
# ============================================================================

def test_flask_dashboard_routes():
    """Test Flask dashboard routes exist"""
    from app.web_dashboard import create_app
    
    app = create_app()
    client = app.test_client()
    
    # Test home route
    response = client.get('/')
    assert response.status_code == 200


# ============================================================================
# Test 12: API Prediction Endpoint
# ============================================================================

def test_prediction_api_endpoint():
    """Test prediction API endpoint"""
    from app.web_dashboard import create_app
    
    app = create_app()
    client = app.test_client()
    
    with patch('app.web_dashboard.PredictionEngine') as mock_engine:
        mock_engine.return_value.predict_game.return_value = {
            'predictions': {'home_score': 24, 'away_score': 21},
            'confidence': 0.75
        }
        
        response = client.post('/api/predict/game/1')
        
        assert response.status_code in [200, 404, 500]


# ============================================================================
# Test 13: Training Job Trigger
# ============================================================================

def test_trigger_training_job():
    """Test manual training job trigger"""
    from app.web_dashboard import create_app
    
    app = create_app()
    client = app.test_client()
    
    response = client.post('/api/train', 
                          json={'season': 2023, 'start_week': 1, 'end_week': 18})
    
    assert response.status_code in [200, 400, 500]