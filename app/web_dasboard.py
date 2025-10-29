from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from typing import Dict, Any

from app.main import NFLPredictionApp
from app.prediction_engine import PredictionEngine
from app.training_pipeline import ChronologicalTrainingPipeline
from app.backup import BackupManager
from automation.monitoring import SystemHealthCheck
from utils.logger import processing_logger as logger


def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Initialize components
    nfl_app = NFLPredictionApp()
    prediction_engine = PredictionEngine()
    training_pipeline = ChronologicalTrainingPipeline()
    backup_manager = BackupManager()
    health_check = SystemHealthCheck()
    
    # ========================================================================
    # WEB ROUTES
    # ========================================================================
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html')
    
    @app.route('/predictions')
    def predictions_page():
        """Predictions view page"""
        return render_template('predictions.html')
    
    @app.route('/training')
    def training_page():
        """Training management page"""
        return render_template('training.html')
    
    @app.route('/monitoring')
    def monitoring_page():
        """System monitoring page"""
        return render_template('monitoring.html')
    
    # ========================================================================
    # API ROUTES
    # ========================================================================
    
    @app.route('/api/status')
    def api_status():
        """Get system status"""
        status = nfl_app.get_system_status()
        health = health_check.run_full_check()
        
        return jsonify({
            'system': status,
            'health': health
        })
    
    @app.route('/api/initialize', methods=['POST'])
    def api_initialize():
        """Initialize the system"""
        result = nfl_app.initialize_system()
        return jsonify(result)
    
    @app.route('/api/predict/game/<int:game_id>', methods=['POST'])
    def api_predict_game(game_id: int):
        """Predict a specific game"""
        result = prediction_engine.predict_game(game_id)
        return jsonify(result)
    
    @app.route('/api/predict/week', methods=['POST'])
    def api_predict_week():
        """Predict all games in a week"""
        data = request.json
        season = data.get('season')
        week = data.get('week')
        
        if not season or not week:
            return jsonify({'error': 'Missing season or week'}), 400
        
        result = prediction_engine.predict_week(season, week)
        return jsonify(result)
    
    @app.route('/api/predict/player', methods=['POST'])
    def api_predict_player():
        """Predict player statistics"""
        data = request.json
        player_id = data.get('player_id')
        game_id = data.get('game_id')
        
        if not player_id or not game_id:
            return jsonify({'error': 'Missing player_id or game_id'}), 400
        
        result = prediction_engine.predict_player_game_stats(player_id, game_id)
        return jsonify(result)
    
    @app.route('/api/predict/season/leaders', methods=['POST'])
    def api_predict_leaders():
        """Predict season statistical leaders"""
        data = request.json
        season = data.get('season')
        category = data.get('category', 'passing_yards')
        
        result = prediction_engine.predict_season_leaders(season, category)
        return jsonify(result)
    
    @app.route('/api/train', methods=['POST'])
    def api_train():
        """Trigger training pipeline"""
        data = request.json
        season = data.get('season')
        start_week = data.get('start_week', 1)
        end_week = data.get('end_week', 18)
        
        if not season:
            return jsonify({'error': 'Missing season'}), 400
        
        # Run training in background (simplified - would use task queue in production)
        result = training_pipeline.process_season(season, start_week, end_week)
        return jsonify(result)
    
    @app.route('/api/backup', methods=['POST'])
    def api_backup():
        """Create database backup"""
        data = request.json or {}
        cloud_upload = data.get('cloud_upload', False)
        
        result = backup_manager.backup_database(cloud_upload=cloud_upload)
        return jsonify(result)
    
    @app.route('/api/backups')
    def api_list_backups():
        """List all backups"""
        backups = backup_manager.list_backups()
        return jsonify({'backups': backups})
    
    @app.route('/api/automation/start', methods=['POST'])
    def api_start_automation():
        """Start automation system"""
        nfl_app.start_automation()
        return jsonify({'status': 'success', 'message': 'Automation started'})
    
    @app.route('/api/automation/stop', methods=['POST'])
    def api_stop_automation():
        """Stop automation system"""
        nfl_app.stop_automation()
        return jsonify({'status': 'success', 'message': 'Automation stopped'})
    
    return app