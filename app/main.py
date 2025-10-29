from typing import Dict, Any
import os

from config.database import create_tables
from automation.orchestrator import AutomationOrchestrator
from app.config_validator import ConfigValidator
from utils.logger import processing_logger as logger


class NFLPredictionApp:
    """Main NFL Prediction Application"""
    
    def __init__(self):
        """Initialize application components"""
        self.db_initialized = False
        self.config = self._load_config()
        self.orchestrator = AutomationOrchestrator()
        self.validator = ConfigValidator()
        
        logger.info("NFLPredictionApp initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        return {
            'app_name': 'NFL ML Prediction System',
            'version': '1.0.0',
            'environment': os.getenv('ENV', 'development')
        }
    
    def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize entire system (first-time setup)
        
        Returns:
            Status dictionary
        """
        try:
            logger.info("Initializing NFL Prediction System...")
            
            # 1. Validate configuration
            validation = self.validator.validate_all()
            if validation['overall_status'] == 'invalid':
                return {
                    'status': 'error',
                    'message': 'Configuration validation failed',
                    'details': validation
                }
            
            # 2. Create database tables
            logger.info("Creating database tables...")
            create_tables()
            self.db_initialized = True
            
            # 3. Initialize automation system
            logger.info("Setting up automation jobs...")
            self.orchestrator.setup_all_jobs(current_season=2024)
            
            logger.info("System initialization complete!")
            return {
                'status': 'success',
                'message': 'System initialized successfully',
                'db_initialized': True
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start_automation(self):
        """Start automated data collection and updates"""
        self.orchestrator.start()
        logger.info("Automation started")
    
    def stop_automation(self):
        """Stop automation system"""
        self.orchestrator.stop()
        logger.info("Automation stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'db_initialized': self.db_initialized,
            'automation_running': self.orchestrator.job_manager.is_running,
            'active_jobs': len(self.orchestrator.job_manager.get_all_jobs())
        }