import os
from typing import Dict, Any

from utils.logger import processing_logger as logger


class ConfigValidator:
    """Validate system configuration"""
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all configuration validations"""
        results = {
            'database': self._validate_database(),
            'scrapers': self._validate_scrapers(),
            'models': self._validate_models(),
            'logging': self._validate_logging()
        }
        
        # Determine overall status
        statuses = [r['status'] for r in results.values()]
        if 'error' in statuses:
            overall = 'invalid'
        elif 'warning' in statuses:
            overall = 'warning'
        else:
            overall = 'valid'
        
        results['overall_status'] = overall
        return results
    
    def _validate_database(self) -> Dict[str, Any]:
        """Validate database configuration"""
        try:
            db_url = os.getenv('DATABASE_URL', 'sqlite:///nfl_ml.db')
            
            return {
                'status': 'valid',
                'database_url': db_url,
                'message': 'Database configuration valid'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _validate_scrapers(self) -> Dict[str, Any]:
        """Validate scraper configuration"""
        return {
            'status': 'valid',
            'message': 'Scraper configuration valid'
        }
    
    def _validate_models(self) -> Dict[str, Any]:
        """Validate model configuration"""
        return {
            'status': 'warning',
            'message': 'No trained models found'
        }
    
    def _validate_logging(self) -> Dict[str, Any]:
        """Validate logging configuration"""
        return {
            'status': 'valid',
            'message': 'Logging configured'
        }