import os
import shutil
from datetime import datetime
from typing import Dict, Any, List

from utils.logger import processing_logger as logger


class BackupManager:
    """Manage database backups"""
    
    def __init__(self):
        self.backup_dir = 'backups'
        self.db_path = os.getenv('DATABASE_URL', 'sqlite:///nfl_ml.db').replace('sqlite:///', '')
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info("BackupManager initialized")
    
    def backup_database(self, cloud_upload: bool = False) -> Dict[str, Any]:
        """
        Create database backup
        
        Args:
            cloud_upload: Whether to upload to cloud storage
            
        Returns:
            Backup status
        """
        try:
            # Check if database exists
            if not os.path.exists(self.db_path):
                return {
                    'status': 'error',
                    'message': 'Database file not found'
                }
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f'nfl_ml_backup_{timestamp}.db'
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to: {backup_path}")
            
            result = {
                'status': 'success',
                'backup_path': backup_path,
                'timestamp': timestamp,
                'size_mb': os.path.getsize(backup_path) / (1024 * 1024)
            }
            
            # Upload to cloud if requested
            if cloud_upload:
                cloud_result = self._upload_to_cloud(backup_path)
                result['cloud_upload'] = cloud_result
            
            return result
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _upload_to_cloud(self, backup_path: str) -> Dict[str, Any]:
        """
        Upload backup to cloud storage (AWS S3, Google Drive, etc.)
        
        Args:
            backup_path: Local backup file path
            
        Returns:
            Upload status
        """
        try:
            # Example using AWS S3 (requires boto3 and credentials)
            # Uncomment and configure when ready to use
            
            # import boto3
            # s3_client = boto3.client('s3')
            # bucket_name = os.getenv('S3_BACKUP_BUCKET', 'nfl-ml-backups')
            # 
            # s3_client.upload_file(
            #     backup_path,
            #     bucket_name,
            #     os.path.basename(backup_path)
            # )
            # 
            # logger.info(f"Backup uploaded to S3: {bucket_name}")
            # return {
            #     'status': 'success',
            #     'bucket': bucket_name,
            #     'key': os.path.basename(backup_path)
            # }
            
            # For now, just log intention
            logger.info("Cloud upload configured but not active")
            return {
                'status': 'skipped',
                'message': 'Configure cloud credentials to enable'
            }
            
        except Exception as e:
            logger.error(f"Cloud upload failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        try:
            backups = []
            
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('.db'):
                    filepath = os.path.join(self.backup_dir, filename)
                    backups.append({
                        'filename': filename,
                        'path': filepath,
                        'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                        'created': datetime.fromtimestamp(os.path.getctime(filepath))
                    })
            
            # Sort by creation time, newest first
            backups.sort(key=lambda x: x['created'], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {str(e)}")
            return []
    
    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore database from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Restore status
        """
        try:
            if not os.path.exists(backup_path):
                return {
                    'status': 'error',
                    'message': 'Backup file not found'
                }
            
            # Create backup of current database first
            current_backup = self.backup_database()
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from: {backup_path}")
            
            return {
                'status': 'success',
                'message': 'Database restored successfully',
                'previous_backup': current_backup.get('backup_path')
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }