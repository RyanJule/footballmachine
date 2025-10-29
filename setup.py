import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple


class NFLMLSetup:
    """Setup orchestrator for NFL ML prediction system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / 'src'
        self.tests_dir = self.project_root / 'tests'
        self.config_dir = self.project_root / 'config'
        self.logs_dir = self.project_root / 'logs'
        self.backups_dir = self.project_root / 'backups'
        
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'end': '\033[0m',
            'bold': '\033[1m'
        }
    
    def print_header(self, text: str):
        """Print colored header"""
        print(f"\n{self.colors['bold']}{self.colors['blue']}")
        print("=" * 70)
        print(f"  {text}")
        print("=" * 70)
        print(self.colors['end'])
    
    def print_step(self, text: str):
        """Print step text"""
        print(f"{self.colors['cyan']}>>> {text}{self.colors['end']}")
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{self.colors['green']}✓ {text}{self.colors['end']}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{self.colors['yellow']}⚠ {text}{self.colors['end']}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{self.colors['red']}✗ {text}{self.colors['end']}")
    
    def run_command(self, cmd: list, description: str = "") -> Tuple[bool, str]:
        """Run a shell command safely"""
        try:
            if description:
                self.print_step(description)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.print_success(description or " ".join(cmd))
                return True, result.stdout
            else:
                self.print_error(f"{description or ' '.join(cmd)}: {result.stderr}")
                return False, result.stderr
                
        except Exception as e:
            self.print_error(f"Command failed: {str(e)}")
            return False, str(e)
    
    def check_python_version(self) -> bool:
        """Check if Python version is >= 3.8"""
        self.print_step("Checking Python version")
        
        if sys.version_info < (3, 8):
            self.print_error(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
            return False
        
        self.print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def create_directories(self) -> bool:
        """Create required directories"""
        self.print_step("Creating project directories")
        
        dirs = [
            self.src_dir / 'app' / 'templates',
            self.src_dir / 'automation',
            self.src_dir / 'config',
            self.src_dir / 'database',
            self.src_dir / 'data_processing',
            self.src_dir / 'scraping',
            self.src_dir / 'models',
            self.src_dir / 'utils',
            self.tests_dir,
            self.logs_dir,
            self.backups_dir
        ]
        
        for dir_path in dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_success(f"Created {dir_path.relative_to(self.project_root)}")
            except Exception as e:
                self.print_error(f"Failed to create {dir_path}: {str(e)}")
                return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.print_header("Installing Dependencies")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.print_error("requirements.txt not found")
            return False
        
        success, output = self.run_command(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        
        if not success:
            self.print_warning("Failed to upgrade pip, continuing anyway...")
        
        success, output = self.run_command(
            [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
            "Installing requirements"
        )
        
        return success
    
    def create_env_file(self) -> bool:
        """Create .env file if it doesn't exist"""
        self.print_step("Setting up environment configuration")
        
        env_file = self.project_root / '.env'
        
        if env_file.exists():
            self.print_warning(".env file already exists, skipping")
            return True
        
        env_content = """# NFL ML Prediction System Configuration

# Database
DATABASE_URL=sqlite:///nfl_ml.db

# Environment
ENV=development
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Web Scraping
SCRAPER_HEADLESS=True
SCRAPER_MAX_RETRIES=3
SCRAPER_BACKOFF_FACTOR=2
MIN_REQUEST_DELAY=1.0
MAX_REQUEST_DELAY=3.0

# Cloud Backup (Optional)
# S3_BACKUP_BUCKET=your-bucket-name
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your-key
# AWS_SECRET_ACCESS_KEY=your-secret

# Model Configuration
MODEL_BATCH_SIZE=32
MODEL_EPOCHS=100
MODEL_VALIDATION_SPLIT=0.2
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            self.print_success(".env file created")
            return True
        except Exception as e:
            self.print_error(f"Failed to create .env file: {str(e)}")
            return False
    
    def create_init_files(self) -> bool:
        """Create __init__.py files in package directories"""
        self.print_step("Creating package __init__ files")
        
        packages = [
            self.src_dir / 'app',
            self.src_dir / 'automation',
            self.src_dir / 'config',
            self.src_dir / 'database',
            self.src_dir / 'data_processing',
            self.src_dir / 'scraping',
            self.src_dir / 'models',
            self.src_dir / 'utils',
            self.src_dir,
        ]
        
        for pkg in packages:
            init_file = pkg / '__init__.py'
            if not init_file.exists():
                try:
                    init_file.touch()
                    self.print_success(f"Created {init_file.relative_to(self.project_root)}")
                except Exception as e:
                    self.print_error(f"Failed to create {init_file}: {str(e)}")
                    return False
        
        return True
    
    def initialize_database(self) -> bool:
        """Initialize database tables"""
        self.print_step("Initializing database")
        
        try:
            # Add src to path
            sys.path.insert(0, str(self.src_dir))
            
            from config.database import create_tables
            create_tables()
            
            self.print_success("Database tables created")
            return True
            
        except Exception as e:
            self.print_error(f"Database initialization failed: {str(e)}")
            self.print_warning("You can initialize the database later via the web dashboard")
            return True  # Don't fail setup for this
    
    def run_tests(self, quick: bool = False) -> bool:
        """Run test suite"""
        self.print_header("Running Tests")
        
        cmd = [sys.executable, '-m', 'pytest', str(self.tests_dir), '-v']
        
        if quick:
            cmd.append('-x')  # Stop on first failure
        
        success, output = self.run_command(cmd, "Running test suite")
        
        if success:
            self.print_success("All tests passed!")
        else:
            self.print_warning("Some tests failed. Review output above.")
        
        return success
    
    def print_next_steps(self):
        """Print next steps for user"""
        self.print_header("Setup Complete!")
        
        print("""
Next Steps:

1. Start the Web Dashboard:
   
   python run.py
   
   Then open: http://localhost:5000

2. Or initialize and run via Python:
   
   from app.main import NFLPredictionApp
   app = NFLPredictionApp()
   app.initialize_system()

3. Configure Optional Cloud Backup:
   
   Edit .env file and add:
   - S3_BACKUP_BUCKET
   - AWS credentials
   
   Then enable in web dashboard

4. View Documentation:
   
   See README.md for complete guide
   See ARCHITECTURE.md for system design
   See API_DOCS.md for endpoints

5. Start Automation:
   
   Once initialized, use web dashboard to:
   - Start automation jobs
   - Trigger data collection
   - Train models
   - View predictions

Documentation Files:
  - README.md - Overview & getting started
  - ARCHITECTURE.md - System design & components
  - API_DOCS.md - Web dashboard API endpoints
  - TRAINING_GUIDE.md - Training pipeline explained
  - DEPLOYMENT.md - Production deployment

For more information, visit the documentation files.
        """)
    
    def run_full_setup(self, run_tests: bool = True):
        """Run complete setup process"""
        self.print_header("NFL ML Prediction System - Setup")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Create directories", self.create_directories),
            ("Create __init__ files", self.create_init_files),
            ("Create environment file", self.create_env_file),
            ("Install dependencies", self.install_dependencies),
            ("Initialize database", self.initialize_database),
        ]
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    self.print_error(f"Setup failed at: {step_name}")
                    return False
            except Exception as e:
                self.print_error(f"Error during {step_name}: {str(e)}")
                return False
        
        if run_tests:
            if not self.run_tests(quick=True):
                self.print_warning("Some tests failed but setup is complete")
        
        self.print_next_steps()
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Setup NFL ML Prediction System'
    )
    parser.add_argument(
        '--no-tests',
        action='store_true',
        help='Skip running tests'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick setup without tests'
    )
    
    args = parser.parse_args()
    
    setup = NFLMLSetup()
    
    if args.quick:
        # Quick setup without tests
        steps = [
            ("Python version check", setup.check_python_version),
            ("Create directories", setup.create_directories),
            ("Create __init__ files", setup.create_init_files),
            ("Create environment file", setup.create_env_file),
            ("Install dependencies", setup.install_dependencies),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                setup.print_error(f"Setup failed at: {step_name}")
                return 1
        
        setup.print_next_steps()
    else:
        # Full setup
        if not setup.run_full_setup(run_tests=not args.no_tests):
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())