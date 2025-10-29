import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.web_dashboard import create_app


def main():
    """Launch the Flask web dashboard"""
    
    # Create Flask app
    app = create_app()
    
    # Get configuration
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print("\n" + "=" * 70)
    print("  NFL ML PREDICTION SYSTEM - WEB DASHBOARD")
    print("=" * 70)
    print(f"\n  Starting server at http://{host}:{port}")
    print(f"  Environment: {os.getenv('ENV', 'development').upper()}")
    print(f"  Debug Mode: {'ON' if debug else 'OFF'}")
    print(f"\n  Press CTRL+C to stop\n")
    print("=" * 70 + "\n")
    
    # Run the Flask app
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug
        )
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()