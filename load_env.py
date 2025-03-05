"""
Load environment variables from .env file.
This script is used to load environment variables from a .env file for local development.
"""
import os
from pathlib import Path

def load_dotenv(dotenv_path=None):
    """
    Load environment variables from .env file.
    
    Args:
        dotenv_path: Path to .env file. If None, looks for .env in current directory.
    
    Returns:
        bool: True if .env file was loaded, False otherwise.
    """
    if dotenv_path is None:
        dotenv_path = Path('.env')
    
    if not os.path.isfile(dotenv_path):
        print(f"Warning: .env file not found at {dotenv_path}")
        return False
    
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            #Remove quotes if present
            if value and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            
            os.environ[key] = value
    
    print(f"Loaded environment variables from {dotenv_path}")
    return True

if __name__ == "__main__":
    #Load environment variables from .env file
    load_dotenv()
    
    #Print loaded environment variables (for debugging)
    print("\nLoaded environment variables:")
    for key in ['PYTHONANYWHERE', 'VERCEL_ENV', 'FLASK_APP', 'FLASK_ENV']:
        if key in os.environ:
            print(f"{key}={os.environ[key]}") 