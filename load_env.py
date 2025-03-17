"""
Load environment variables from .env file.
This script is used to load environment variables from a .env file for local development.
"""
import os
from pathlib import Path

def load_dotenv(dotenv_path=None):
    """
    Load environment variables from .env file
    
    Args:
        dotenv_path: Path to .env file (defaults to .env in current directory)
    """
    #First, try with the provided path if specified
    if dotenv_path is not None and os.path.isfile(dotenv_path):
        success = _load_env_from_file(dotenv_path)
        if success:
            print(f"Loaded environment from specified path: {dotenv_path}")
            return True
    
    #If no path specified or file not found, try local .env file first
    local_env_path = '.env'
    if os.path.isfile(local_env_path):
        success = _load_env_from_file(local_env_path)
        if success:
            print(f"Loaded environment from local path: {local_env_path}")
            return True
    
    #If local load fails, try PythonAnywhere paths
    home_dir = os.path.expanduser('~')
    possible_paths = [
        os.path.join(home_dir, 'Ficksaudio-Flask-Web', '.env'),
        os.path.join(home_dir, '.env')
    ]
    
    #Try each possible PythonAnywhere path
    for path in possible_paths:
        if os.path.isfile(path):
            success = _load_env_from_file(path)
            if success:
                #After loading, set the PYTHONANYWHERE flag since we're clearly in that environment
                if 'PYTHONANYWHERE' not in os.environ:
                    os.environ['PYTHONANYWHERE'] = 'true'
                print(f"Loaded environment from PythonAnywhere path: {path}")
                return True
    
    #No .env file found anywhere
    print("No .env file found in any location")
    return False

def _load_env_from_file(file_path):
    """
    Helper function to load variables from a specific .env file
    
    Args:
        file_path: Path to the .env file to load
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        #Load variables from the specified .env file
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                
                #Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                #Parse key-value pairs using = delimiter
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    #Remove surrounding quotes if present
                    if value and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                        
                    #Set environment variable if not already set
                    if key and key not in os.environ:
                        os.environ[key] = value
        return True
    except Exception as e:
        print(f"Error loading environment from {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    #Load environment from default locations
    load_dotenv()
    
    #Display loaded environment variables for verification
    print("\nLoaded environment variables:")
    for key in ['PYTHONANYWHERE', 'VERCEL_ENV', 'FLASK_APP', 'FLASK_ENV']:
        if key in os.environ:
            print(f"{key}={os.environ[key]}") 