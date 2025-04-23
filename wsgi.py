import sys
import os

#Add your project directory to the Python path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

#Set PythonAnywhere environment variable
os.environ['PYTHONANYWHERE'] = 'true'

#Import the Flask app
from app import app as application 