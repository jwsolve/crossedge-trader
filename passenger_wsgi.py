import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from dashboard import app as application

# For debugging
print("Dashboard WSGI loaded successfully", file=sys.stderr)
