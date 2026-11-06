import sys
import os

# Add parent directory to sys.path so app.py and modules can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
