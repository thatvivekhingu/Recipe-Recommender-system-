import sys
import os

# Add project root directory to sys.path so app.py and local modules are discovered
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app