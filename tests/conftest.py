"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Add the project root directory to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
