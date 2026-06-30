"""
Ensures the repo root (where core.py lives) is on sys.path when pytest
runs from CI or any working directory, not just when run locally from
inside the project folder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
