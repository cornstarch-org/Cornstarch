"""Pytest configuration for importing the local Cornstarch package."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
