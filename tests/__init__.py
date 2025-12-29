"""
Project JANUS Visualization Test Suite
=======================================

Comprehensive test suite for all 13 JANUS visualizations.

Test Categories:
- Syntax and import tests
- Data generation tests
- Visualization output tests
- Performance tests
- Accessibility tests (color-blind safe palettes)
- Reproducibility tests (fixed seeds)

Usage:
    pytest                          # Run all tests
    pytest -v                       # Verbose output
    pytest -m "not slow"           # Skip slow tests
    pytest tests/test_visual_1.py  # Run specific test file
    pytest --cov                    # With coverage report

Author: Project JANUS Team
License: MIT
"""

__version__ = "1.0.0"
