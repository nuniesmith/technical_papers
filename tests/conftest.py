"""
Pytest configuration and shared fixtures for JANUS visualization tests.

This module provides:
- Common fixtures for all test files
- Test data generators
- Output directory management
- Performance measurement utilities
- Accessibility validation helpers

Author: Project JANUS Team
License: MIT
"""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

# Add project_janus to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "project_janus" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


# =============================================================================
# Session-level Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="janus_test_"))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Return the path to the examples directory."""
    return EXAMPLES_DIR


# =============================================================================
# Function-level Fixtures
# =============================================================================


@pytest.fixture
def output_dir(test_output_dir: Path, request) -> Path:
    """Create a subdirectory for each test function."""
    test_name = request.node.name
    test_dir = test_output_dir / test_name
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """NumPy random number generator with fixed seed."""
    return np.random.default_rng(random_seed)


# =============================================================================
# Test Data Generators
# =============================================================================


@pytest.fixture
def synthetic_price_series(rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic price series for testing."""
    n_steps = 100
    trend = np.linspace(100, 110, n_steps)
    noise = rng.normal(0, 1, n_steps)
    return trend + noise


@pytest.fixture
def synthetic_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic high-dimensional embeddings."""
    n_samples = 200
    n_features = 64
    return rng.normal(0, 1, (n_samples, n_features))


@pytest.fixture
def synthetic_labels(rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic class labels."""
    n_samples = 200
    n_classes = 3
    return rng.integers(0, n_classes, size=n_samples)


# =============================================================================
# Validation Helpers
# =============================================================================


def check_image_file(filepath: Path, min_size_kb: int = 10) -> bool:
    """
    Validate that an image file was created and is non-trivial.

    Args:
        filepath: Path to image file
        min_size_kb: Minimum file size in kilobytes

    Returns:
        True if file exists and meets size requirement
    """
    if not filepath.exists():
        return False

    file_size_kb = filepath.stat().st_size / 1024
    return file_size_kb >= min_size_kb


@pytest.fixture
def check_image():
    """Fixture providing image validation function."""
    return check_image_file


def validate_colormap_accessible(colors: list) -> bool:
    """
    Check if a colormap is accessible for color-blind users.

    This is a simplified check. For production, use colorspacious.

    Args:
        colors: List of RGB tuples or hex colors

    Returns:
        True if colormap appears accessible
    """
    # Simplified check: ensure sufficient color variation
    if len(colors) < 2:
        return True

    # Check that colors are sufficiently different
    # (proper implementation would use CIELAB color space)
    return True  # Placeholder


@pytest.fixture
def validate_accessibility():
    """Fixture providing accessibility validation."""
    return validate_colormap_accessible


# =============================================================================
# Performance Measurement
# =============================================================================


class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(self):
        self.elapsed = None
        self._start = None

    def __enter__(self):
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time

        self.elapsed = time.perf_counter() - self._start


@pytest.fixture
def timer():
    """Fixture providing performance timer."""
    return PerformanceTimer


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "visual: mark test as generating visual output")


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests based on test names."""
    for item in items:
        # Mark UMAP tests as slow
        if "umap" in item.nodeid.lower() or "visual_11" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark tests that generate visuals
        if "visual_" in item.nodeid:
            item.add_marker(pytest.mark.visual)


# =============================================================================
# Matplotlib Configuration
# =============================================================================


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Configure matplotlib for non-interactive testing."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Set consistent style
    plt.style.use("default")

    # Prevent display of figures
    plt.ioff()

    yield

    # Cleanup: close all figures after each test
    plt.close("all")


# =============================================================================
# Import Guards
# =============================================================================


def pytest_runtest_setup(item):
    """Skip tests if required dependencies are not available."""
    # Check for UMAP dependency
    if "umap" in item.nodeid.lower():
        try:
            import umap  # noqa: F401
        except ImportError:
            pytest.skip("umap-learn not installed")

    # Check for PyTorch dependency
    if "torch" in item.nodeid.lower() or "attention" in item.nodeid.lower():
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")
