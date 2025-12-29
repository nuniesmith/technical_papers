# Contributing to Project JANUS

Thank you for your interest in contributing to Project JANUS! This document provides guidelines and instructions for development.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/nuniesmith/technical_papers.git
cd technical_papers

# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run all checks
make check
```

## 🛠️ Development Setup

### Prerequisites

- **Python**: 3.9 or higher
- **LaTeX**: TeX Live (for PDF compilation)
- **Git**: Version control

### Installation Options

#### Option 1: Full Development Setup (Recommended)

```bash
# Install all dependencies including optional ones
make install-all
```

This installs:

- Core dependencies (numpy, matplotlib, scipy, scikit-learn)
- Optional dependencies (umap-learn, torch, seaborn, etc.)
- Development tools (pytest, ruff, black, mypy, pre-commit)

#### Option 2: Core Dependencies Only

```bash
# Install only core dependencies
make install
```

#### Option 3: Manual Installation

```bash
# Core + development
pip install -e .[dev]

# Everything
pip install -e .[all,dev]
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. You have two installation options:

#### Option A: Virtual Environment (Default)

```bash
# Activate venv first
source venv/bin/activate

# Install hooks (done automatically with make install-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Note**: You must activate the venv before committing: `source venv/bin/activate`

#### Option B: Global Installation (Recommended for Convenience)

Install pre-commit globally so it works without activating the venv:

```bash
# One-time setup: Install pipx
sudo apt install pipx
pipx ensurepath

# Install pre-commit globally
pipx install pre-commit

# Install Git hooks
pre-commit install

# Now commits work without venv activation
git commit -m "your message"
```

**Benefits**:

- ✅ No need to activate venv before committing
- ✅ Works across all your Git repositories
- ✅ Isolated installation (pipx manages its own venv)
- ✅ Simplifies daily workflow

## 🔄 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow our coding standards:

- **Python**: Follow PEP 8, use type hints where appropriate
- **Line length**: 100 characters (configured in pyproject.toml)
- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain *why*, not *what*

### 3. Format Code

```bash
# Auto-format with black and isort
make format

# Or manually:
black project_janus/examples tests
isort project_janus/examples tests
```

### 4. Run Linters

```bash
# Run all linters
make lint

# Fix auto-fixable issues
make ruff-fix
```

### 5. Run Tests

```bash
# Quick tests (excludes slow tests)
make test

# All tests
make test-all

# Specific test file
pytest tests/test_visual_5.py -v

# With coverage
make test-coverage
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new visualization feature"

# Commit message format:
# - feat: new feature
# - fix: bug fix
# - docs: documentation changes
# - test: test additions/changes
# - refactor: code refactoring
# - style: formatting changes
# - chore: maintenance tasks
```

## ✅ Code Quality

### Linting

We use **Ruff** for fast, modern linting:

```bash
# Check for issues
ruff check project_janus/examples tests

# Auto-fix issues
ruff check --fix project_janus/examples tests
```

### Formatting

We use **Black** for consistent code formatting:

```bash
# Check formatting
black --check project_janus/examples tests

# Apply formatting
black project_janus/examples tests
```

### Import Sorting

We use **isort** to organize imports:

```bash
# Check import order
isort --check-only project_janus/examples tests

# Sort imports
isort project_janus/examples tests
```

### Type Checking

We use **mypy** for type checking (optional but recommended):

```bash
mypy project_janus/examples tests
```

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_all_visuals.py      # Smoke tests for all visuals
├── test_visual_5.py         # Detailed tests for Visual 5
└── test_visual_*.py         # Tests for other visuals
```

### Writing Tests

```python
import pytest
import sys
from pathlib import Path

# Import visualization module
sys.path.insert(0, str(Path(__file__).parent.parent / "project_janus" / "examples"))
import visual_5_ltn_truth_surface as v5


class TestVisualization:
    """Test suite for Visual 5."""

    def test_basic_functionality(self, output_dir):
        """Test basic visualization generation."""
        output_file = output_dir / "test.png"

        fig = v5.visualize_truth_surface(
            operation="and",
            save_path=str(output_file),
            dpi=100,
        )

        assert fig is not None
        assert output_file.exists()
        assert output_file.stat().st_size > 10000


    @pytest.mark.slow
    def test_performance(self, output_dir, timer):
        """Test performance meets tier requirements."""
        with timer() as t:
            v5.visualize_truth_surface(
                operation="and",
                save_path=str(output_dir / "perf.png"),
                dpi=100,
            )

        assert t.elapsed < 5.0, f"Took {t.elapsed:.2f}s (expected < 5s)"
```

### Test Markers

- `@pytest.mark.slow` - Slow tests (skipped by default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.visual` - Tests that generate visualizations

### Running Tests

```bash
# Quick tests only
make test-quick

# All tests except slow
make test

# All tests including slow
make test-all

# Visual generation tests
make test-visual

# Integration tests
make test-integration

# Specific marker
pytest -v -m "not slow and not integration"
```

## 📚 Documentation

### Code Documentation

- Use **Google-style docstrings** for all public functions and classes
- Include type hints for function parameters and return values
- Document complex algorithms with inline comments

Example:

```python
def visualize_truth_surface(
    operation: str = "and",
    save_path: Optional[str] = None,
    show_gradients: bool = True,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate 3D surface plot comparing Łukasiewicz and Boolean logic.

    Implements Visual 5 specification from JANUS documentation.

    Args:
        operation: Logic operation - 'and', 'or', or 'implies'
        save_path: If provided, save figure to this path
        show_gradients: If True, overlay gradient vectors
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object

    Raises:
        ValueError: If operation is not recognized

    Example:
        >>> fig = visualize_truth_surface(
        ...     operation="and",
        ...     save_path="output.png",
        ...     dpi=300
        ... )
    """
    ...
```

### LaTeX Documentation

For changes to the technical paper (`janus.tex`):

1. Follow existing formatting conventions
2. Use `\texttt{}` for code/filenames
3. Use `tcolorbox` for important notes
4. Reference equations consistently

## 📤 Submitting Changes

### Pull Request Process

1. **Update your branch**

   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Ensure all checks pass**

   ```bash
   make ci  # Simulates CI pipeline locally
   ```

3. **Push your branch**

   ```bash
   git push origin your-branch
   ```

4. **Create Pull Request**

   - Go to GitHub and create a PR from your branch
   - Fill out the PR template
   - Link any related issues

5. **CI Checks**

   GitHub Actions will automatically:
   - Run linters (ruff, black, isort)
   - Run type checker (mypy)
   - Run tests on Python 3.9, 3.10, 3.11, 3.12
   - Build LaTeX PDFs
   - Generate coverage reports

6. **Code Review**

   - Address reviewer feedback
   - Keep commits atomic and well-described
   - Squash commits if requested

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (run `make format`)
- [ ] All linters pass (run `make lint`)
- [ ] Tests pass (run `make test-all`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No debug code or print statements
- [ ] Pre-commit hooks pass

## 🎨 Visualization Guidelines

When adding new visualizations:

1. **Follow the specification** in `visualization_specification.md`
2. **Use accessibility standards**:
   - WCAG 2.1 AA compliant
   - Color-blind safe palettes
   - Redundant encodings (color + shape/pattern)
3. **Performance tiers**:
   - Tier 1: < 100ms (real-time)
   - Tier 2: < 1s (interactive)
   - Tier 3: < 60s (batch)
   - Tier 4: Static (no time constraint)
4. **Reproducibility**:
   - Use fixed seeds for random generation
   - Document all parameters
   - Provide CLI interface
5. **Output quality**:
   - Default: 300 DPI
   - Support configurable DPI
   - Use vector formats where appropriate

## 🐛 Reporting Issues

When reporting bugs, please include:

- **Python version**: `python --version`
- **Installed packages**: `pip list`
- **Operating system**: Linux/macOS/Windows
- **Error message**: Full traceback
- **Minimal reproduction**: Smallest code example that reproduces the issue

## 💬 Getting Help

- **Documentation**: Check `project_janus/README.md` and `visualization_specification.md`
- **Examples**: See `project_janus/examples/` for reference implementations
- **Tests**: Look at `tests/` for usage examples
- **Issues**: Search existing GitHub issues

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

## 🙏 Thank You

Your contributions help make Project JANUS better for everyone. We appreciate your time and effort!

---

**Quick Reference:**

```bash
make help              # Show all available commands
make install-dev       # Setup development environment
make format            # Format code
make lint              # Run linters
make test              # Run tests
make check             # Format + lint + test
make ci                # Simulate full CI pipeline
```
