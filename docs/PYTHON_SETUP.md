# Python Testing & Linting Setup for Project JANUS

This document summarizes the complete Python development infrastructure setup for Project JANUS visualizations.

## 📋 Overview

We've implemented a modern Python development workflow with:

- ✅ **Unpinned dependencies** (flexible version ranges)
- ✅ **Modern linting** (Ruff - fast and comprehensive)
- ✅ **Code formatting** (Black + isort)
- ✅ **Type checking** (mypy)
- ✅ **Testing framework** (pytest with coverage)
- ✅ **Pre-commit hooks** (automated quality checks)
- ✅ **CI/CD integration** (GitHub Actions)
- ✅ **Make-based workflow** (convenient commands)

## 🗂️ New Project Structure

```
technical_papers/
├── pyproject.toml                 # Modern Python project configuration
├── Makefile                       # Development task automation
├── CONTRIBUTING.md                # Development workflow guide
├── .pre-commit-config.yaml       # Pre-commit hooks configuration
├── .markdownlint.json            # Markdown linting rules
│
├── project_janus/
│   ├── examples/
│   │   ├── visual_*.py           # 13 visualization scripts
│   │   ├── test_all_visuals.sh   # Comprehensive test script
│   │   ├── test_v7_v11.sh        # Focused test script
│   │   └── requirements.txt      # Simple requirements file
│   └── README.md
│
└── tests/                         # NEW: Test suite
    ├── __init__.py
    ├── conftest.py               # Pytest configuration & fixtures
    ├── test_all_visuals.py       # Smoke tests for all 13 visuals
    └── test_visual_5.py          # Detailed example test suite
```

## 🔧 Installation

### Option 1: Core Dependencies Only

```bash
pip install -e .
```

Installs: `numpy`, `scipy`, `matplotlib`, `scikit-learn`

### Option 2: Development Setup (Recommended)

```bash
make install-dev
```

Installs:
- Core dependencies
- Development tools (pytest, ruff, black, mypy)
- Pre-commit hooks

### Option 3: Everything

```bash
make install-all
```

Installs:
- Core dependencies
- Optional features (umap-learn, torch, seaborn, networkx, plotly)
- Development tools
- Pre-commit hooks

## 🧪 Testing

### Quick Start

```bash
# Run tests (excludes slow tests like UMAP)
make test

# Run quick smoke tests only
make test-quick

# Run ALL tests including slow ones
make test-all

# Run with coverage report
make test-coverage
```

### Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.slow` - Slow tests (UMAP, large datasets)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.visual` - Tests that generate visual output

```bash
# Skip slow tests (default)
pytest -m "not slow"

# Run only visual generation tests
pytest -m visual

# Run specific test file
pytest tests/test_visual_5.py -v
```

### Test Structure

Each visualization has comprehensive tests:

1. **Import tests** - Module can be imported
2. **Execution tests** - Code runs without errors
3. **Output tests** - Files are generated and valid
4. **Data generation tests** - Synthetic data is correct
5. **Reproducibility tests** - Fixed seeds work
6. **Performance tests** - Meets tier requirements
7. **Edge case tests** - Handles invalid inputs

### Example Test Output

```bash
$ make test
pytest -v -m "not slow" --cov=project_janus --cov-report=term-missing

tests/test_all_visuals.py::TestVisualizationImports::test_import_visual_1 PASSED
tests/test_all_visuals.py::TestVisualizationImports::test_import_visual_5 PASSED
tests/test_visual_5.py::TestLukasiewiczOperations::test_lukasiewicz_and_boundary_cases PASSED
tests/test_visual_5.py::TestVisualizationGeneration::test_visualize_truth_surface_and PASSED
...

---------- coverage: platform linux, python 3.11 ----------
Name                                        Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
project_janus/examples/visual_1_*.py          245     12    95%   89-92
project_janus/examples/visual_5_*.py          187      8    96%   234-237
...
-------------------------------------------------------------------------
TOTAL                                        2847    127    96%

✓ All tests passed
```

## 🎨 Code Quality

### Linting with Ruff

Ruff is a fast, modern Python linter that replaces flake8, pylint, and more:

```bash
# Check code
make lint

# Auto-fix issues
make ruff-fix
```

### Formatting with Black

Black provides consistent, opinionated formatting:

```bash
# Check formatting
black --check project_janus/examples tests

# Apply formatting
make format
```

### Import Sorting with isort

isort organizes imports alphabetically and by type:

```bash
# Check imports
isort --check-only project_janus/examples tests

# Sort imports
isort project_janus/examples tests
```

### Type Checking with mypy

mypy provides optional static type checking:

```bash
mypy project_janus/examples tests
```

### All-in-One Check

```bash
# Format + Lint + Test
make check

# Simulate full CI pipeline
make ci
```

## 🪝 Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit:

```bash
# Install hooks (done automatically with make install-dev)
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks include:
- Trailing whitespace removal
- YAML/TOML/JSON validation
- Python formatting (black)
- Import sorting (isort)
- Linting (ruff)
- Type checking (mypy)
- Shell script linting (shellcheck)
- Markdown linting (markdownlint)

## 🤖 CI/CD Integration

GitHub Actions now runs comprehensive checks on every push:

### Python CI Pipeline

**Matrix Testing** (Python 3.9, 3.10, 3.11, 3.12):
1. Install dependencies
2. Run Ruff linter
3. Run Black formatter check
4. Run isort import check
5. Run mypy type checker
6. Run pytest (excluding slow tests)
7. Upload coverage to Codecov

**Full Test Suite** (Python 3.11):
1. Install all dependencies
2. Run full test suite (including slow tests)
3. Generate HTML coverage report
4. Upload test artifacts

### LaTeX CI Pipeline

1. Compile all `.tex` files
2. Upload PDF artifacts
3. Commit PDFs back to repository

### CI Summary

Final job provides summary of all pipeline results.

## 📝 Configuration Files

### pyproject.toml

Modern Python project configuration:

```toml
[project]
name = "janus-visualizations"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "scipy",
    "matplotlib",
    "scikit-learn",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "black", "isort", "mypy", "pre-commit"]
all = ["umap-learn", "torch", "seaborn", ...]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.black]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--cov=project_janus"]
```

### Makefile

Convenient development commands:

```bash
make help              # Show all commands
make install-dev       # Setup development
make test              # Run tests
make lint              # Run linters
make format            # Format code
make check             # All quality checks
make ci                # Simulate CI pipeline
```

## 🚀 Development Workflow

### Recommended Workflow

1. **Setup** (one time):
   ```bash
   make install-dev
   ```

2. **Before coding**:
   ```bash
   git checkout -b feature/my-feature
   ```

3. **While coding**:
   ```bash
   # Format your code
   make format
   
   # Run tests continuously
   pytest tests/test_visual_5.py -v
   ```

4. **Before committing**:
   ```bash
   # Run all checks
   make check
   
   # Or let pre-commit handle it
   git commit -m "feat: add new feature"
   ```

5. **Before pushing**:
   ```bash
   # Simulate full CI
   make ci
   ```

## 📊 Coverage Reports

### Terminal Coverage

```bash
pytest --cov=project_janus --cov-report=term-missing
```

Shows coverage with line numbers of missing coverage.

### HTML Coverage

```bash
make test-coverage
open htmlcov/index.html
```

Interactive HTML report showing exactly which lines need tests.

### XML Coverage (for CI)

```bash
pytest --cov=project_janus --cov-report=xml
```

Used by Codecov for tracking coverage over time.

## 🎯 Performance Testing

Performance tests verify visualizations meet tier requirements:

```python
@pytest.mark.slow
def test_tier_2_performance(output_dir, timer):
    """Test Tier 2 visuals complete in < 1s."""
    with timer() as t:
        visualize_opal_decision(
            regime="volatile",
            n_steps=100,
            save_path=str(output_dir / "perf.png"),
            dpi=100
        )
    
    assert t.elapsed < 1.0, f"Took {t.elapsed:.2f}s (expected < 1s)"
```

## 🐛 Troubleshooting

### Import Errors

If you see import errors in tests:

```bash
# Reinstall in editable mode
pip install -e .
```

### Missing Dependencies

For optional dependencies:

```bash
# Install specific optional group
pip install -e .[advanced]  # umap, torch, seaborn
pip install -e .[graph]     # networkx
pip install -e .[all]       # everything
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# See what failed
pre-commit run --all-files

# Fix formatting issues
make format

# Try again
git commit -m "your message"
```

### CI Failures

If CI fails but tests pass locally:

```bash
# Simulate CI locally
make ci

# Check specific Python version
pyenv install 3.9.18
pyenv local 3.9.18
pytest
```

## 📚 Additional Resources

- **pyproject.toml**: Full configuration reference
- **CONTRIBUTING.md**: Detailed development guide
- **tests/conftest.py**: Pytest fixtures and helpers
- **tests/test_visual_5.py**: Example test suite
- **.pre-commit-config.yaml**: Pre-commit hook configuration

## ✅ Quick Reference

### Installation
```bash
make install-dev        # Development setup
make install-all        # Everything
```

### Testing
```bash
make test              # Quick tests
make test-all          # All tests
make test-coverage     # With HTML report
```

### Code Quality
```bash
make format            # Format code
make lint              # Check code
make check             # Format + lint + test
make ci                # Full CI simulation
```

### Visualization
```bash
make visual-generate   # Generate all visuals
make visual-v7-v11     # Generate V7 and V11
make visual-clean      # Clean outputs
```

### LaTeX
```bash
make latex             # Build PDFs
make latex-clean       # Clean aux files
```

### Cleanup
```bash
make clean             # Python cache
make clean-all         # Everything
```

---

**Status**: ✅ Complete Python development infrastructure ready for FKS implementation!