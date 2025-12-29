# Quick Start Guide - Project JANUS

Get up and running in 5 minutes!

## 🚀 One-Command Setup

```bash
./setup.sh
```

That's it! This will:
1. ✅ Create a Python virtual environment
2. ✅ Install all development dependencies
3. ✅ Set up pre-commit hooks
4. ✅ Verify the installation

## 📋 Step-by-Step Setup

### Option A: Automated (Recommended)

```bash
# Full development setup
./setup.sh

# Or minimal (core only)
./setup.sh --minimal

# Or everything (including optional deps)
./setup.sh --all
```

### Option B: Manual

```bash
# 1. Create virtual environment
make venv

# 2. Install dependencies
make install-dev

# 3. Activate virtual environment
source venv/bin/activate
```

## ✅ Verify Installation

```bash
# Check everything is working
make info

# Run tests
make test

# Expected output:
# ✓ All tests passed
# Coverage: ~96%
```

## 🎨 Try It Out

### Generate Your First Visualization

```bash
# Activate venv (if not already active)
source venv/bin/activate

# Generate Łukasiewicz truth surfaces
cd project_janus/examples
python visual_5_ltn_truth_surface.py --save-all --output-dir ../../outputs

# Check outputs
ls -lh ../../outputs/
```

### Run the Test Suite

```bash
# Quick tests (30 seconds)
make test

# All tests including slow ones (2-5 minutes)
make test-all

# Generate HTML coverage report
make test-coverage
open htmlcov/index.html
```

### Check Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run everything (format + lint + test)
make check
```

## 🛠️ Common Commands

| Command | Description |
|---------|-------------|
| `make venv` | Create virtual environment |
| `make install-dev` | Install dev dependencies |
| `make test` | Run tests (fast) |
| `make test-all` | Run all tests |
| `make lint` | Check code quality |
| `make format` | Format code |
| `make check` | Format + lint + test |
| `make ci` | Simulate full CI pipeline |
| `make clean` | Clean build artifacts |
| `make help` | Show all commands |

## 🐍 Virtual Environment

### Activate
```bash
source venv/bin/activate
```

### Deactivate
```bash
deactivate
```

### Remove
```bash
make clean-venv
```

## 📊 What's Included

After setup, you have:

- ✅ **Python 3.9+** virtual environment
- ✅ **Core libraries**: numpy, scipy, matplotlib, scikit-learn
- ✅ **Dev tools**: pytest, ruff, black, isort, mypy
- ✅ **Pre-commit hooks**: Automatic code quality checks
- ✅ **13 visualizations**: All JANUS visual implementations
- ✅ **Comprehensive tests**: 96% coverage

## 🎯 Next Steps

### For Development

1. **Read the docs**:
   - `CONTRIBUTING.md` - Development workflow
   - `PYTHON_SETUP.md` - Detailed setup guide
   - `project_janus/README.md` - Project overview

2. **Explore the code**:
   - `project_janus/examples/` - Visualization scripts
   - `tests/` - Test examples

3. **Start coding**:
   ```bash
   git checkout -b feature/my-feature
   # Make changes
   make check
   git commit -m "feat: add new feature"
   ```

### For FKS Implementation

1. **Review visualizations**:
   ```bash
   cd project_janus/examples
   ls visual_*.py
   ```

2. **Check test patterns**:
   ```bash
   less tests/test_visual_5.py
   ```

3. **Use fixtures**:
   - See `tests/conftest.py` for helpers

4. **Run specific visual**:
   ```bash
   python visual_1_gaf_pipeline.py --help
   python visual_7_opal_decision.py --save-all
   ```

## 🐛 Troubleshooting

### Virtual environment not found
```bash
make venv
```

### Import errors
```bash
source venv/bin/activate
make install-dev
```

### Pre-commit hook failures
```bash
make format
git commit -m "your message"
```

### Tests failing
```bash
# Check your environment
make info

# Reinstall dependencies
make clean-venv
make venv
make install-dev
```

## 💬 Getting Help

- **Commands**: `make help`
- **Project info**: `make info`
- **Full docs**: See `CONTRIBUTING.md`
- **Examples**: Check `project_janus/examples/`

## 🎉 You're Ready!

Your environment is now set up for:
- ✅ Developing JANUS visualizations
- ✅ Testing algorithms
- ✅ Implementing the FKS project
- ✅ Contributing to the project

**Time invested**: 5 minutes  
**Time saved per day**: 15+ minutes (automated checks)

Happy coding! 🚀
