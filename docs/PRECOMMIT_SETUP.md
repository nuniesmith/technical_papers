# Pre-commit Setup Guide

This guide explains how to set up pre-commit hooks for the Project JANUS repository.

## What are Pre-commit Hooks?

Pre-commit hooks automatically check your code before each commit, ensuring:

- Code is properly formatted (Black, isort)
- No linting errors (Ruff)
- Type hints are correct (mypy)
- Markdown, YAML, and other files are valid

## Installation Options

### Option 1: Global Installation (Recommended)

Install pre-commit globally so it works without activating the virtual environment.

```bash
# Step 1: Install pipx (if not already installed)
sudo apt install pipx
pipx ensurepath

# Step 2: Install pre-commit globally
pipx install pre-commit

# Step 3: Install Git hooks in the repository
cd /path/to/technical_papers
pre-commit install

# Step 4: Test it works
pre-commit --version
```

**Pros**:

- ✅ Works without activating venv
- ✅ Available system-wide
- ✅ Simplifies daily workflow
- ✅ No "pre-commit not found" errors

**Cons**:

- ⚠️ Requires pipx installation
- ⚠️ One-time system setup needed

### Option 2: Virtual Environment Installation

Install pre-commit in the project's virtual environment.

```bash
# Step 1: Create and activate venv
make venv
source venv/bin/activate

# Step 2: Install dependencies
make install-dev

# Step 3: Install Git hooks
pre-commit install

# Step 4: Remember to activate venv before committing
source venv/bin/activate
git commit -m "your message"
```

**Pros**:

- ✅ No system-level installation needed
- ✅ Isolated to project
- ✅ Matches CI environment exactly

**Cons**:

- ⚠️ Must activate venv before every commit
- ⚠️ Easy to forget and get "not found" errors

## Usage

Once installed (either way), pre-commit runs automatically on `git commit`:

```bash
# Pre-commit runs automatically
git commit -m "Add new feature"

# If hooks fail, fix the issues and commit again
git add .
git commit -m "Add new feature"
```

### Manual Execution

Run hooks without committing:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black
pre-commit run ruff
```

### Bypassing Hooks

**⚠️ Not recommended**, but you can bypass hooks if needed:

```bash
git commit --no-verify -m "Emergency fix"
```

Note: CI will still run all checks, so bypassing locally just delays feedback.

## Troubleshooting

### Error: "pre-commit: command not found"

**Cause**: You installed pre-commit in the venv but didn't activate it.

**Solutions**:

1. Activate venv: `source venv/bin/activate`
2. OR install globally: `pipx install pre-commit`

### Error: "RPC request RunGitHook failed"

**Cause**: Same as above - pre-commit not in PATH.

**Solution**: Use Option 1 (global installation) from this guide.

### Hooks are slow

**Cause**: First run downloads hook environments.

**Solution**:

- Wait for first run to complete (one-time setup)
- Subsequent runs are much faster (~5-10 seconds)

### Want to update pre-commit

```bash
# Global installation
pipx upgrade pre-commit

# Venv installation
source venv/bin/activate
pip install --upgrade pre-commit
```

### Want to uninstall hooks

```bash
# Remove Git hooks
pre-commit uninstall

# Uninstall pre-commit (global)
pipx uninstall pre-commit

# Uninstall pre-commit (venv)
source venv/bin/activate
pip uninstall pre-commit
```

## Configured Hooks

The repository runs these hooks on every commit:

1. **trailing-whitespace**: Remove trailing whitespace
2. **end-of-file-fixer**: Ensure files end with newline
3. **check-yaml**: Validate YAML syntax
4. **check-added-large-files**: Prevent large file commits
5. **black**: Format Python code
6. **isort**: Sort Python imports
7. **ruff**: Lint Python code
8. **mypy**: Type check Python code
9. **markdownlint**: Lint Markdown files
10. **shellcheck**: Lint shell scripts

See `.pre-commit-config.yaml` for full configuration.

## CI Integration

Pre-commit checks also run in GitHub Actions CI:

- Every push and PR triggers the same checks
- CI fails if any hook fails
- This ensures code quality regardless of local setup

Even if you skip hooks locally (`--no-verify`), CI will catch issues.

## Recommendation

**For most users**: Use **Option 1 (Global Installation)**

- Simplest daily workflow
- No need to remember venv activation
- Works across all projects

**For strict isolation**: Use **Option 2 (Venv Installation)**

- Matches CI environment exactly
- No system-level tools required
- Good for reproducibility testing

## Summary

```bash
# Quick setup (recommended)
sudo apt install pipx
pipx install pre-commit
pre-commit install

# Now commit normally
git commit -m "your changes"
```

That's it! Pre-commit will keep your code clean automatically.
