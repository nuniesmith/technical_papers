# CI/CD and Codecov Setup Guide

This guide explains the unified CI/CD pipeline for Project JANUS, including documentation builds, Rust testing, and code coverage with Codecov.

## 📋 Overview

The unified CI workflow (`.github/workflows/ci.yml`) handles:

1. **📚 Documentation Pipeline**
   - LaTeX quality checks
   - PDF compilation (all 6 documents)
   - Documentation metrics
   - Automatic PDF commits to repository

2. **🦀 Rust Code Pipeline** (when Rust code exists)
   - Multi-platform testing (Linux, macOS, Windows)
   - Code coverage with Codecov
   - Performance benchmarks
   - Security audits
   - Release artifact builds

## 🚀 Quick Start

### For Documentation Only (Current State)

The workflow is ready to use! Just push your `.tex` files:

```bash
git add project_janus/*.tex
git commit -m "Update documentation"
git push
```

GitHub Actions will automatically:
- ✅ Check documentation quality
- ✅ Compile all 6 PDFs
- ✅ Commit PDFs back to the repository
- ✅ Generate documentation metrics

### For Future Rust Implementation

When you add Rust code, the workflow automatically detects it and runs the full test suite with coverage.

## 🔧 Codecov Setup

### Step 1: Sign Up for Codecov

1. Go to [codecov.io](https://codecov.io)
2. Click **"Sign up with GitHub"**
3. Authorize Codecov to access your repositories

### Step 2: Add Your Repository

1. Once logged in, click **"Add new repository"**
2. Find `technical_papers` in the list
3. Click **"Setup repo"**

### Step 3: Get Your Codecov Token

1. In your repository settings on Codecov, find the **"Repository Upload Token"**
2. Copy the token (looks like: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`)

### Step 4: Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **"New repository secret"**
4. Name: `CODECOV_TOKEN`
5. Value: Paste your Codecov token
6. Click **"Add secret"**

### Step 5: Test It!

Once you have Rust code in your repository:

```bash
# Create a simple Rust project structure
cargo init --name janus-forward
mkdir src
echo 'fn main() { println!("Hello JANUS!"); }' > src/main.rs

# Add a test
echo '#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}' >> src/main.rs

# Commit and push
git add Cargo.toml src/
git commit -m "Add Rust implementation skeleton"
git push
```

GitHub Actions will:
1. Run tests on Linux, macOS, and Windows
2. Generate coverage report
3. Upload to Codecov
4. Display coverage badge

## 📊 Viewing Coverage Reports

### On Codecov Dashboard

After your first successful push with tests:

1. Go to [codecov.io/gh/YOUR_USERNAME/technical_papers](https://codecov.io)
2. View coverage percentage, trends, and file-by-file breakdown
3. See which lines are covered/uncovered

### In GitHub Pull Requests

Codecov automatically comments on PRs with:
- Coverage change (±%)
- Diff coverage (coverage of changed lines)
- Detailed file-by-file breakdown

### As Artifacts

HTML coverage reports are uploaded as artifacts on every run:
1. Go to your workflow run
2. Scroll to **Artifacts** section
3. Download `rust-coverage-report`
4. Open `index.html` in your browser

## 🎯 Workflow Triggers

The CI runs on:

### Documentation Changes
```yaml
paths:
  - "project_janus/*.tex"
  - "scripts/build.sh"
  - ".github/workflows/ci.yml"
```

Triggers: Documentation pipeline only

### Rust Code Changes
```yaml
paths:
  - "src/**"
  - "Cargo.toml"
  - "Cargo.lock"
```

Triggers: Both documentation and Rust pipelines

### Manual Trigger
```bash
# Via GitHub UI: Actions → JANUS Unified CI → Run workflow
# Or via GitHub CLI:
gh workflow run ci.yml
```

## 📈 Adding Coverage Badges

### Codecov Badge

Add to your `README.md`:

```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/technical_papers/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/technical_papers)
```

### CI Status Badge

Add to your `README.md`:

```markdown
[![CI](https://github.com/YOUR_USERNAME/technical_papers/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/technical_papers/actions/workflows/ci.yml)
```

## 🔍 Understanding the Pipeline

### Job Dependencies

```
documentation-quality
    ↓
build-latex (runs in parallel after quality checks)
    ↓
ci-summary

rust-check (detects if Rust code exists)
    ↓
rust-test (multi-platform)
    ↓
rust-coverage (Linux only)
    ↓
rust-security, rust-benchmark, rust-artifacts (parallel)
    ↓
ci-summary
```

### Conditional Execution

The Rust pipeline only runs when:
1. Rust source code exists (`src/` directory and `Cargo.toml`)
2. Rust files were modified in the commit
3. Manually triggered with `workflow_dispatch`

This saves CI minutes when only documentation changes.

## 🛠️ Customization

### Adjust Coverage Thresholds

Edit `.github/workflows/ci.yml`:

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    files: lcov.info
    fail_ci_if_error: true  # Fail if coverage upload fails
    verbose: true
```

### Add Coverage Requirements

Create `codecov.yml` in your repository root:

```yaml
coverage:
  status:
    project:
      default:
        target: 80%  # Fail if project coverage < 80%
        threshold: 5%  # Allow 5% drop
    patch:
      default:
        target: 90%  # Require 90% coverage on new code

comment:
  layout: "reach,diff,flags,tree"
  behavior: default
  require_changes: false
```

### Disable Specific Jobs

To skip certain jobs, add `if: false`:

```yaml
rust-benchmark:
  name: ⚡ Rust Performance Benchmarks
  runs-on: ubuntu-latest
  if: false  # Disable benchmarks
```

## 🐛 Troubleshooting

### "Codecov token not found"

**Problem:** CI fails with "Error: Codecov token not found"

**Solution:**
1. Verify token is added to GitHub Secrets
2. Check secret name is exactly `CODECOV_TOKEN`
3. Re-run the workflow

### Coverage not uploading

**Problem:** Tests pass but coverage doesn't appear on Codecov

**Solution:**
1. Check `lcov.info` was generated (download artifacts)
2. Verify Codecov token is valid
3. Check Codecov dashboard for upload errors
4. Try re-running with `verbose: true`

### "No Rust code found"

**Problem:** Rust jobs are skipped

**Solution:** This is expected! The workflow detects if Rust code exists. To enable:
1. Add `Cargo.toml` to repository root
2. Create `src/` directory with Rust code
3. Commit and push

### PDF commits failing

**Problem:** "Push failed" when committing PDFs

**Solution:**
1. Check `contents: write` permission is set
2. Verify branch protection rules allow GitHub Actions to push
3. Check if `[skip ci]` is in commit message (prevents loops)

## 📚 Additional Resources

- **Codecov Documentation:** https://docs.codecov.com/
- **GitHub Actions Documentation:** https://docs.github.com/actions
- **cargo-llvm-cov:** https://github.com/taiki-e/cargo-llvm-cov
- **Rust Testing Guide:** https://doc.rust-lang.org/book/ch11-00-testing.html

## 🎓 Best Practices

### For Maximum Coverage

1. **Write comprehensive tests:**
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_happy_path() { /* ... */ }

       #[test]
       fn test_edge_cases() { /* ... */ }

       #[test]
       fn test_error_handling() { /* ... */ }
   }
   ```

2. **Test all modules:**
   - Unit tests in each module
   - Integration tests in `tests/` directory
   - Doc tests in documentation comments

3. **Use property-based testing:**
   ```rust
   use quickcheck::quickcheck;

   quickcheck! {
       fn reversible(xs: Vec<i32>) -> bool {
           xs == reverse(&reverse(&xs))
       }
   }
   ```

### For CI Performance

1. **Use caching effectively** (already configured)
2. **Run expensive jobs conditionally** (benchmarks only on `main`)
3. **Skip redundant tests** (nightly builds excluded on macOS/Windows)

### For Documentation

1. **Keep LaTeX clean** - Quality checks catch common issues
2. **Use meaningful commit messages** - Shows in build logs
3. **Review metrics** - Track documentation growth over time

## 🔐 Security Notes

- ✅ `CODECOV_TOKEN` is a **secret** - never commit it to code
- ✅ GitHub automatically redacts secrets in logs
- ✅ Use `secrets.GITHUB_TOKEN` for GitHub API calls (automatically provided)
- ✅ Dependabot and security audits run automatically

## 📞 Getting Help

If you encounter issues:

1. Check workflow logs in GitHub Actions
2. Review this guide's troubleshooting section
3. Check [Codecov Community Forums](https://community.codecov.com/)
4. Open an issue in the repository

---

**Happy Testing! 🚀**