# JANUS CI/CD Quick Reference

> **TL;DR:** Push your code, GitHub Actions handles the rest. Documentation builds automatically. Rust testing activates when you add Rust code.

## 🚦 Current Status

- ✅ **Documentation Pipeline**: Active and working
- ⏳ **Rust Pipeline**: Activates when you add Rust code
- 📊 **Codecov Integration**: Ready (needs token setup)

## 📝 One-Minute Setup

### For Documentation (Ready Now)

```bash
# Edit LaTeX files
vim project_janus/main.tex

# Commit and push
git add project_janus/*.tex
git commit -m "Update documentation"
git push

# ✨ PDFs auto-generated and committed back to repo
```

### For Codecov (When You Have Tests)

1. **Get Token**: Go to [codecov.io](https://codecov.io) → Sign in → Add repo → Copy token
2. **Add Secret**: GitHub repo → Settings → Secrets → New secret
   - Name: `CODECOV_TOKEN`
   - Value: (paste token)
3. **Done!** Coverage reports upload automatically

## 🎯 What Runs When

| You Push... | What Happens |
|-------------|--------------|
| `.tex` files | ✅ Quality checks<br>✅ Build PDFs<br>✅ Commit PDFs back |
| `src/**` + `Cargo.toml` | ✅ Above +<br>✅ Run tests (Linux/Mac/Win)<br>✅ Generate coverage<br>✅ Security audit<br>✅ Build binaries |
| Both | ✅ Everything! |

## 📊 Where to Find Results

### GitHub Actions Tab
- **Summary**: See all job results
- **Artifacts**: Download PDFs, coverage reports, binaries
- **Logs**: Debug any failures

### Codecov Dashboard
- **Coverage %**: Overall and per-file
- **Trends**: Coverage over time
- **PR Comments**: Coverage changes in PRs

## 🔧 Common Commands

### Trigger CI Manually
```bash
# Via GitHub CLI
gh workflow run ci.yml

# Via web: Actions tab → "JANUS Unified CI" → "Run workflow"
```

### Build Locally (Documentation)
```bash
# One command
./scripts/build.sh

# Manual
cd project_janus
pdflatex -jobname=janus_main main.tex
pdflatex -jobname=janus_main main.tex  # Run twice!
```

### Test Locally (Rust)
```bash
# Run tests
cargo test --all-features

# Generate coverage
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html  # View report
```

## 🐛 Quick Troubleshooting

### PDFs Not Building?
1. Check LaTeX syntax errors in logs
2. Verify all required packages installed
3. Try building locally first

### Coverage Not Uploading?
1. Check `CODECOV_TOKEN` is in GitHub Secrets
2. Verify token name is exactly `CODECOV_TOKEN`
3. Check Codecov dashboard for errors

### Rust Jobs Skipped?
- **Expected!** Only runs when `src/` and `Cargo.toml` exist
- To enable: Add Rust code to repository

## 📚 Full Documentation

- **Detailed CI Guide**: [`CI_SETUP.md`](../CI_SETUP.md)
- **Build Instructions**: [`BUILDING.md`](../BUILDING.md)
- **Main README**: [`README.md`](../README.md)

## 🎓 Pro Tips

1. **Use `[skip ci]` in commits** to skip CI:
   ```bash
   git commit -m "Fix typo [skip ci]"
   ```

2. **Watch the Summary** - Best overview of what ran:
   - GitHub Actions → Your workflow run → Scroll to bottom

3. **Download Artifacts** for offline viewing:
   - HTML coverage reports
   - Compiled PDFs
   - Release binaries

4. **Check metrics** in documentation pipeline:
   - Total lines of LaTeX
   - PDF sizes
   - Document structure

## 🚀 Adding Rust Implementation

Ready to implement the JANUS system in Rust?

```bash
# Initialize Rust workspace
cargo init --name janus-forward
mkdir src

# Add first module
cat > src/main.rs << 'EOF'
fn main() {
    println!("JANUS Forward Service starting...");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
EOF

# Commit and push
git add Cargo.toml src/
git commit -m "Add Rust implementation skeleton"
git push

# 🎉 Full CI pipeline now active!
```

## 📞 Need Help?

1. Check workflow logs in Actions tab
2. Review error messages carefully
3. Consult full docs linked above
4. Open an issue if stuck

---

**Remember:** The CI is designed to help you, not block you. If something fails, it's catching a real issue! 🛡️