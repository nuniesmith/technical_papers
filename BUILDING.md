# Building JANUS Documentation

This guide explains how to build the JANUS technical documentation from LaTeX source files.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Build Methods](#build-methods)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Prerequisites

### Required Software

1. **LaTeX Distribution**
   - **Linux (Ubuntu/Debian)**: `texlive-full`
   - **macOS**: MacTeX
   - **Windows**: MiKTeX or TeX Live
   - **Arch Linux**: `texlive-most`

2. **Bash Shell**
   - Linux/macOS: Built-in
   - Windows: Git Bash, WSL, or Cygwin

### Installation Commands

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y texlive-full
```

#### macOS (Homebrew)
```bash
brew install --cask mactex
# After installation, update PATH:
eval "$(/usr/libexec/path_helper)"
```

#### Arch Linux
```bash
sudo pacman -S texlive-most texlive-latexextra
```

#### Windows
1. Download [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/)
2. Install Git Bash from [git-scm.com](https://git-scm.com/)
3. Run all commands in Git Bash

### Required LaTeX Packages

The following packages are required:
- `pmboxdraw` - For box drawing characters
- `newunicodechar` - For Unicode symbol support
- `algorithm` / `algpseudocode` - For algorithm formatting
- `tcolorbox` - For colored boxes
- `mathtools` - For advanced math formatting
- `hyperref` - For hyperlinks
- `xurl` - For URL formatting

Most packages are included in `texlive-full` or MacTeX complete installations.

## Quick Start

### One-Command Build (Recommended)

```bash
cd technical_papers
chmod +x scripts/build.sh
./scripts/build.sh
```

This script will:
1. ✅ Check for required LaTeX packages
2. 🔧 Auto-install missing dependencies (on supported systems)
3. 📄 Compile all 6 PDFs:
   - `janus_main.pdf` - Architecture Overview
   - `janus_forward.pdf` - Forward Service
   - `janus_backward.pdf` - Backward Service
   - `janus_neuromorphic_architecture.pdf` - Neuromorphic Design
   - `janus_rust_implementation.pdf` - Rust Implementation
   - `complete.pdf` - Complete Combined Edition
4. 🧹 Clean up auxiliary files
5. 📊 Display build summary

**Output Location:** `pdf/` directory

## Build Methods

### Method 1: Automated Build Script (Recommended)

**Full Build:**
```bash
./scripts/build.sh
```

**Script Features:**
- Automatic dependency checking
- Colored output with progress indicators
- Two-pass compilation for TOC/references
- Automatic cleanup
- Git status check for committing PDFs

### Method 2: Manual Build (Individual Documents)

Navigate to the project directory:
```bash
cd technical_papers/project_janus
```

#### Build Architecture Overview
```bash
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
mv janus_main.pdf ../pdf/
```

#### Build Forward Service
```bash
pdflatex -interaction=nonstopmode -jobname=janus_forward forward.tex
pdflatex -interaction=nonstopmode -jobname=janus_forward forward.tex
mv janus_forward.pdf ../pdf/
```

#### Build Backward Service
```bash
pdflatex -interaction=nonstopmode -jobname=janus_backward backward.tex
pdflatex -interaction=nonstopmode -jobname=janus_backward backward.tex
mv janus_backward.pdf ../pdf/
```

#### Build Neuromorphic Architecture
```bash
pdflatex -interaction=nonstopmode -jobname=janus_neuromorphic_architecture neuro.tex
pdflatex -interaction=nonstopmode -jobname=janus_neuromorphic_architecture neuro.tex
mv janus_neuromorphic_architecture.pdf ../pdf/
```

#### Build Rust Implementation
```bash
pdflatex -interaction=nonstopmode -jobname=janus_rust_implementation rust.tex
pdflatex -interaction=nonstopmode -jobname=janus_rust_implementation rust.tex
mv janus_rust_implementation.pdf ../pdf/
```

#### Build Complete Edition
```bash
pdflatex -interaction=nonstopmode complete.tex
pdflatex -interaction=nonstopmode complete.tex
pdflatex -interaction=nonstopmode complete.tex  # 3 passes for master doc
mv complete.pdf ../pdf/
```

**Why two/three passes?**
- First pass: Generate content
- Second pass: Build table of contents and resolve references
- Third pass (complete edition): Ensure all cross-references between parts are resolved

### Method 3: Using latexmk (Advanced)

For automatic multi-pass compilation:

```bash
cd project_janus
latexmk -pdf -jobname=janus_main main.tex
latexmk -pdf -jobname=janus_forward forward.tex
latexmk -pdf -jobname=janus_backward backward.tex
latexmk -pdf -jobname=janus_neuromorphic_architecture neuro.tex
latexmk -pdf -jobname=janus_rust_implementation rust.tex
latexmk -pdf complete.tex
```

### Method 4: GitHub Actions (Automated CI/CD)

Every push to `main` branch automatically builds all PDFs:

1. Edit `.tex` files locally
2. Commit and push:
   ```bash
   git add project_janus/*.tex
   git commit -m "Update documentation"
   git push
   ```
3. GitHub Actions will:
   - Compile all 6 PDFs
   - Commit them back to the repository
   - Make them available as artifacts

View workflow: `.github/workflows/build-pdf.yml`

## Troubleshooting

### Problem: "pdflatex: command not found"

**Solution:**
```bash
# Check if LaTeX is installed
which pdflatex

# If not found, install TeX Live
# Ubuntu/Debian:
sudo apt-get install texlive-full

# macOS:
brew install --cask mactex

# Then restart your terminal
```

### Problem: Missing LaTeX Packages

**Error Example:**
```
! LaTeX Error: File `pmboxdraw.sty' not found.
```

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-fonts-extra texlive-latex-extra texlive-science
```

**Solution (macOS with MacTeX):**
```bash
sudo tlmgr update --self
sudo tlmgr install pmboxdraw newunicodechar algorithm2e tcolorbox mathtools
```

**Solution (MiKTeX - Windows):**
1. Open MiKTeX Console
2. Go to "Packages"
3. Search for missing package
4. Click "Install"

Or enable auto-install:
1. Open MiKTeX Console
2. Go to Settings
3. Set "Install missing packages on-the-fly" to "Yes"

### Problem: Unicode Character Errors

**Error Example:**
```
! Package inputenc Error: Unicode character → (U+2192)
```

**Solution:**
This is already handled by `\newunicodechar` declarations in the `.tex` files. If you see this error:

1. Check that `\usepackage{newunicodechar}` is present
2. Ensure you're using UTF-8 encoding
3. Verify the character is declared in the preamble

### Problem: Build Script Permission Denied

**Error:**
```bash
bash: ./scripts/build.sh: Permission denied
```

**Solution:**
```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### Problem: PDFs Not Generated

**Symptoms:**
- Script completes but no PDFs in `pdf/` directory
- `.log` files show errors

**Solution:**
```bash
# Check the log files for specific errors
cd project_janus
cat janus_main.log | grep -i error

# Common fixes:
# 1. Remove auxiliary files and rebuild
rm -f *.aux *.log *.out *.toc
./scripts/build.sh

# 2. Update LaTeX distribution
# Ubuntu:
sudo apt-get update && sudo apt-get upgrade texlive-*

# macOS:
sudo tlmgr update --self --all
```

### Problem: Out of Memory

**Error:**
```
TeX capacity exceeded, sorry [main memory size=...]
```

**Solution:**
Increase TeX memory limits by editing `texmf.cnf`:

```bash
# Find texmf.cnf location
kpsewhich texmf.cnf

# Edit (may require sudo)
# Increase: main_memory = 12000000
# Increase: extra_mem_bot = 12000000
```

Or use LuaLaTeX (has dynamic memory):
```bash
lualatex -interaction=nonstopmode -jobname=janus_main main.tex
```

## FAQ

### Q: How long does compilation take?

**A:** On a typical system:
- Single document: 10-30 seconds (2 passes)
- Complete edition: 45-90 seconds (3 passes)
- Full suite (6 PDFs): 2-4 minutes

### Q: Can I build just one document?

**A:** Yes! Navigate to `project_janus/` and run:
```bash
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
```

### Q: Why do we need multiple passes?

**A:** LaTeX builds documents in passes:
1. **First pass**: Generates content, creates auxiliary files
2. **Second pass**: Resolves cross-references, builds TOC
3. **Third pass** (for `complete.tex`): Ensures all inter-document references are correct

### Q: Can I use XeLaTeX or LuaLaTeX instead of pdfLaTeX?

**A:** Yes, but you'll need to modify font settings:

```bash
xelatex -interaction=nonstopmode -jobname=janus_main main.tex
# or
lualatex -interaction=nonstopmode -jobname=janus_main main.tex
```

Update `.tex` files to use `fontspec` instead of `helvet` for better Unicode support.

### Q: How do I commit the generated PDFs?

**A:** The build script shows git status. To commit:

```bash
git add pdf/*.pdf
git commit -m "Update compiled PDFs"
git push
```

Or let GitHub Actions handle it automatically on push.

### Q: Can I customize the output filename?

**A:** Yes, use the `-jobname` flag:
```bash
pdflatex -interaction=nonstopmode -jobname=my_custom_name main.tex
```

### Q: How do I build on Windows without WSL?

**A:**
1. Install MiKTeX: https://miktex.org/download
2. Install Git Bash: https://git-scm.com/
3. Open Git Bash in the repository
4. Run: `./scripts/build.sh`

Alternatively, use PowerShell:
```powershell
cd project_janus
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
```

### Q: The complete edition (`complete.tex`) fails to compile

**A:** This document uses `\input{}` to include other `.tex` files. Ensure:

1. All individual `.tex` files are syntactically correct
2. You're running from the `project_janus/` directory
3. All files are in the same directory

Common fix:
```bash
cd project_janus
# Test individual documents first
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
# If that works, then try complete:
pdflatex -interaction=nonstopmode janus_complete.tex
```

### Q: How do I update the build script?

**A:** The build script is located at `scripts/build.sh`. Edit it with any text editor:

```bash
nano scripts/build.sh
# or
vim scripts/build.sh
# or
code scripts/build.sh  # VS Code
```

After editing, test with:
```bash
bash -n scripts/build.sh  # Syntax check
./scripts/build.sh        # Full test
```

## Build Performance Tips

### Speed Up Compilation

1. **Use draft mode for quick previews:**
   ```bash
   pdflatex -interaction=nonstopmode -draftmode main.tex
   ```

2. **Skip second pass during development:**
   ```bash
   pdflatex -interaction=nonstopmode -jobname=janus_main main.tex
   # Skip second pass, just view with broken references
   ```

3. **Use `latexmk` with continuous preview:**
   ```bash
   latexmk -pdf -pvc -jobname=janus_main main.tex
   # Opens PDF viewer, auto-recompiles on file changes
   ```

4. **Parallel builds (if building all documents):**
   ```bash
   # In separate terminals:
   cd project_janus
   pdflatex -interaction=nonstopmode -jobname=janus_main main.tex &
   pdflatex -interaction=nonstopmode -jobname=janus_forward forward.tex &
   pdflatex -interaction=nonstopmode -jobname=janus_backward backward.tex &
   wait
   ```

### Reduce PDF File Size

If PDFs are too large:

```bash
# Using Ghostscript
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=janus_main_compressed.pdf janus_main.pdf

# Or use pdf2ps + ps2pdf
pdf2ps janus_main.pdf janus_main.ps
ps2pdf janus_main.ps janus_main_compressed.pdf
```

## Additional Resources

- **LaTeX Documentation**: https://www.latex-project.org/help/documentation/
- **TeX Stack Exchange**: https://tex.stackexchange.com/
- **LaTeX Wikibook**: https://en.wikibooks.org/wiki/LaTeX
- **Overleaf Documentation**: https://www.overleaf.com/learn

## Getting Help

If you encounter issues not covered here:

1. **Check the log files**: Look in `project_janus/*.log` for detailed errors
2. **Search TeX Stack Exchange**: Most LaTeX errors have been solved there
3. **Open an issue**: https://github.com/nuniesmith/technical_papers/issues
4. **Provide details**: Include OS, TeX distribution version, and full error messages

---

**Happy Building! 📚**