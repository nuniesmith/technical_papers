# Technical Papers Repository

[![Build LaTeX Documents](https://github.com/nuniesmith/technical_papers/actions/workflows/ci.yml/badge.svg)](https://github.com/nuniesmith/technical_papers/actions/workflows/ci.yml)

> *A collection of technical papers and documentation with automated LaTeX compilation*

## 📖 Overview

This repository hosts technical papers and documentation with automated PDF generation. Each project maintains its source LaTeX files (`.tex`) and compiled PDFs (`.pdf`) in the same directory.

**Current Projects:**
- **Project JANUS** - Neuromorphic Trading Intelligence System

## 🏗️ Repository Structure

```
technical_papers/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated PDF compilation
├── project_janus/                 # Example project
│   ├── main.tex                   # LaTeX source
│   ├── main.pdf                   # Compiled PDF (auto-generated)
│   ├── forward.tex
│   ├── forward.pdf
│   ├── complete.tex               # Unified document
│   ├── complete.pdf
│   └── extract_content.sh         # Content extraction utility
├── scripts/
│   └── build.sh                   # Project-specific build script
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Download Pre-Built PDFs

PDFs are automatically compiled and committed to the repository:

```bash
git clone https://github.com/nuniesmith/technical_papers.git
cd technical_papers

# Navigate to any project directory
cd project_janus

# PDFs are located next to their source .tex files
ls -lh *.pdf
```

### Option 2: Build Locally

#### Prerequisites

- **LaTeX Distribution**: TeX Live (Linux/Windows) or MacTeX (macOS)
- **Bash**: Available on Linux/macOS/WSL

#### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y texlive-full
```

**macOS (Homebrew):**
```bash
brew install --cask mactex
```

**macOS (MacPorts):**
```bash
sudo port install texlive
```

**Arch Linux:**
```bash
sudo pacman -S texlive-most
```

**Fedora/RHEL:**
```bash
sudo dnf install texlive-scheme-full
```

#### Build Project Documents

If the project includes a build script:

```bash
cd technical_papers
chmod +x scripts/build.sh
./scripts/build.sh
```

Or build individual documents manually:

```bash
cd project_name

# Build a document (run twice for TOC/references)
pdflatex document.tex
pdflatex document.tex

# Or use latexmk for automatic handling
latexmk -pdf document.tex
```

## 🤖 Automated CI/CD

Every push to the `main` branch triggers automated PDF compilation via GitHub Actions.

**Features:**
- ✅ **Auto-discovery**: Automatically finds all `.tex` files with `\documentclass`
- ✅ **Smart compilation**: Runs `pdflatex` twice for TOC and cross-references
- ✅ **Same-directory output**: PDFs are created next to their source files
- ✅ **Auto-commit**: Compiled PDFs are committed back to the repository
- ✅ **Loop prevention**: Uses `[skip ci]` to prevent infinite build loops
- ✅ **Build reports**: Summary of successful/failed compilations

**Workflow:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

### Adding New Documents

Simply create a new `.tex` file anywhere in the repository:

```latex
\documentclass{article}
\begin{document}
Your content here
\end{document}
```

The CI system will automatically detect and compile it!

## 📁 Project Guidelines

### Creating a New Project

1. **Create project directory:**
   ```bash
   mkdir new_project_name
   cd new_project_name
   ```

2. **Add LaTeX documents:**
   ```bash
   # Create your .tex files
   touch paper.tex
   ```

3. **Optional: Add build script:**
   ```bash
   # Create project-specific build automation
   touch build.sh
   chmod +x build.sh
   ```

4. **Commit and push:**
   ```bash
   git add new_project_name/
   git commit -m "Add new project: Project Name"
   git push
   ```

The CI system will automatically compile your documents.

### Project Structure Recommendations

**Simple project (single document):**
```
my_paper/
├── paper.tex
└── paper.pdf (auto-generated)
```

**Complex project (multiple documents):**
```
my_project/
├── main.tex
├── main.pdf
├── appendix_a.tex
├── appendix_a.pdf
├── complete.tex          # Optional: unified document
├── complete.pdf
├── extract_content.sh    # Optional: content extraction
└── README.md             # Optional: project-specific docs
```

## 📝 LaTeX Best Practices

### Required Packages

Ensure your documents load commonly available packages:

```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
```

### Cross-References

Run compilation twice for proper TOC and references:

```bash
pdflatex document.tex  # First pass
pdflatex document.tex  # Second pass (resolves references)
```

Or use `latexmk` for automatic handling:

```bash
latexmk -pdf document.tex
```

### Unicode Support

For unicode characters in LaTeX:

```latex
\usepackage{newunicodechar}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{≥}{\ensuremath{\geq}}
```

## 🔧 Build Script Template

Create a `build.sh` in your project directory:

```bash
#!/bin/bash
set -e

# Compile all documents
for tex_file in *.tex; do
    if [ -f "$tex_file" ]; then
        echo "Building $tex_file..."
        pdflatex -interaction=nonstopmode "$tex_file"
        pdflatex -interaction=nonstopmode "$tex_file"
    fi
done

# Cleanup auxiliary files
rm -f *.aux *.log *.out *.toc *.synctex.gz

echo "Build complete!"
```

## 📊 Current Projects

### Project JANUS - Neuromorphic Trading Intelligence

A comprehensive technical specification for a brain-inspired algorithmic trading system.

**Documents:**
- `main.pdf` - Architectural overview and philosophy
- `forward.pdf` - Real-time trading service
- `backward.pdf` - Memory consolidation and learning
- `neuro.pdf` - Neuromorphic architecture mapping
- `rust.pdf` - Rust implementation guide
- `complete.pdf` - Unified document

**Location:** [`project_janus/`](project_janus/)

---

## 🤝 Contributing

Contributions are welcome! To add or improve documentation:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/new-paper
   ```
3. **Add your LaTeX files**
4. **Test compilation locally**
5. **Submit a pull request**

### Contribution Guidelines

- ✅ Use standard LaTeX packages (avoid exotic dependencies)
- ✅ Include README in project directories for complex projects
- ✅ Test compilation before pushing
- ✅ Use meaningful commit messages
- ✅ Keep source files (`.tex`) and PDFs in the same directory

## 📄 License

Unless otherwise specified, content in this repository is licensed under:

- **Documentation/Papers**: [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code/Scripts**: MIT License

Individual projects may specify their own licenses.

## 🛠️ Troubleshooting

### LaTeX Compilation Errors

**Issue:** "File not found" errors
```bash
# Solution: Install missing packages
sudo apt-get install texlive-latex-extra texlive-fonts-extra
```

**Issue:** Unicode characters not rendering
```bash
# Solution: Add unicode support
\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
```

**Issue:** References showing as "??"
```bash
# Solution: Run pdflatex twice
pdflatex document.tex
pdflatex document.tex
```

### CI/CD Issues

**Issue:** Build fails on GitHub Actions
- Check the Actions tab for detailed logs
- Ensure all required packages are in `texlive-full`
- Test locally before pushing

**Issue:** Infinite commit loops
- Workflow uses `[skip ci]` in commit messages
- Ensure you're not modifying committed PDFs manually

## 📚 Resources

### LaTeX Documentation
- [LaTeX Project](https://www.latex-project.org/)
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [CTAN - Comprehensive TeX Archive Network](https://ctan.org/)

### GitHub Actions
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [LaTeX Action by xu-cheng](https://github.com/xu-cheng/latex-action)

## 👤 Maintainer

**Jordan Smith**
- GitHub: [@nuniesmith](https://github.com/nuniesmith)

## 🙏 Acknowledgments

- LaTeX Project contributors
- GitHub Actions community
- Open source LaTeX package maintainers

---

**Adding a new project?** Just create a directory, add your `.tex` files, commit, and push. The CI system handles the rest!