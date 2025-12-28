# Project JANUS - Consolidated Documentation

This directory contains the unified technical specification for **Project JANUS: Neuromorphic Trading Intelligence**.

## 📄 Main Document

**`janus.tex`** - The complete, consolidated technical specification that combines all aspects of the Project JANUS system into a single comprehensive document.

### What's Inside

The unified document includes all five parts of the JANUS architecture:

1. **Part 1: Main Architecture** - System design, philosophical foundation, and architectural overview
2. **Part 2: Forward Service (Janus Bifrons)** - Real-time decision-making, pattern recognition, and trade execution
3. **Part 3: Backward Service (Janus Consivius)** - Memory consolidation, schema formation, and learning algorithms
4. **Part 4: Neuromorphic Architecture** - Brain-region mapping and biological inspiration
5. **Part 5: Rust Implementation** - Production-ready implementation with deployment guides

## 🚀 Quick Start

### Compile the Document

```bash
cd project_janus
pdflatex janus.tex
pdflatex janus.tex  # Run twice for proper table of contents
```

### Automated CI/CD

The GitHub Actions workflow automatically compiles `janus.tex` (and all other `.tex` files in the repository) on every push to `main`. The generated PDFs are:

- **Uploaded as artifacts** (available for 90 days)
- **Committed back to the repository** (for easy access)

## 📚 Document Structure

```
technical_papers/
├── project_janus/
│   ├── janus.tex          # ← Consolidated document (THIS IS THE ONE)
│   ├── janus.pdf          # ← Auto-generated PDF
│   └── README.md          # ← You are here
├── main.tex               # Individual standalone documents (legacy)
├── forward.tex
├── backward.tex
├── neuro.tex
├── rust.tex
├── main_content.tex       # Content-only file (used by janus.tex)
└── complete.tex           # Old master document (superseded by janus.tex)
```

## 🎯 Key Features

- **Single Source of Truth**: All technical specifications in one place
- **Comprehensive**: ~1000+ lines covering architecture, algorithms, and implementation
- **Production-Ready**: Includes Rust code examples and deployment guides
- **Neuromorphic Design**: Brain-inspired architecture with biological validation
- **Neuro-Symbolic**: Combines deep learning (ViViT, GAF) with logic (LTN)

## 📖 Reading Guide

### For Different Audiences

**Quantitative Researchers:**
1. Part 1 (Main Architecture) - System overview
2. Part 2 (Forward Service) - Trading algorithms
3. Part 3 (Backward Service) - Learning mechanisms

**Machine Learning Engineers:**
1. Part 5 (Rust Implementation) - Tech stack
2. Part 2 (Forward Service) - Model architecture
3. Part 3 (Backward Service) - Training pipeline

**System Architects:**
1. Part 1 (Main Architecture) - System design
2. Part 4 (Neuromorphic Architecture) - Component mapping
3. Part 5 (Rust Implementation) - Deployment strategy

**Neuroscience Enthusiasts:**
1. Part 4 (Neuromorphic Architecture) - Brain mapping
2. Part 1 (Main Architecture) - Philosophical motivation
3. Parts 2 & 3 - Biological implementation

## 🔧 Technical Details

### Dependencies

The document uses standard LaTeX packages:
- `amsmath`, `amssymb` - Mathematical typesetting
- `listings`, `algorithm` - Code formatting
- `tcolorbox` - Colored boxes and highlights
- `hyperref` - PDF bookmarks and links
- `tikz`, `pgfplots` - Diagrams (if needed)

### Colors

The document uses a consistent color scheme:
- **Janus Blue** (`#003366`) - Main architecture
- **Forward Blue** (`#2962FF`) - Real-time service
- **Backward Purple** (`#8A2BE2`) - Learning service
- **Neuro Green** (`#228B22`) - Neuromorphic components
- **Rust Orange** (`#FF8C00`) - Implementation details

## 🤖 Automated Compilation

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:

1. Finds all `.tex` files with `\documentclass` (main documents)
2. Compiles each document twice (for references/TOC)
3. Uploads PDFs as artifacts
4. Commits PDFs back to the repository

This means **you never need to manually compile** - just push your `.tex` changes!

## 📝 Making Changes

To update the documentation:

1. Edit `janus.tex` directly, or
2. Edit `main_content.tex` (which is included by `janus.tex`), or
3. Edit individual `.tex` files and merge changes into `janus.tex`

Then:
```bash
git add .
git commit -m "Update documentation"
git push
```

The CI will automatically compile and commit the PDF.

## 🌟 Why One Consolidated Document?

### Advantages

✅ **Single source of truth** - No version conflicts  
✅ **Easier navigation** - One table of contents  
✅ **Better cross-references** - Links work across parts  
✅ **Simpler deployment** - One PDF to distribute  
✅ **Consistent formatting** - Unified style and colors  

### Legacy Documents

The individual `.tex` files (`main.tex`, `forward.tex`, etc.) are still available for:
- Historical reference
- Standalone compilation
- Modular development

But **`janus.tex` is the recommended master document**.

## 📬 Contact & Repository

- **Author:** Jordan Smith
- **GitHub:** [github.com/nuniesmith/technical_papers](https://github.com/nuniesmith/technical_papers)
- **License:** See repository for details

---

*"The god of beginnings and transitions, looking simultaneously to the future and the past."*