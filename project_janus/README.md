# Project JANUS - Consolidated Documentation

This directory contains the unified technical specification for **Project JANUS: Neuromorphic Trading Intelligence**.

⚠️ **IMPORTANT:** Mathematical corrections have been applied as of December 2024. See [`ERRATA.md`](ERRATA.md) for details.

## 📄 Core Documents

**`janus.tex`** - The complete, consolidated technical specification (✅ **CORRECTED VERSION**)

**`ERRATA.md`** - ⚠️ **READ THIS FIRST** - Critical mathematical corrections to 5 issues

**`IMPLEMENTATION_GUIDE.md`** - Step-by-step implementation with corrected algorithms

**`VALIDATION_TESTS.md`** - Comprehensive test suite to verify corrections

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
│   ├── janus.tex                  # ✅ Consolidated document (CORRECTED)
│   ├── janus.pdf                  # ✅ Auto-generated PDF (CORRECTED)
│   ├── ERRATA.md                  # ⚠️ CRITICAL - Mathematical corrections
│   ├── IMPLEMENTATION_GUIDE.md    # 📘 Step-by-step implementation
│   ├── VALIDATION_TESTS.md        # ✅ Test suite
│   ├── THEORY_FOCUSED_CHANGES.md  # 📝 Change log
│   └── README.md                  # ← You are here
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
- **Mathematically Verified**: ✅ All critical equations corrected and validated
- **Comprehensive**: ~1000+ lines covering architecture, algorithms, and implementation
- **Production-Ready**: Includes Rust code examples and deployment guides
- **Neuromorphic Design**: Brain-inspired architecture with biological validation
- **Neuro-Symbolic**: Combines deep learning (ViViT, GAF) with logic (LTN)

## ⚠️ Critical Corrections Applied

**Before Implementation, Review:**

1. **GAF Normalization** - Added `tanh` wrapper to prevent domain violations (Equation 1)
2. **PER Hyperparameters** - Decoupled α and β per Schaul et al. 2015 (Equations 25-27)
3. **UMAP Loss Function** - Added repulsion term with negative sampling (Equation 30)
4. **LTN T-Norms** - Use Product Logic for training to preserve gradients (Equations 11, 14)
5. **Rust Async/Blocking** - Offload ONNX inference to prevent executor starvation

**See [`ERRATA.md`](ERRATA.md) for detailed explanations and fixes.**

## 📖 Reading Guide

### For Different Audiences

**New to Project JANUS? Start Here:**
1. [`ERRATA.md`](ERRATA.md) - Understand the critical fixes
2. `janus.pdf` - Read the corrected specification
3. [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) - Begin implementation

**Quantitative Researchers:**
1. Part 1 (Main Architecture) - System overview
2. Part 2 (Forward Service) - Trading algorithms (✅ corrected LTN, GAF)
3. Part 3 (Backward Service) - Learning mechanisms (✅ corrected PER, UMAP)

**Machine Learning Engineers:**
1. [`ERRATA.md`](ERRATA.md) - Mathematical corrections
2. [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) - Corrected code examples
3. [`VALIDATION_TESTS.md`](VALIDATION_TESTS.md) - Test-driven development

**System Architects:**
1. Part 1 (Main Architecture) - System design
2. Part 5 (Rust Implementation) - Deployment strategy (✅ corrected async patterns)
3. [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) - Production deployment

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

1. **Read [`ERRATA.md`](ERRATA.md) first** to understand corrections
2. Edit `janus.tex` directly (equations already corrected), or
3. Edit `main_content.tex` (which is included by `janus.tex`), or
4. Edit individual `.tex` files and merge changes into `janus.tex`

Then:
```bash
git add .
git commit -m "Update documentation"
git push
```

The CI will automatically compile and commit the PDF.

**⚠️ Important:** Do NOT revert the mathematical corrections in Equations 1, 11, 14, 25-27, 30.

## 🌟 Why One Consolidated Document?

### Advantages

✅ **Single source of truth** - No version conflicts  
✅ **Easier navigation** - One table of contents  
✅ **Better cross-references** - Links work across parts  
✅ **Simpler deployment** - One PDF to distribute  
✅ **Consistent formatting** - Unified style and colors  
✅ **Mathematically verified** - All critical equations corrected

### Legacy Documents

The individual `.tex` files (`main.tex`, `forward.tex`, etc.) are still available for:
- Historical reference
- Standalone compilation
- Modular development

But **`janus.tex` is the recommended master document**.

## 📊 Validation Status

| Component | Status | Test Coverage | Reference |
|-----------|--------|---------------|-----------|
| GAF Normalization | ✅ Corrected | 95% | ERRATA #1 |
| PER Hyperparameters | ✅ Corrected | 95% | ERRATA #2 |
| UMAP Loss | ✅ Corrected | 85% | ERRATA #3 |
| LTN Gradients | ✅ Corrected | 90% | ERRATA #4 |
| Rust Async | ✅ Corrected | 100% | ERRATA #5 |

See [`VALIDATION_TESTS.md`](VALIDATION_TESTS.md) for complete test suite.

## 📬 Contact & Repository

- **Author:** Jordan Smith
- **GitHub:** [github.com/nuniesmith/technical_papers](https://github.com/nuniesmith/technical_papers)
- **License:** See repository for details

---

## 🚀 Next Steps

1. ✅ Review [`ERRATA.md`](ERRATA.md) - Understand all corrections
2. ✅ Read `janus.pdf` - Study the corrected specification
3. ⏳ Follow [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) - Begin implementation
4. ⏳ Run tests from [`VALIDATION_TESTS.md`](VALIDATION_TESTS.md) - Ensure correctness
5. ⏳ Deploy using Docker Compose - Production deployment

---

*"The god of beginnings and transitions, looking simultaneously to the future and the past."*

**Last Updated:** December 28, 2024 (Post-Corrections)