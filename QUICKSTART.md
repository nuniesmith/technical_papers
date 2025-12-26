# Quick Start Guide: Building the Complete JANUS Documentation

This guide shows you how to build `complete.pdf` - a single PDF containing all 5 Project JANUS technical documents.

---

## Prerequisites

You need a LaTeX distribution installed:

- **Linux (Ubuntu/Debian):** `sudo apt-get install texlive-full`
- **macOS:** Install [MacTeX](https://www.tug.org/mactex/)
- **Windows:** Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

---

## Method 1: Automated Build (Recommended) ⚡

Use the build script to compile everything automatically:

```bash
cd technical_papers
chmod +x scripts/build.sh
./scripts/build.sh
```

**Output:** All PDFs will be in the `pdf/` directory:
- `janus_main.pdf`
- `janus_forward.pdf`
- `janus_backward.pdf`
- `janus_neuromorphic_architecture.pdf`
- `janus_rust_implementation.pdf`
- **`janus_complete.pdf`** ← All 5 documents in one file!

---

## Method 2: Manual Build 🔧

If you prefer to build manually or the script doesn't work:

### Step 1: Compile Individual PDFs

```bash
cd technical_papers/project_janus

# Build each document (run twice for TOC/references)
for doc in main forward backward neuro rust; do
    pdflatex -interaction=nonstopmode $doc.tex
    pdflatex -interaction=nonstopmode $doc.tex
done
```

### Step 2: Build Complete PDF

```bash
# Now compile complete.tex (which merges the PDFs)
pdflatex -interaction=nonstopmode complete.tex
pdflatex -interaction=nonstopmode complete.tex
```

### Step 3: Clean Up (Optional)

```bash
rm -f *.aux *.log *.out *.toc *.synctex.gz
```

---

## Method 3: CI Pipeline (Automatic) 🤖

Every push to the `main` branch automatically:

1. ✅ Compiles all 5 individual PDFs
2. ✅ Compiles `complete.pdf` (all 5 merged)
3. ✅ Commits PDFs back to repository
4. ✅ Creates downloadable artifacts

**Just push your changes and CI does the rest!**

```bash
git add project_janus/*.tex
git commit -m "Update documentation"
git push
```

Wait a few minutes and check the repository - all PDFs will be updated automatically.

---

## Understanding `complete.tex`

The `complete.tex` file creates a unified document by:

1. **Title Page:** Professional cover with color-coded table of contents
2. **Reading Guides:** Tailored paths for different audiences
3. **PDF Inclusion:** Uses `\includepdf` to merge all 5 documents
4. **Navigation:** PDF bookmarks for easy jumping between parts

**Key Feature:** Each part preserves its original formatting, TOC, and page numbers from the individual PDFs.

---

## Troubleshooting

### Problem: "pdflatex: command not found"

**Solution:** Install LaTeX (see Prerequisites above)

### Problem: "File 'main.pdf' not found"

**Solution:** You must compile the individual PDFs first before building `complete.pdf`

```bash
cd project_janus
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# ... repeat for forward, backward, neuro, rust ...
# THEN compile complete.tex
```

### Problem: "Package pdfpages not found"

**Solution:** Install the missing package:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra

# macOS (if tlmgr is available)
sudo tlmgr install pdfpages

# Windows (MiKTeX)
# Open MiKTeX Console > Packages > Search "pdfpages" > Install
```

### Problem: Build script fails on Windows

**Solution:** Use WSL (Windows Subsystem for Linux) or Git Bash, or compile manually with the commands above

---

## File Sizes (Approximate)

| Document | Pages | Size |
|----------|-------|------|
| Main Architecture | ~45 | ~230 KB |
| Forward Service | ~50 | ~260 KB |
| Backward Service | ~55 | ~287 KB |
| Neuromorphic Architecture | ~40 | ~221 KB |
| Rust Implementation | ~60 | ~268 KB |
| **Complete Suite** | **~250** | **~1.3 MB** |

---

## What's Inside `complete.pdf`?

```
┌─────────────────────────────────────┐
│ Title Page                          │
│ Reading Guides for Different Roles  │
├─────────────────────────────────────┤
│ Part 1: Main Architecture           │
│   • Philosophical Foundation        │
│   • System Design                   │
│   • Integration Strategy            │
├─────────────────────────────────────┤
│ Part 2: Forward Service             │
│   • DiffGAF Visual Encoding         │
│   • Logic Tensor Networks           │
│   • Basal Ganglia Decision Engine   │
├─────────────────────────────────────┤
│ Part 3: Backward Service            │
│   • Memory Hierarchy                │
│   • Sharp-Wave Ripple Simulation    │
│   • Schema Consolidation            │
├─────────────────────────────────────┤
│ Part 4: Neuromorphic Architecture   │
│   • Brain-Region Mapping            │
│   • Component Design Patterns       │
│   • Information Flow                │
├─────────────────────────────────────┤
│ Part 5: Rust Implementation         │
│   • Production Architecture         │
│   • Deployment Guide                │
│   • Implementation Roadmap          │
└─────────────────────────────────────┘
```

---

## Next Steps

1. ✅ Build `complete.pdf` using one of the methods above
2. ✅ Review the unified documentation
3. ✅ Share with your team or collaborators
4. ✅ Use as a comprehensive reference during implementation

---

## Questions?

- **Repository:** https://github.com/nuniesmith/technical_papers
- **Issues:** Open a GitHub issue if you encounter problems
- **Individual PDFs:** Each document is also available separately in `project_janus/` or `pdf/`

---

**Happy Reading! 📚**

*"The god of beginnings and transitions, looking simultaneously to the future and the past."*