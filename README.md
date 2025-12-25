# Project JANUS - Neuromorphic Trading Intelligence

[![JANUS Unified CI](https://github.com/nuniesmith/technical_papers/actions/workflows/ci.yml/badge.svg)](https://github.com/nuniesmith/technical_papers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nuniesmith/technical_papers/branch/main/graph/badge.svg)](https://codecov.io/gh/nuniesmith/technical_papers)

> *A Brain-Inspired Architecture for Autonomous Financial Systems*

## 📖 Overview

Project JANUS is a comprehensive neuromorphic trading intelligence system that bridges neuroscience, machine learning, and quantitative finance. The system features a dual-architecture design:

- **Janus Bifrons (Forward Service)**: Real-time trading, pattern recognition, and execution
- **Janus Consivius (Backward Service)**: Memory consolidation, schema formation, and learning

## 📚 Documentation Suite

This repository contains five interconnected technical documents that can be read independently or as a unified whole:

### Individual Documents

1. **📘 JANUS Main Architecture** (`janus_main.pdf`)
   - Philosophical foundation and system design
   - Architectural overview and component integration
   - Safety, compliance, and validation strategies

2. **🔵 JANUS Forward Service** (`janus_forward.pdf`)
   - Real-time decision-making system
   - DiffGAF visual pattern recognition
   - Logic Tensor Networks for constraint enforcement
   - Basal Ganglia-inspired decision engine

3. **🟣 JANUS Backward Service** (`janus_backward.pdf`)
   - Three-timescale memory hierarchy
   - Sharp-Wave Ripple (SWR) simulation
   - Schema consolidation and long-term learning
   - UMAP-based cognitive visualization

4. **🟢 JANUS Neuromorphic Architecture** (`janus_neuromorphic_architecture.pdf`)
   - Brain-region to trading-component mapping
   - Neuroscience-inspired design patterns
   - Information flow diagrams

5. **🟠 JANUS Rust Implementation** (`janus_rust_implementation.pdf`)
   - Production-ready ML system with Rust
   - FastAPI gateway architecture
   - Docker/Kubernetes deployment guide
   - Migration roadmap from Python to Rust

### Complete Edition

**📕 JANUS Complete** (`complete.pdf`)
- All five volumes combined into a single comprehensive document
- Unified table of contents and cross-references
- Master implementation checklist
- Consolidated bibliography

## 🏗️ Repository Structure

```
technical_papers/
├── .github/
│   └── workflows/
│       └── build-pdf.yml          # Automated PDF generation
├── project_janus/
│   ├── main.tex                   # Volume I: Architecture
│   ├── forward.tex                # Volume II: Forward Service
│   ├── backward.tex               # Volume III: Backward Service
│   ├── neuro.tex                  # Volume IV: Neuromorphic Architecture
│   ├── rust.tex                   # Volume V: Rust Implementation
│   └── complete.tex         # Combined master document
├── scripts/
│   └── build.sh                   # One-click build script
├── pdf/                           # Generated PDFs (auto-committed)
│   ├── janus_main.pdf
│   ├── janus_forward.pdf
│   ├── janus_backward.pdf
│   ├── janus_neuromorphic_architecture.pdf
│   ├── janus_rust_implementation.pdf
│   └── complete.pdf
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Download Pre-Built PDFs

The latest PDFs are automatically generated and available in the `pdf/` directory:

```bash
git clone https://github.com/nuniesmith/technical_papers.git
cd technical_papers/pdf
# Open any PDF
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

**macOS:**
```bash
brew install --cask mactex
```

**Arch Linux:**
```bash
sudo pacman -S texlive-most
```

#### Build All Documents

```bash
cd technical_papers
chmod +x scripts/build.sh
./scripts/build.sh
```

This will:
1. ✅ Check for required LaTeX packages
2. 🔧 Auto-install missing dependencies (if possible)
3. 📄 Compile all 6 PDFs (5 individual + 1 complete)
4. 🧹 Clean up auxiliary files
5. 📊 Display build summary

Generated PDFs will be in the `pdf/` directory.

#### Build Individual Documents

```bash
cd technical_papers/project_janus

# Build main architecture
pdflatex -interaction=nonstopmode -jobname=janus_main main.tex

# Build forward service
pdflatex -interaction=nonstopmode -jobname=janus_forward forward.tex

# Build backward service
pdflatex -interaction=nonstopmode -jobname=janus_backward backward.tex

# Build neuromorphic architecture
pdflatex -interaction=nonstopmode -jobname=janus_neuromorphic_architecture neuro.tex

# Build Rust implementation
pdflatex -interaction=nonstopmode -jobname=janus_rust_implementation rust.tex

# Build complete edition
pdflatex -interaction=nonstopmode complete.tex
```

**Note:** Run `pdflatex` twice for each document to properly generate table of contents and cross-references.

## 🤖 Automated CI/CD (GitHub Actions)

Every push to the `main` branch automatically:

**Documentation Pipeline:**
1. Runs quality checks on LaTeX files
2. Compiles all 6 PDFs
3. Commits them back to the repository
4. Generates documentation metrics

**Rust Code Pipeline** (when Rust code exists):
1. Runs tests on Linux, macOS, and Windows
2. Generates code coverage reports (Codecov)
3. Runs security audits and benchmarks
4. Builds release artifacts

View the workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

**Setup Guide:** See [`CI_SETUP.md`](CI_SETUP.md) for Codecov configuration and detailed CI documentation.

## 📋 Implementation Roadmap

The complete implementation can be broken down into phases:

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1** | Weeks 1-4 | Infrastructure setup (Rust workspace, Docker, databases) |
| **Phase 2** | Weeks 5-8 | Forward Service (DiffGAF, LTN, decision engine) |
| **Phase 3** | Weeks 9-12 | Backward Service (memory hierarchy, SWR simulation) |
| **Phase 4** | Weeks 13-14 | Training Gateway (FastAPI, Celery, ONNX export) |
| **Phase 5** | Weeks 15-16 | Integration & Testing (backtesting, benchmarks) |
| **Phase 6** | Weeks 17-20 | Production Deployment (Kubernetes, monitoring) |
| **Phase 7** | Ongoing | Optimization (profiling, pure Rust ML) |

See the **Master Implementation Checklist** in `complete.pdf` for detailed tasks.

## 🧠 Key Technologies

### Machine Learning
- **PyTorch**: Model training
- **ONNX Runtime**: Rust-based inference
- **Logic Tensor Networks**: Constraint enforcement
- **Vision Transformers (ViViT)**: Spatiotemporal pattern recognition
- **UMAP**: Dimensionality reduction and visualization

### System Architecture
- **Rust**: High-performance services (Forward/Backward)
- **FastAPI**: Python gateway for training
- **Tokio**: Async runtime
- **Qdrant**: Vector database for schemas
- **Redis**: Caching layer
- **PostgreSQL**: System state

### Deployment
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus + Grafana**: Monitoring
- **GitHub Actions**: CI/CD

## 🎯 Design Principles

1. **Neuromorphic Architecture**: Every component maps to a brain region
2. **Dual-Process Design**: Separate "wake" (trading) and "sleep" (learning) states
3. **Rust-First ML**: Production inference in Rust, training in Python
4. **Explainability**: Glass-box design with full auditability
5. **Safety by Design**: Circuit breakers, kill switches, and compliance constraints

## 📊 Performance Targets

| Metric | Target | Component |
|--------|--------|-----------|
| Inference Latency | < 10ms | Forward Service |
| Throughput | > 1000 req/s | Forward Service |
| Memory Footprint | < 4GB | Forward Service (hot path) |
| Batch Processing | 100K episodes | Backward Service (sleep cycle) |
| Model Update Frequency | Every 24h | Training Gateway |

## 🔒 Safety & Compliance

- **Wash Sale Detection**: Automated constraint enforcement
- **Position Limits**: Hard-coded safety bounds
- **Circuit Breakers**: Multi-level kill switches
- **Audit Trail**: Complete decision logging
- **Regulatory Compliance**: Built-in compliance checks

## 📖 Reading Guide

### For Quantitative Researchers
1. Start with `janus_main.pdf` (architecture overview)
2. Read `janus_forward.pdf` (trading algorithms)
3. Review `janus_backward.pdf` (learning system)

### For ML Engineers
1. Begin with `janus_rust_implementation.pdf` (tech stack)
2. Study `janus_forward.pdf` (model architecture)
3. Explore `janus_backward.pdf` (training pipeline)

### For System Architects
1. Review `janus_main.pdf` (system design)
2. Read `janus_neuromorphic_architecture.pdf` (component mapping)
3. Study `janus_rust_implementation.pdf` (deployment)

### For Neuroscience Enthusiasts
1. Start with `janus_neuromorphic_architecture.pdf` (brain mapping)
2. Read `janus_main.pdf` (philosophical motivation)

## 🤝 Contributing

This is a technical specification repository. For contributions:

1. **Documentation Improvements**: Submit PRs with LaTeX changes
2. **Typo Fixes**: Open issues or submit quick PRs
3. **Implementation Questions**: Open issues for discussion

## 📄 License

Copyright © 2025 Jordan Smith

This documentation is released under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The actual implementation code (when released) will use MIT License.

## 👤 Author

**Jordan Smith**
- GitHub: [@nuniesmith](https://github.com/nuniesmith)
- Email: jordan@example.com

## 🙏 Acknowledgments

This work builds upon decades of research in:
- Neuroscience (György Buzsáki, Demis Hassabis)
- Reinforcement Learning (DeepMind, OpenAI)
- Quantitative Finance (Robert Almgren, Neil Chriss)
- Systems Programming (Rust community)

## 📚 References

Key papers and books that influenced this architecture:

1. Buzsáki, G. (2015). *The Brain from Inside Out*. Oxford University Press.
2. Hassabis, D., et al. (2017). Neuroscience-inspired artificial intelligence. *Neuron*, 95(2), 245-258.
3. Badreddine, S., et al. (2022). Logic tensor networks. *Artificial Intelligence*, 303, 103649.
4. McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
5. Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5-40.

---

*"The god of beginnings and transitions, looking simultaneously to the future and the past."*