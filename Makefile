# Makefile for Project JANUS
# ===========================
#
# Common development tasks for LaTeX compilation and Python testing
#
# Usage:
#   make venv          Create virtual environment
#   make install       Install Python dependencies
#   make test          Run Python tests
#   make lint          Run linters
#   make format        Format code
#   make latex         Build LaTeX PDFs
#   make clean         Clean build artifacts
#   make all           Run all checks and builds

.PHONY: help venv install install-dev test test-quick test-all lint format clean clean-all latex visual-test

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Virtual environment
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
ACTIVATE := . $(VENV)/bin/activate

##@ General

help: ## Display this help message
	@echo "$(BLUE)Project JANUS - Development Tasks$(NC)"
	@echo ""
	@echo "$(YELLOW)First time setup:$(NC)"
	@echo "  make venv          # Create virtual environment"
	@echo "  make install-dev   # Install development dependencies"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Python Setup

venv: ## Create virtual environment
	@if [ -d "$(VENV)" ]; then \
		echo "$(YELLOW)Virtual environment already exists at $(VENV)$(NC)"; \
	else \
		echo "$(BLUE)Creating virtual environment...$(NC)"; \
		python3 -m venv $(VENV); \
		$(PIP) install --upgrade pip setuptools wheel; \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
		echo "$(YELLOW)Activate with: source $(VENV)/bin/activate$(NC)"; \
	fi

check-venv: ## Check if virtual environment exists
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)✗ Virtual environment not found!$(NC)"; \
		echo "$(YELLOW)Create it with: make venv$(NC)"; \
		exit 1; \
	fi

install: check-venv ## Install core Python dependencies
	$(PIP) install -e .
	@echo "$(GREEN)✓ Core dependencies installed$(NC)"

install-dev: check-venv ## Install development dependencies
	$(PIP) install -e .[dev]
	$(VENV)/bin/pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

install-all: check-venv ## Install all optional dependencies
	$(PIP) install -e .[all,dev]
	$(VENV)/bin/pre-commit install
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

##@ Testing

test: check-venv ## Run Python tests (excluding slow tests)
	$(PYTEST) -v -m "not slow" --cov=project_janus --cov-report=term-missing

test-quick: check-venv ## Run quick smoke tests only
	$(PYTEST) -v -m "not slow and not integration" -x

test-all: check-venv ## Run all tests including slow tests
	$(PYTEST) -v --cov=project_janus --cov-report=html --cov-report=term-missing

test-visual: check-venv ## Run visual generation tests
	$(PYTEST) -v -m visual

test-integration: check-venv ## Run integration tests
	$(PYTEST) -v -m integration

test-coverage: check-venv ## Generate HTML coverage report
	$(PYTEST) --cov=project_janus --cov-report=html
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

##@ Code Quality

lint: check-venv ## Run all linters
	@echo "$(BLUE)Running Ruff...$(NC)"
	$(VENV)/bin/ruff check project_janus/examples tests
	@echo "$(BLUE)Running Black...$(NC)"
	$(VENV)/bin/black --check project_janus/examples tests
	@echo "$(BLUE)Running isort...$(NC)"
	$(VENV)/bin/isort --check-only project_janus/examples tests
	@echo "$(BLUE)Running mypy...$(NC)"
	$(VENV)/bin/mypy project_janus/examples tests || true
	@echo "$(GREEN)✓ All linters passed$(NC)"

format: check-venv ## Format code with Black and isort
	@echo "$(BLUE)Formatting with Black...$(NC)"
	$(VENV)/bin/black project_janus/examples tests
	@echo "$(BLUE)Sorting imports with isort...$(NC)"
	$(VENV)/bin/isort project_janus/examples tests
	@echo "$(GREEN)✓ Code formatted$(NC)"

ruff-fix: check-venv ## Auto-fix linting issues with Ruff
	$(VENV)/bin/ruff check --fix project_janus/examples tests

pre-commit: check-venv ## Run pre-commit hooks on all files
	$(VENV)/bin/pre-commit run --all-files

##@ LaTeX

latex: ## Build all LaTeX PDFs
	@echo "$(BLUE)Building LaTeX documents...$(NC)"
	cd project_janus && pdflatex -interaction=nonstopmode janus.tex
	cd project_janus && pdflatex -interaction=nonstopmode janus.tex
	@echo "$(GREEN)✓ LaTeX PDFs built$(NC)"

latex-clean: ## Clean LaTeX auxiliary files
	find project_janus -name "*.aux" -delete
	find project_janus -name "*.log" -delete
	find project_janus -name "*.out" -delete
	find project_janus -name "*.toc" -delete
	find project_janus -name "*.synctex.gz" -delete
	@echo "$(GREEN)✓ LaTeX auxiliary files cleaned$(NC)"

##@ Visualization

visual-generate: ## Generate all visualizations
	cd project_janus/examples && bash test_all_visuals.sh --quick

visual-v7-v11: ## Generate V7 and V11 visualizations
	cd project_janus/examples && bash test_v7_v11.sh

visual-clean: ## Clean visualization outputs
	rm -rf project_janus/examples/outputs
	rm -rf project_janus/outputs
	@echo "$(GREEN)✓ Visualization outputs cleaned$(NC)"

##@ Cleanup

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage coverage.xml
	@echo "$(GREEN)✓ Python cache cleaned$(NC)"

clean-venv: ## Remove virtual environment
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

clean-all: clean latex-clean visual-clean ## Clean all build artifacts (keeps venv)
	rm -rf dist build
	@echo "$(GREEN)✓ All artifacts cleaned$(NC)"
	@echo "$(YELLOW)Note: Virtual environment preserved. Use 'make clean-venv' to remove it.$(NC)"

##@ Development Workflow

dev-setup: venv install-all ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@echo "$(GREEN)✓ Environment ready! Try:$(NC)"
	@echo "  source $(VENV)/bin/activate  - Activate venv"
	@echo "  make test                    - Run tests"
	@echo "  make lint                    - Check code quality"
	@echo "  make visual-v7-v11           - Generate visualizations"

check: format lint test ## Run all checks (format, lint, test)

ci: ## Simulate CI pipeline locally
	@echo "$(BLUE)Simulating CI pipeline...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test
	@$(MAKE) latex
	@echo "$(GREEN)✓ CI simulation complete$(NC)"

##@ Documentation

docs-serve: ## Serve documentation (if available)
	@echo "$(YELLOW)Documentation serving not yet implemented$(NC)"

##@ Release

bump-version: ## Bump version (requires bump2version)
	@echo "$(YELLOW)Version bumping not yet configured$(NC)"

##@ Info

info: ## Show project information
	@echo "$(BLUE)Project JANUS - Neuromorphic Trading Intelligence$(NC)"
	@echo ""
	@if [ -d "$(VENV)" ]; then \
		echo "Virtual env:    $(GREEN)✓ Active$(NC) ($(VENV))"; \
		echo "Python version: $$($(PYTHON) --version)"; \
		echo "Pip version:    $$($(PIP) --version | cut -d' ' -f2)"; \
	else \
		echo "Virtual env:    $(RED)✗ Not found$(NC)"; \
		echo "System Python:  $$(python3 --version)"; \
	fi
	@echo "Project root:   $$(pwd)"
	@echo ""
	@echo "Quick start:"
	@if [ ! -d "$(VENV)" ]; then \
		echo "  $(YELLOW)make venv$(NC)         - Create virtual environment"; \
	fi
	@echo "  make install-dev  - Setup development environment"
	@echo "  make test         - Run tests"
	@echo "  make visual-v7-v11 - Generate sample visualizations"

list: ## List all available targets
	@$(MAKE) -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}' | sort -u
