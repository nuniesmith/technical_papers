#!/bin/bash
###############################################################################
# Project JANUS - Quick Setup Script
###############################################################################
#
# This script automates the initial setup of the development environment.
#
# Usage:
#   ./setup.sh              # Full setup (venv + dev dependencies)
#   ./setup.sh --minimal    # Minimal setup (venv + core only)
#   ./setup.sh --all        # Full setup with all optional dependencies
#
# Author: Project JANUS Team
# License: MIT
###############################################################################

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
PYTHON_MIN_VERSION="3.9"

# Parse arguments
MODE="dev"
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MODE="minimal"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --help|-h)
            echo "Project JANUS Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal    Install core dependencies only"
            echo "  --all        Install all optional dependencies"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Default: Development setup (core + dev tools)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

###############################################################################
# Functions
###############################################################################

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_python_version() {
    local python_cmd=$1
    local version=$($python_cmd --version 2>&1 | awk '{print $2}')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
        return 0
    else
        return 1
    fi
}

###############################################################################
# Main Setup
###############################################################################

print_header "Project JANUS - Development Environment Setup"

# Step 1: Check Python version
print_info "Checking Python version..."

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v $cmd >/dev/null 2>&1; then
        if check_python_version $cmd; then
            PYTHON_CMD=$cmd
            PYTHON_VERSION=$($cmd --version 2>&1 | awk '{print $2}')
            print_success "Found $cmd (version $PYTHON_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python $PYTHON_MIN_VERSION or higher not found"
    echo ""
    echo "Please install Python $PYTHON_MIN_VERSION or higher:"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  macOS:         brew install python@3.11"
    echo "  Fedora:        sudo dnf install python3.11"
    exit 1
fi

# Step 2: Create virtual environment
print_info "Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        print_info "Removed existing virtual environment"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
else
    print_success "Virtual environment ready"
fi

# Step 3: Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 4: Upgrade pip
print_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel -q
print_success "Package tools upgraded"

# Step 5: Install dependencies based on mode
print_info "Installing dependencies (mode: $MODE)..."

case $MODE in
    minimal)
        print_info "Installing core dependencies only..."
        pip install -e . -q
        print_success "Core dependencies installed"
        ;;
    all)
        print_info "Installing all dependencies (this may take a few minutes)..."
        pip install -e .[all,dev] -q
        print_success "All dependencies installed"
        ;;
    dev)
        print_info "Installing development dependencies..."
        pip install -e .[dev] -q
        print_success "Development dependencies installed"
        ;;
esac

# Step 6: Install pre-commit hooks (if dev mode)
if [ "$MODE" = "dev" ] || [ "$MODE" = "all" ]; then
    print_info "Installing pre-commit hooks..."
    pre-commit install > /dev/null 2>&1
    print_success "Pre-commit hooks installed"
fi

# Step 7: Verify installation
print_info "Verifying installation..."

# Check key packages
MISSING=()
for pkg in numpy matplotlib scipy sklearn; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    print_success "All core packages verified"
else
    print_warning "Some packages failed to import: ${MISSING[*]}"
fi

# Check dev tools (if dev mode)
if [ "$MODE" = "dev" ] || [ "$MODE" = "all" ]; then
    DEV_MISSING=()
    for tool in pytest ruff black isort mypy; do
        if ! command -v $tool >/dev/null 2>&1; then
            DEV_MISSING+=("$tool")
        fi
    done

    if [ ${#DEV_MISSING[@]} -eq 0 ]; then
        print_success "All development tools verified"
    else
        print_warning "Some dev tools not found: ${DEV_MISSING[*]}"
    fi
fi

###############################################################################
# Summary
###############################################################################

print_header "Setup Complete! 🎉"

echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo -e "  Location: ${VENV_DIR}/"
echo -e "  Python:   $($PYTHON_CMD --version)"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "1. ${YELLOW}Activate the virtual environment:${NC}"
    echo -e "   source $VENV_DIR/bin/activate"
    echo ""
fi

echo -e "2. ${YELLOW}Run tests:${NC}"
echo -e "   make test"
echo ""

echo -e "3. ${YELLOW}Generate visualizations:${NC}"
echo -e "   cd project_janus/examples"
echo -e "   python visual_5_ltn_truth_surface.py --save-all"
echo ""

if [ "$MODE" = "dev" ] || [ "$MODE" = "all" ]; then
    echo -e "4. ${YELLOW}Development workflow:${NC}"
    echo -e "   make format    # Format code"
    echo -e "   make lint      # Check code quality"
    echo -e "   make check     # Run all checks"
    echo ""
fi

echo -e "${BLUE}Available Commands:${NC}"
echo -e "  make help      # Show all make targets"
echo -e "  make info      # Show project information"
echo -e "  make test      # Run tests"
echo -e "  make ci        # Simulate CI pipeline"
echo ""

echo -e "${GREEN}Ready for FKS implementation! 🚀${NC}"
echo ""

# Keep virtual environment activated if script was sourced
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not active"
    echo -e "Run: ${YELLOW}source $VENV_DIR/bin/activate${NC}"
fi
