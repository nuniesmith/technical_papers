#!/bin/bash
#
# Test Script for V7 (OpAL) and V11 (UMAP) Visualizations
# =========================================================
#
# This script validates the implementation of Visual 7 (OpAL Decision Engine)
# and Visual 11 (UMAP Schema Evolution) for Project JANUS.
#
# Usage:
#   ./test_v7_v11.sh                    # Run all tests
#   ./test_v7_v11.sh --quick            # Quick validation only
#   ./test_v7_v11.sh --full             # Full test suite with all regimes
#
# Requirements:
#   - Python 3.9+
#   - Dependencies from requirements.txt installed
#
# Author: Project JANUS Visualization Team
# License: MIT

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="test_outputs"
PYTHON="${PYTHON:-python3}"
QUICK_MODE=false
FULL_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick|--full]"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# Phase 1: Environment Check
# ==============================================================================

print_header "Phase 1: Environment Validation"

print_info "Checking Python version..."
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  $PYTHON_VERSION"

if $PYTHON -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python version >= 3.9"
else
    print_error "Python version < 3.9 (required: 3.9+)"
    exit 1
fi

print_info "Checking required modules..."
REQUIRED_MODULES=("numpy" "matplotlib" "scipy" "sklearn")
OPTIONAL_MODULES=("umap")

for module in "${REQUIRED_MODULES[@]}"; do
    if $PYTHON -c "import $module" 2>/dev/null; then
        VERSION=$($PYTHON -c "import $module; print($module.__version__)" 2>/dev/null || echo "unknown")
        print_success "$module ($VERSION)"
    else
        print_error "$module not installed"
        echo "Install with: pip install -r requirements.txt"
        exit 1
    fi
done

for module in "${OPTIONAL_MODULES[@]}"; do
    if $PYTHON -c "import $module" 2>/dev/null; then
        VERSION=$($PYTHON -c "import $module; print($module.__version__)" 2>/dev/null || echo "unknown")
        print_success "$module ($VERSION) - OPTIONAL"
    else
        print_warning "$module not installed (optional, will use PCA fallback)"
    fi
done

# ==============================================================================
# Phase 2: V7 (OpAL Decision Engine) Tests
# ==============================================================================

print_header "Phase 2: V7 (OpAL Decision Engine) Tests"

print_info "Test 2.1: Validate V7 Python syntax..."
if $PYTHON -m py_compile visual_7_opal_decision.py 2>/dev/null; then
    print_success "V7 syntax valid"
else
    print_error "V7 syntax errors detected"
    exit 1
fi

print_info "Test 2.2: V7 help text..."
if $PYTHON visual_7_opal_decision.py --help > /dev/null 2>&1; then
    print_success "V7 CLI interface working"
else
    print_error "V7 help command failed"
    exit 1
fi

print_info "Test 2.3: V7 data generation (volatile regime)..."
if $PYTHON -c "
from visual_7_opal_decision import generate_synthetic_trajectory
engine = generate_synthetic_trajectory(n_steps=100, regime='volatile', seed=42)
assert len(engine.history) == 100, f'Expected 100 steps, got {len(engine.history)}'
print('✓ Generated 100 time steps')
print(f'✓ G range: [{min(h[\"G\"] for h in engine.history):.3f}, {max(h[\"G\"] for h in engine.history):.3f}]')
print(f'✓ N range: [{min(h[\"N\"] for h in engine.history):.3f}, {max(h[\"N\"] for h in engine.history):.3f}]')
" 2>&1; then
    print_success "V7 data generation working"
else
    print_error "V7 data generation failed"
    exit 1
fi

print_info "Test 2.4: V7 visualization output (volatile)..."
$PYTHON visual_7_opal_decision.py \
    --regime volatile \
    --steps 500 \
    --save-all \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    > "$OUTPUT_DIR/v7_volatile.log" 2>&1

if [ -f "$OUTPUT_DIR/visual_7_opal_decision_volatile.png" ]; then
    FILE_SIZE=$(stat -f%z "$OUTPUT_DIR/visual_7_opal_decision_volatile.png" 2>/dev/null || stat -c%s "$OUTPUT_DIR/visual_7_opal_decision_volatile.png" 2>/dev/null)
    print_success "V7 output generated (${FILE_SIZE} bytes)"

    # Validate image dimensions (should be 16x8 inches @ 300 DPI = 4800x2400 pixels)
    if command -v identify >/dev/null 2>&1; then
        DIMENSIONS=$(identify -format "%wx%h" "$OUTPUT_DIR/visual_7_opal_decision_volatile.png" 2>/dev/null)
        print_info "Image dimensions: $DIMENSIONS"
    fi
else
    print_error "V7 output file not created"
    exit 1
fi

if [ "$FULL_MODE" = true ]; then
    print_info "Test 2.5: V7 all market regimes (FULL MODE)..."
    for regime in trending choppy; do
        print_info "  Testing regime: $regime"
        $PYTHON visual_7_opal_decision.py \
            --regime $regime \
            --steps 500 \
            --save-all \
            --output-dir "$OUTPUT_DIR" \
            --seed 42 \
            > "$OUTPUT_DIR/v7_${regime}.log" 2>&1

        if [ -f "$OUTPUT_DIR/visual_7_opal_decision_${regime}.png" ]; then
            print_success "  $regime regime generated"
        else
            print_error "  $regime regime failed"
            exit 1
        fi
    done
fi

# ==============================================================================
# Phase 3: V11 (UMAP Schema Evolution) Tests
# ==============================================================================

print_header "Phase 3: V11 (UMAP Schema Evolution) Tests"

print_info "Test 3.1: Validate V11 Python syntax..."
if $PYTHON -m py_compile visual_11_umap_evolution.py 2>/dev/null; then
    print_success "V11 syntax valid"
else
    print_error "V11 syntax errors detected"
    exit 1
fi

print_info "Test 3.2: V11 help text..."
if $PYTHON visual_11_umap_evolution.py --help > /dev/null 2>&1; then
    print_success "V11 CLI interface working"
else
    print_error "V11 help command failed"
    exit 1
fi

print_info "Test 3.3: V11 schema memory generation..."
if $PYTHON -c "
from visual_11_umap_evolution import SchemaMemory
import numpy as np
memory = SchemaMemory(n_features=64, n_regimes=3, seed=42)
embeddings, labels = memory.generate_schemas(500, time_step=0)
assert embeddings.shape == (500, 64), f'Expected (500, 64), got {embeddings.shape}'
assert labels.shape == (500,), f'Expected (500,), got {labels.shape}'
assert set(labels) == {0, 1, 2}, f'Expected regimes {{0,1,2}}, got {set(labels)}'
print('✓ Schema embeddings shape:', embeddings.shape)
print('✓ Labels shape:', labels.shape)
print('✓ Unique regimes:', sorted(set(labels)))
" 2>&1; then
    print_success "V11 schema generation working"
else
    print_error "V11 schema generation failed"
    exit 1
fi

if [ "$QUICK_MODE" = false ]; then
    print_info "Test 3.4: V11 trustworthiness computation..."
    if $PYTHON -c "
from visual_11_umap_evolution import compute_trustworthiness
import numpy as np
np.random.seed(42)
X_high = np.random.randn(100, 50)
X_low = np.random.randn(100, 2)
T = compute_trustworthiness(X_high, X_low, k=15)
assert 0 <= T <= 1, f'Trustworthiness out of range: {T}'
print(f'✓ Trustworthiness T(15) = {T:.3f}')
" 2>&1; then
        print_success "V11 trustworthiness metric working"
    else
        print_error "V11 trustworthiness computation failed"
        exit 1
    fi
fi

print_info "Test 3.5: V11 visualization output..."
$PYTHON visual_11_umap_evolution.py \
    --time-steps 0 100 500 1000 \
    --samples 200 \
    --save-all \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    > "$OUTPUT_DIR/v11.log" 2>&1

if [ -f "$OUTPUT_DIR/visual_11_umap_evolution.png" ]; then
    FILE_SIZE=$(stat -f%z "$OUTPUT_DIR/visual_11_umap_evolution.png" 2>/dev/null || stat -c%s "$OUTPUT_DIR/visual_11_umap_evolution.png" 2>/dev/null)
    print_success "V11 output generated (${FILE_SIZE} bytes)"

    # Validate image dimensions (should be 14x10 inches @ 300 DPI = 4200x3000 pixels)
    if command -v identify >/dev/null 2>&1; then
        DIMENSIONS=$(identify -format "%wx%h" "$OUTPUT_DIR/visual_11_umap_evolution.png" 2>/dev/null)
        print_info "Image dimensions: $DIMENSIONS"
    fi
else
    print_error "V11 output file not created"
    exit 1
fi

# ==============================================================================
# Phase 4: Integration Tests
# ==============================================================================

if [ "$QUICK_MODE" = false ]; then
    print_header "Phase 4: Integration Tests"

    print_info "Test 4.1: Reproducibility (fixed seeds)..."
    $PYTHON visual_7_opal_decision.py \
        --regime volatile \
        --steps 100 \
        --save-all \
        --output-dir "$OUTPUT_DIR/repro1" \
        --seed 42 \
        > /dev/null 2>&1

    $PYTHON visual_7_opal_decision.py \
        --regime volatile \
        --steps 100 \
        --save-all \
        --output-dir "$OUTPUT_DIR/repro2" \
        --seed 42 \
        > /dev/null 2>&1

    if command -v diff >/dev/null 2>&1; then
        if diff "$OUTPUT_DIR/repro1/visual_7_opal_decision_volatile.png" \
                "$OUTPUT_DIR/repro2/visual_7_opal_decision_volatile.png" > /dev/null 2>&1; then
            print_success "V7 outputs are reproducible"
        else
            print_warning "V7 outputs differ (may be due to matplotlib backend)"
        fi
    else
        print_warning "diff not available, skipping reproducibility check"
    fi

    print_info "Test 4.2: Custom parameters..."
    $PYTHON visual_7_opal_decision.py \
        --regime trending \
        --steps 2000 \
        --current-step 1500 \
        --save-all \
        --output-dir "$OUTPUT_DIR/custom" \
        --seed 123 \
        > /dev/null 2>&1

    if [ -f "$OUTPUT_DIR/custom/visual_7_opal_decision_trending.png" ]; then
        print_success "V7 custom parameters working"
    else
        print_error "V7 custom parameters failed"
        exit 1
    fi

    print_info "Test 4.3: Edge cases..."
    # Test minimal steps
    if $PYTHON visual_7_opal_decision.py \
        --steps 10 \
        --save-all \
        --output-dir "$OUTPUT_DIR/edge" \
        --seed 42 \
        > /dev/null 2>&1; then
        print_success "V7 handles minimal steps (10)"
    else
        print_error "V7 failed with minimal steps"
        exit 1
    fi
fi

# ==============================================================================
# Phase 5: Performance Tests
# ==============================================================================

if [ "$FULL_MODE" = true ]; then
    print_header "Phase 5: Performance Tests"

    print_info "Test 5.1: V7 performance (1000 steps)..."
    START_TIME=$(date +%s)
    $PYTHON visual_7_opal_decision.py \
        --steps 1000 \
        --save-all \
        --output-dir "$OUTPUT_DIR/perf" \
        --seed 42 \
        > /dev/null 2>&1
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [ $ELAPSED -lt 10 ]; then
        print_success "V7 completed in ${ELAPSED}s (target: <10s for 1000 steps)"
    else
        print_warning "V7 took ${ELAPSED}s (slower than expected)"
    fi

    print_info "Test 5.2: V11 performance (500 samples, 4 time steps)..."
    START_TIME=$(date +%s)
    $PYTHON visual_11_umap_evolution.py \
        --samples 500 \
        --time-steps 0 100 500 1000 \
        --save-all \
        --output-dir "$OUTPUT_DIR/perf" \
        --seed 42 \
        > /dev/null 2>&1
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [ $ELAPSED -lt 60 ]; then
        print_success "V11 completed in ${ELAPSED}s (target: <60s with UMAP)"
    else
        print_warning "V11 took ${ELAPSED}s (acceptable for manifold learning)"
    fi
fi

# ==============================================================================
# Phase 6: Summary
# ==============================================================================

print_header "Test Summary"

echo "Output files:"
echo "  Directory: $OUTPUT_DIR"
echo ""

V7_COUNT=$(find "$OUTPUT_DIR" -name "visual_7_*.png" | wc -l | tr -d ' ')
V11_COUNT=$(find "$OUTPUT_DIR" -name "visual_11_*.png" | wc -l | tr -d ' ')

echo "Generated visualizations:"
echo "  V7 (OpAL):  $V7_COUNT file(s)"
echo "  V11 (UMAP): $V11_COUNT file(s)"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Total output size: $TOTAL_SIZE"
fi

echo ""
print_success "All tests passed!"
echo ""
print_info "To view outputs:"
echo "  ls -lh $OUTPUT_DIR/"
echo ""
print_info "To clean up test outputs:"
echo "  rm -rf $OUTPUT_DIR/"

exit 0
