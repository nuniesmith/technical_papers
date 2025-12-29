#!/usr/bin/env bash
###############################################################################
# Project JANUS - All Visualizations Test Suite
#
# Comprehensive test script for all 13 production visualizations.
# Tests syntax, execution, output generation, and basic performance.
#
# Author: Project JANUS Visualization Team
# Date: December 2024
# License: MIT
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
QUICK_MODE=false
VERBOSE=false
OUTPUT_DIR="../outputs/test_$(date +%Y%m%d_%H%M%S)"
DPI=150  # Lower DPI for faster testing

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run quick tests (skip heavy visuals)"
            echo "  --verbose, -v     Show detailed output"
            echo "  --output-dir DIR  Output directory for test results"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

run_test() {
    local test_name=$1
    local command=$2

    print_test "$test_name"

    if [ "$VERBOSE" = true ]; then
        if eval "$command"; then
            print_success "$test_name passed"
            return 0
        else
            print_error "$test_name failed"
            return 1
        fi
    else
        if eval "$command" > /dev/null 2>&1; then
            print_success "$test_name passed"
            return 0
        else
            print_error "$test_name failed"
            return 1
        fi
    fi
}

###############################################################################
# Pre-flight Checks
###############################################################################

print_header "Pre-flight Checks"

# Check Python version
print_test "Python version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9.0"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python $PYTHON_VERSION (>= 3.9 required)"
else
    print_error "Python $PYTHON_VERSION is too old (>= 3.9 required)"
    exit 1
fi

# Check required dependencies
print_test "Core dependencies"
MISSING_DEPS=()

for dep in numpy matplotlib scipy scikit-learn; do
    if ! python3 -c "import ${dep//-/_}" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    print_success "All core dependencies installed"
else
    print_error "Missing dependencies: ${MISSING_DEPS[*]}"
    print_info "Install with: pip install ${MISSING_DEPS[*]}"
    exit 1
fi

# Check optional dependencies
print_test "Optional dependencies"
OPTIONAL_AVAILABLE=()
OPTIONAL_MISSING=()

for dep in "umap:umap" "torch:torch"; do
    IFS=':' read -r pkg_name import_name <<< "$dep"
    if python3 -c "import $import_name" 2>/dev/null; then
        OPTIONAL_AVAILABLE+=("$pkg_name")
    else
        OPTIONAL_MISSING+=("$pkg_name")
    fi
done

if [ ${#OPTIONAL_AVAILABLE[@]} -gt 0 ]; then
    print_info "Optional available: ${OPTIONAL_AVAILABLE[*]}"
fi
if [ ${#OPTIONAL_MISSING[@]} -gt 0 ]; then
    print_info "Optional missing: ${OPTIONAL_MISSING[*]} (will use fallbacks)"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

echo ""

###############################################################################
# Test Suite
###############################################################################

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

test_visual() {
    local visual_id=$1
    local visual_name=$2
    local script_name=$3
    local tier=$4
    local skip_in_quick=$5

    print_header "Visual $visual_id: $visual_name (Tier $tier)"

    # Skip heavy tests in quick mode
    if [ "$QUICK_MODE" = true ] && [ "$skip_in_quick" = true ]; then
        print_info "Skipped (quick mode)"
        ((SKIPPED_TESTS++))
        echo ""
        return
    fi

    # Check if script exists
    if [ ! -f "$script_name" ]; then
        print_error "Script not found: $script_name"
        ((FAILED_TESTS++))
        echo ""
        return
    fi

    # Test 1: Syntax check
    if run_test "Syntax check" "python3 -m py_compile $script_name"; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))

    # Test 2: Help flag
    if run_test "Help flag" "python3 $script_name --help"; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))

    # Test 3: Output generation
    local output_pattern="${OUTPUT_DIR}/visual_${visual_id}_*.png"
    if run_test "Generate output" "python3 $script_name --save-all --output-dir $OUTPUT_DIR --dpi $DPI --seed 42"; then
        ((PASSED_TESTS++))

        # Test 4: Verify output exists
        if ls $output_pattern 1> /dev/null 2>&1; then
            print_success "Output files created"
            ((PASSED_TESTS++))

            # Test 5: Check file size (should be > 10KB)
            local file_count=$(ls $output_pattern 2>/dev/null | wc -l)
            local small_files=0
            for file in $output_pattern; do
                if [ -f "$file" ]; then
                    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
                    if [ "$size" -lt 10240 ]; then
                        ((small_files++))
                    fi
                fi
            done

            if [ "$small_files" -eq 0 ]; then
                print_success "All output files valid (> 10KB)"
                ((PASSED_TESTS++))
            else
                print_error "$small_files file(s) suspiciously small (< 10KB)"
                ((FAILED_TESTS++))
            fi
            ((TOTAL_TESTS++))
        else
            print_error "No output files created"
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    else
        ((FAILED_TESTS++))
        ((TOTAL_TESTS += 2))  # Skip dependent tests
    fi
    ((TOTAL_TESTS++))

    # Test 6: Performance check (Tier 2 should be < 5s in test mode)
    if [ "$tier" = "2" ]; then
        print_test "Performance check (< 5s)"
        START_TIME=$(date +%s)
        python3 "$script_name" --save-all --output-dir "$OUTPUT_DIR" --dpi 100 --seed 42 > /dev/null 2>&1
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        if [ "$ELAPSED" -lt 5 ]; then
            print_success "Performance OK (${ELAPSED}s < 5s)"
            ((PASSED_TESTS++))
        else
            print_error "Performance slow (${ELAPSED}s >= 5s)"
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi

    echo ""
}

###############################################################################
# Run Tests for All 13 Visualizations
###############################################################################

# Phenomenological (Perception)
test_visual "1" "GAF Pipeline" "visual_1_gaf_pipeline.py" "2" false
test_visual "2" "LOB vs GAF Comparison" "visual_2_lob_gaf_comparison.py" "2" false
test_visual "3" "ViViT Factorized Attention" "visual_3_vivit_attention.py" "2" false
test_visual "6" "Multimodal Fusion Gate" "visual_6_fusion_gate.py" "2" false

# Internal-State (Cognition)
test_visual "4" "LTN Grounding Graph" "visual_4_ltn_grounding.py" "4" false
test_visual "5" "Łukasiewicz Truth Surfaces" "visual_5_ltn_truth_surface.py" "4" false
test_visual "7" "OpAL Decision Engine" "visual_7_opal_decision.py" "2" false
test_visual "8" "Mahalanobis Ellipsoid" "visual_8_mahalanobis.py" "2" false
test_visual "9" "Memory Consolidation Cycle" "visual_9_memory_consolidation.py" "3" false
test_visual "10" "Recall Gate Comparator" "visual_10_recall_gate.py" "3" false
test_visual "11" "UMAP Schema Evolution" "visual_11_umap_evolution.py" "3" true

# System (Architecture)
test_visual "12" "Runtime Topology" "visual_12_runtime_topology.py" "4" false
test_visual "13" "Microservices Ecosystem" "visual_13_microservices_ecosystem.py" "4" false

###############################################################################
# Final Report
###############################################################################

print_header "Test Summary"

echo ""
echo "Total Tests:   $TOTAL_TESTS"
echo -e "${GREEN}Passed:        $PASSED_TESTS${NC}"
if [ "$FAILED_TESTS" -gt 0 ]; then
    echo -e "${RED}Failed:        $FAILED_TESTS${NC}"
else
    echo "Failed:        $FAILED_TESTS"
fi
if [ "$SKIPPED_TESTS" -gt 0 ]; then
    echo -e "${YELLOW}Skipped:       $SKIPPED_TESTS${NC}"
fi
echo ""

# Calculate success rate
if [ "$TOTAL_TESTS" -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate:  ${SUCCESS_RATE}%"
else
    echo "Success Rate:  N/A"
fi

echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Generated Files:  $(ls $OUTPUT_DIR/*.png 2>/dev/null | wc -l)"
echo ""

# Exit with error if any tests failed
if [ "$FAILED_TESTS" -gt 0 ]; then
    print_error "Some tests failed!"
    exit 1
else
    print_success "All tests passed!"

    # Show file sizes
    if [ "$VERBOSE" = true ]; then
        echo ""
        print_info "Generated files:"
        ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || true
    fi

    exit 0
fi
