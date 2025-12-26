#!/bin/bash

# JANUS Master Build Script
# Author: Jordan Smith | Date: 2025-12-25
# Purpose: One-click installation and compilation of the JANUS documentation suite.

set -e

# --- Configuration ---
PROJECT_DIR="project_janus"
PDF_DIR="pdf"
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Define documents with their source names and output names
declare -A DOCS=(
    ["main"]="janus_main"
    ["forward"]="janus_forward"
    ["backward"]="janus_backward"
    ["neuro"]="janus_neuromorphic_architecture"
    ["rust"]="janus_rust_implementation"
)

# Complete document that merges all 5 PDFs
COMPLETE_DOC="complete"

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   JANUS Documentation: Automated Build System      ${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. Environment Check
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}✗ Error: pdflatex not found.${NC}"
    echo "Please install TeX Live (e.g., sudo apt install texlive-full) and try again."
    exit 1
fi

# 2. Dependency Management
echo -e "\n${BLUE}📦 Checking LaTeX dependencies...${NC}"
MISSING_PKGS=false
for pkg in pmboxdraw newunicodechar algorithm algpseudocode mathtools tcolorbox; do
    if ! kpsewhich "$pkg.sty" &> /dev/null; then
        echo -e "${YELLOW}  ! Missing package: $pkg${NC}"
        MISSING_PKGS=true
    fi
done

if [ "$MISSING_PKGS" = true ]; then
    echo -e "${BLUE}📥 Attempting to install missing packages...${NC}"
    if [ -f /etc/debian_version ]; then
        sudo apt-get update -qq && sudo apt-get install -y -qq texlive-fonts-extra texlive-latex-extra texlive-science
    elif [ -f /etc/fedora-release ]; then
        sudo dnf install -y texlive-newunicodechar texlive-pmboxdraw texlive-algorithm2e texlive-tcolorbox
    elif [ -f /etc/arch-release ]; then
        sudo pacman -S --noconfirm texlive-fontsextra texlive-latexextra texlive-science
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v tlmgr &> /dev/null; then
            sudo tlmgr install pmboxdraw newunicodechar algorithm2e tcolorbox mathtools
        else
            echo -e "${RED}! macOS detected without tlmgr. Please install MacTeX/Homebrew-Cask.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ OS not recognized. Continuing in fallback mode...${NC}"
    fi
else
    echo -e "${GREEN}✓ All dependencies met.${NC}"
fi

# 3. Check project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}✗ Error: $PROJECT_DIR directory not found!${NC}"
    exit 1
fi

# 4. Compilation
echo -e "\n${BLUE}🔨 Compiling JANUS Suite (Dual-Pass)...${NC}"
mkdir -p "$PDF_DIR"
SUCCESS=0
FAIL=0

# Save current directory
ORIGINAL_DIR=$(pwd)

# Change to project directory
cd "$PROJECT_DIR"

for src in "${!DOCS[@]}"; do
    output="${DOCS[$src]}"

    if [ ! -f "${src}.tex" ]; then
        echo -e "${RED}  ✗ File not found: ${src}.tex${NC}"
        ((FAIL++))
        continue
    fi

    echo -ne "  Building ${src}.tex → ${output}.pdf... "

    # Run pdflatex twice for TOC/References
    if pdflatex -interaction=nonstopmode -jobname="${output}" "${src}.tex" > /dev/null 2>&1 && \
       pdflatex -interaction=nonstopmode -jobname="${output}" "${src}.tex" > /dev/null 2>&1; then

        if [ -f "${output}.pdf" ]; then
            mv "${output}.pdf" "../${PDF_DIR}/${output}.pdf"
            echo -e "${GREEN}DONE${NC}"
            ((SUCCESS++))
        else
            echo -e "${RED}FAILED (PDF not generated)${NC}"
            ((FAIL++))
        fi
    else
        echo -e "${RED}FAILED (compilation error)${NC}"
        echo -e "${YELLOW}    Check ${output}.log for details${NC}"
        ((FAIL++))
    fi
done

# Return to original directory
cd "$ORIGINAL_DIR"

# 4.5. Build Complete PDF (requires individual PDFs to exist)
echo -e "\n${BLUE}📦 Building complete documentation (all 5 PDFs merged)...${NC}"
cd "$PROJECT_DIR"

if [ -f "${COMPLETE_DOC}.tex" ]; then
    echo -ne "  Building ${COMPLETE_DOC}.tex → janus_complete.pdf... "

    # Check if all required PDFs exist
    MISSING_PDF=false
    for src in "${!DOCS[@]}"; do
        if [ ! -f "${src}.pdf" ]; then
            echo -e "${YELLOW}SKIPPED${NC}"
            echo -e "${YELLOW}    Missing ${src}.pdf - compile individual PDFs first${NC}"
            MISSING_PDF=true
            break
        fi
    done

    if [ "$MISSING_PDF" = false ]; then
        # Run pdflatex twice for complete document
        if pdflatex -interaction=nonstopmode -jobname="janus_complete" "${COMPLETE_DOC}.tex" > /dev/null 2>&1 && \
           pdflatex -interaction=nonstopmode -jobname="janus_complete" "${COMPLETE_DOC}.tex" > /dev/null 2>&1; then

            if [ -f "janus_complete.pdf" ]; then
                mv "janus_complete.pdf" "../${PDF_DIR}/janus_complete.pdf"
                echo -e "${GREEN}DONE${NC}"
                ((SUCCESS++))
            else
                echo -e "${RED}FAILED (PDF not generated)${NC}"
                ((FAIL++))
            fi
        else
            echo -e "${RED}FAILED (compilation error)${NC}"
            echo -e "${YELLOW}    Check janus_complete.log for details${NC}"
            ((FAIL++))
        fi
    fi
else
    echo -e "${YELLOW}  ⚠ complete.tex not found - skipping complete PDF${NC}"
fi

cd "$ORIGINAL_DIR"

# 5. Cleanup
echo -e "\n${BLUE}🧹 Cleaning up auxiliary files...${NC}"
cd "$PROJECT_DIR"
rm -f *.aux *.log *.out *.toc *.synctex.gz 2>/dev/null
# Keep individual PDFs for complete.tex to reference
cd "$ORIGINAL_DIR"

# 6. Summary
echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}Build Complete: $SUCCESS Succeeded${NC} | ${RED}$FAIL Failed${NC}"
echo -e "PDFs are located in: ${GREEN}$(pwd)/$PDF_DIR/${NC}"
if [ $SUCCESS -gt 0 ]; then
    ls -lh "$PDF_DIR"/*.pdf 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'
fi
echo -e "${BLUE}====================================================${NC}"

# 7. Git status check
if [ -d ".git" ]; then
    echo -e "\n${BLUE}📋 Git Status:${NC}"
    git status --short "$PDF_DIR/"
    echo -e "\n${YELLOW}💡 To commit PDFs, run:${NC}"
    echo -e "   git add $PDF_DIR/*.pdf"
    echo -e "   git commit -m 'Update compiled PDFs'"
    echo -e "   git push"
fi
