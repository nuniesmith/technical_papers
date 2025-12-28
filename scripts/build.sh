#!/bin/bash

# JANUS Master Build Script
# Author: Jordan Smith
# Purpose: Build individual PDFs and complete unified PDF

set -e

# --- Configuration ---
PROJECT_DIR="project_janus"
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Define documents (output PDFs will be: main.pdf, forward.pdf, etc.)
DOCS=(
    "main"
    "forward"
    "backward"
    "neuro"
    "rust"
)

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
for pkg in pmboxdraw newunicodechar algorithm algpseudocode mathtools tcolorbox listings; do
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
            sudo tlmgr install pmboxdraw newunicodechar algorithm2e tcolorbox mathtools listings
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

# Save current directory
ORIGINAL_DIR=$(pwd)

# Change to project directory
cd "$PROJECT_DIR"

# 4. Extract content for complete.tex
echo -e "\n${BLUE}📝 Extracting content from source files...${NC}"
if [ -f "extract_content.sh" ]; then
    chmod +x extract_content.sh
    ./extract_content.sh
else
    echo -e "${YELLOW}⚠ extract_content.sh not found - complete.tex may fail${NC}"
fi

# 5. Compilation
echo -e "\n${BLUE}🔨 Compiling individual PDFs (Dual-Pass)...${NC}"
SUCCESS=0
FAIL=0

for src in "${DOCS[@]}"; do
    if [ ! -f "${src}.tex" ]; then
        echo -e "${RED}  ✗ File not found: ${src}.tex${NC}"
        ((FAIL++))
        continue
    fi

    echo -ne "  Building ${src}.tex → ${src}.pdf... "

    # Run pdflatex twice for TOC/References
    if pdflatex -interaction=nonstopmode "${src}.tex" > /dev/null 2>&1 && \
       pdflatex -interaction=nonstopmode "${src}.tex" > /dev/null 2>&1; then

        if [ -f "${src}.pdf" ]; then
            echo -e "${GREEN}DONE${NC}"
            ((SUCCESS++))
        else
            echo -e "${RED}FAILED (PDF not generated)${NC}"
            ((FAIL++))
        fi
    else
        echo -e "${RED}FAILED (compilation error)${NC}"
        echo -e "${YELLOW}    Check ${src}.log for details${NC}"
        ((FAIL++))
    fi
done

# 6. Build Complete PDF (unified document)
echo -e "\n${BLUE}📦 Building unified complete.pdf...${NC}"

if [ -f "${COMPLETE_DOC}.tex" ]; then
    echo -ne "  Building ${COMPLETE_DOC}.tex → ${COMPLETE_DOC}.pdf... "

    # Check if content files exist
    MISSING_CONTENT=false
    for src in "${DOCS[@]}"; do
        if [ ! -f "${src}_content.tex" ]; then
            echo -e "${YELLOW}SKIPPED${NC}"
            echo -e "${YELLOW}    Missing ${src}_content.tex - run extract_content.sh first${NC}"
            MISSING_CONTENT=true
            break
        fi
    done

    if [ "$MISSING_CONTENT" = false ]; then
        # Run pdflatex three times for complete document (TOC needs extra pass)
        if pdflatex -interaction=nonstopmode "${COMPLETE_DOC}.tex" > /dev/null 2>&1 && \
           pdflatex -interaction=nonstopmode "${COMPLETE_DOC}.tex" > /dev/null 2>&1 && \
           pdflatex -interaction=nonstopmode "${COMPLETE_DOC}.tex" > /dev/null 2>&1; then

            if [ -f "${COMPLETE_DOC}.pdf" ]; then
                echo -e "${GREEN}DONE${NC}"
                ((SUCCESS++))
            else
                echo -e "${RED}FAILED (PDF not generated)${NC}"
                ((FAIL++))
            fi
        else
            echo -e "${RED}FAILED (compilation error)${NC}"
            echo -e "${YELLOW}    Check ${COMPLETE_DOC}.log for details${NC}"
            ((FAIL++))
        fi
    fi
else
    echo -e "${YELLOW}  ⚠ complete.tex not found - skipping unified PDF${NC}"
fi

# 7. Cleanup auxiliary files (keep PDFs)
echo -e "\n${BLUE}🧹 Cleaning up auxiliary files...${NC}"
rm -f *.aux *.log *.out *.toc *.synctex.gz *.lof *.lot *.loa 2>/dev/null
# Remove content extraction files (regenerated each build)
rm -f *_content.tex 2>/dev/null
echo -e "${GREEN}✓ Cleanup complete${NC}"

cd "$ORIGINAL_DIR"

# 8. Summary
echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}Build Complete: $SUCCESS Succeeded${NC} | ${RED}$FAIL Failed${NC}"
echo -e "PDFs are located in: ${GREEN}$(pwd)/$PROJECT_DIR/${NC}"
if [ $SUCCESS -gt 0 ]; then
    ls -lh "$PROJECT_DIR"/*.pdf 2>/dev/null | awk '{print "  • " $9 " (" $5 ")"}'
fi
echo -e "${BLUE}====================================================${NC}"

# 9. Git status check
if [ -d ".git" ]; then
    echo -e "\n${BLUE}📋 Git Status:${NC}"
    git status --short "$PROJECT_DIR/"*.pdf
    echo -e "\n${YELLOW}💡 To commit PDFs, run:${NC}"
    echo -e "   git add $PROJECT_DIR/*.pdf"
    echo -e "   git commit -m 'Update compiled PDFs'"
    echo -e "   git push"
fi
