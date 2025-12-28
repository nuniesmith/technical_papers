#!/bin/bash

# Extract content from individual .tex files for complete.tex compilation
# This script strips the preamble and document environment, keeping only the content

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Extracting content from .tex files..."

# Function to extract content between \begin{document} and \end{document}
extract_content() {
    local input_file="$1"
    local output_file="$2"

    if [ ! -f "$input_file" ]; then
        echo "Error: $input_file not found!"
        return 1
    fi

    # Extract content between \begin{document} and \end{document}
    # Also remove the final \end{document} line
    sed -n '/\\begin{document}/,/\\end{document}/p' "$input_file" | \
        sed '1d;$d' > "$output_file"

    echo "✓ Extracted: $output_file"
}

cd "$PROJECT_DIR"

# Extract content from each source file
extract_content "main.tex" "main_content.tex"
extract_content "forward.tex" "forward_content.tex"
extract_content "backward.tex" "backward_content.tex"
extract_content "neuro.tex" "neuro_content.tex"
extract_content "rust.tex" "rust_content.tex"

echo ""
echo "Content extraction complete!"
echo "You can now compile complete.tex"
