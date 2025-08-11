#!/bin/bash
# Build documentation script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building gym-tl-tools documentation...${NC}"

# Change to docs directory
cd "$(dirname "$0")/docs"

# Clean previous build
echo -e "${BLUE}Cleaning previous build...${NC}"
rm -rf _build/

# Build documentation
echo -e "${BLUE}Building HTML documentation...${NC}"
uv run sphinx-build -b html . _build/html

echo -e "${GREEN}Documentation built successfully!${NC}"
echo -e "${GREEN}Open _build/html/index.html in your browser to view.${NC}"

# Optionally open in browser (uncomment next line)
# python -m webbrowser file://$(pwd)/_build/html/index.html
