#!/bin/bash
# ReFocused-AI Quick Start Script
# This script provides the fastest way to setup and start training

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   ReFocused-AI Quick Start Script   ${NC}"
echo -e "${GREEN}=====================================${NC}"

# Make this script executable
chmod +x "$0"

# Set working directory to script location
cd "$(dirname "$0")"

# Check if auto_setup.py exists
if [ ! -f "auto_setup.py" ]; then
    echo -e "${YELLOW}auto_setup.py not found. Downloading it...${NC}"
    # Download the script from GitHub if not exists
    curl -s -o auto_setup.py https://raw.githubusercontent.com/Mahdi-196/ReFocused-Ai/main/auto_setup.py
    chmod +x auto_setup.py
fi

# Parse command line arguments
TEST_ONLY=false
FULL_TRAINING=false
NO_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_ONLY=true
            shift
            ;;
        --full)
            FULL_TRAINING=true
            shift
            ;;
        --no-download)
            NO_DOWNLOAD=true
            shift
            ;;
        --help)
            echo "Usage: ./quick_start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test         Run a test training job after setup"
            echo "  --full         Run full training job after setup"
            echo "  --no-download  Skip downloading training data"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command with options
CMD="python3 auto_setup.py"

if [ "$NO_DOWNLOAD" = true ]; then
    CMD="$CMD --no_download"
fi

if [ "$TEST_ONLY" = true ]; then
    CMD="$CMD --test_only"
fi

if [ "$FULL_TRAINING" = true ]; then
    CMD="$CMD --full_training"
fi

# Run the setup script
echo -e "${BLUE}Running: $CMD${NC}"
eval $CMD

# Add environment variables to current session
if [ -f "/home/ubuntu/h100_env.sh" ]; then
    echo -e "${BLUE}Sourcing environment variables...${NC}"
    source /home/ubuntu/h100_env.sh
fi

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   Quick start complete!   ${NC}"
echo -e "${GREEN}=====================================${NC}" 