#!/bin/bash
# AutoG script for automatic graph schema generation
# Usage: bash scripts/autog.sh [dataset]
# Run all datasets if no argument provided

set -e  # Exit on error

# Color output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}AutoG: Automatic Graph Schema Generation${NC}"
echo ""

# Default settings
SCHEMA_PATH="datasets"
METHOD="autog-s"  # AutoG-Selective (final schema)
SEED=0

# Function to run AutoG for a dataset
run_autog() {
    local dataset=$1
    local task=$2

    echo -e "${GREEN}Running AutoG for ${dataset} (task: ${task})${NC}"
    python3 -m main.autog \
        "$dataset" \
        "$SCHEMA_PATH" \
        "$METHOD" \
        "$task" \
        --types-file type.txt \
        --seed "$SEED"

    echo ""
}

# Run specific dataset or all
case "${1:-all}" in
    mag)
        run_autog "mag" "venue"
        ;;
    movielens)
        run_autog "movielens" "ratings"
        ;;
    ieeecis)
        run_autog "ieeecis" "fraud"
        ;;
    avs)
        run_autog "avs" "repeater"
        ;;
    diginetica)
        run_autog "diginetica" "purchase"
        ;;
    retailrocket)
        run_autog "retailrocket" "cvr"
        ;;
    stackexchange)
        run_autog "stackexchange" "churn"
        ;;
    all)
        echo "Running AutoG for all datasets..."
        echo ""
        run_autog "mag" "venue"
        run_autog "movielens" "ratings"
        run_autog "ieeecis" "fraud"
        run_autog "avs" "repeater"
        run_autog "diginetica" "purchase"
        run_autog "retailrocket" "cvr"
        run_autog "stackexchange" "churn"
        ;;
    *)
        echo "Usage: bash scripts/autog.sh [dataset]"
        echo ""
        echo "Available datasets:"
        echo "  mag         - MAG venue prediction"
        echo "  movielens   - MovieLens ratings"
        echo "  ieeecis    - IEEE-CIS fraud detection"
        echo "  avs         - AVS repeater prediction"
        echo "  diginetica  - Diginetica purchase prediction"
        echo "  retailrocket - RetailRocket CVR prediction"
        echo "  stackexchange - StackExchange churn prediction"
        echo "  all         - Run all datasets (default)"
        exit 1
        ;;
esac

echo -e "${GREEN}AutoG completed successfully!${NC}"
echo ""
echo "Note: If you see issues, try deleting the round_0 directory from the output."
