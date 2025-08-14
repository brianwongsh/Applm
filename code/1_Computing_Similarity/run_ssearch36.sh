#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Usage ---
# This script requires two arguments:
# 1. The path to the input FASTA file.
# 2. The path to the output file.
#
# Example:
# ./run_ssearch.sh my_proteins.fasta search_results.txt
# --------------------

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    echo "Usage: $0 <FASTA_FILE> <OUTPUT_FILE>"
    exit 1
fi

# Assign command-line arguments to variables
# $1 is the first argument, $2 is the second
FASTA_FILE="$1"
OUTPUT_FILE="$2"

# --- Main Execution ---

# Inform the user what the script is doing
echo "Input FASTA file: $FASTA_FILE"
echo "Output file:      $OUTPUT_FILE"
echo "Running ssearch36..."

# Run the ssearch36 command
# Note the use of "$FASTA_FILE" and "$OUTPUT_FILE" to correctly use the variables
ssearch36 -s BL62 -E 1e+10 -C 10 -T 16 "$FASTA_FILE" "$FASTA_FILE" | grep '>>' -A 3 > "$OUTPUT_FILE"

# Run the Python script to process the output
python misc/trim_ssearch36.py "$OUTPUT_FILE"

echo "Processing complete."