#!/bin/bash

# Directory to store output files
OUTPUT_DIR="/app/output"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the Python simulation script
python /app/monte_carlo_simulation.py

# Output completion message
echo "Simulation completed. Results are stored in the output directory."
