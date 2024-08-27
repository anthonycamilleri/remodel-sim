#!/bin/bash

# Define the image name
IMAGE_NAME="monte-carlo-simulation"

# Define the output directory on the host machine
HOST_OUTPUT_DIR="C:/SimResults/output"

# Ensure the output directory exists on the host machine
mkdir -p "$HOST_OUTPUT_DIR"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container with volume mapping for output
docker run --rm -it -v "$HOST_OUTPUT_DIR:/app/output" $IMAGE_NAME
