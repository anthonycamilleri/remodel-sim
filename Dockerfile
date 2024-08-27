# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary Python packages first (dependencies layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and other necessary files into the Docker image
COPY monte_carlo_simulation.py config.csv /app/

# Copy the Bash script to run the simulation
COPY run_simulation.sh /app/
RUN chmod +x /app/run_simulation.sh

# Command to run the Bash script when the container starts
CMD ["/app/run_simulation.sh"]
