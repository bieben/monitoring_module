# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create required directories
RUN mkdir -p models
RUN mkdir -p port_mappings

# Copy application files
COPY . .

# Make sure the resource monitoring script is executable
RUN chmod +x show_model_resources.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_SERVICE_PORT_START=8000
ENV MODEL_SERVICE_PORT_END=8100
ENV REGISTRY_PORT=5050
ENV KAFKA_ENABLED=false

# Expose the ports (main service and registry)
EXPOSE 5000 5050 8000-8100

# Run the application
CMD ["python", "app.py"]

# For standalone model service image labels
LABEL org.opencontainers.image.title="Model Service"
LABEL org.opencontainers.image.description="Microservice architecture for AI model deployment with resource monitoring"
LABEL org.opencontainers.image.version="1.1.0"
