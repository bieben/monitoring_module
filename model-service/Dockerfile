# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy application files
COPY . .

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
