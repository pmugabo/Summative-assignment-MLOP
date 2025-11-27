# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]