FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ ./models/
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

