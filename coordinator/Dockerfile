FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zip \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Copy application files and install Python dependencies
COPY . /app/
RUN pip install --no-cache-dir rpyc

CMD ["python3", "-u", "coordinator.py"]
