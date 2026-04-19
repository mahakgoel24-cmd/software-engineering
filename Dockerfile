FROM python:3.11-slim

# Prevent Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (required for scipy, qiskit, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first (for caching)
COPY requirements.txt .

# Install EVERYTHING in one go (IMPORTANT)
RUN pip install --no-cache-dir --prefer-binary --upgrade-strategy eager -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Correct Railway startup
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
