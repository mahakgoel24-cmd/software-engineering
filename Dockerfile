FROM python:3.11-slim

# Prevent Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only necessary system deps (keep for qiskit/torch)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install torch separately (FASTER + avoids CUDA issues)
RUN pip install --no-cache-dir --prefer-binary \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project files
COPY . .

# Railway requires dynamic port
EXPOSE 8000

# IMPORTANT: use shell form so $PORT works
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
