# ---------- STAGE 1: BUILD ----------
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps TEMPORARILY
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install dependencies into custom folder
RUN pip install --no-cache-dir --prefer-binary \
    --prefix=/install \
    -r requirements.txt

# ---------- STAGE 2: FINAL ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only installed packages (NO build tools)
COPY --from=builder /install /usr/local

# Copy app
COPY . .

# Remove unnecessary files to shrink image
RUN rm -rf /root/.cache \
    && find /usr/local -type d -name "__pycache__" -exec rm -r {} + \
    && find /usr/local -type d -name "tests" -exec rm -r {} + \
    && find /usr/local -name "*.pyc" -delete

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
