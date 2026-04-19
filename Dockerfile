FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

# install torch separately (fast path)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# install rest
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# copy app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
