# Simple CPU-based image for MNIST LeNet project
FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but utile per matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs ./configs
COPY src ./src

# Data / artifacts / outputs as volumes (possono essere montati da fuori)
VOLUME ["/app/data", "/app/artifacts", "/app/outputs"]

ENV PYTHONUNBUFFERED=1

# Default: training
CMD ["python", "-m", "src.models.train", "--config", "configs/dev.yaml", "--seed", "42"]
