# DEPRECATED: project no longer deployed (GCP free trial expired). Kept for reference.
# This Dockerfile was used to package the FastAPI inference service for Google Cloud Run.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for Pillow / Torch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (torch is pinned in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model checkpoint into the image
COPY . .

# Cloud Run injects PORT (typically 8080)
ENV PORT=8080
EXPOSE 8080

# Bind to 0.0.0.0:$PORT for Cloud Run compatibility
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
