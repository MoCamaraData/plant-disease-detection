FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dépendances système pour Pillow / Torch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python (torch inclus si tu l'as mis dans requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code + le modèle dans l'image
COPY . .

# Cloud Run fournit PORT (souvent 8080)
ENV PORT=8080
EXPOSE 8080

# IMPORTANT : écouter sur 0.0.0.0:$PORT (Cloud Run)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
