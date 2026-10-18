# ============================================================
# Sentinel Fraud ML — Dockerfile
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer puerto de la API
EXPOSE 8000

# Por defecto ejecuta la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]