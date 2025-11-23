FROM python:3.13-slim AS base

WORKDIR /app

# Opciones para que el container sea "limpio"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema mínimas (ajusta si usas OpenCV, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Exponer puerto
EXPOSE 8000

# Asumiendo que en main.py hay una variable app = FastAPI(...)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
