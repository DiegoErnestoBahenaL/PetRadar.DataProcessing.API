FROM python:3.13-slim AS base

WORKDIR /app

# Opciones para que el container sea "limpio"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Si tienes requirements.txt, mantenlo en la raíz del repo
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Exponer puerto
EXPOSE 8000

# Ajusta el módulo/paquete según tu app real:
# por ejemplo, si tu FastAPI está en app/main.py con app = FastAPI()
# entonces sería "app.main:app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
