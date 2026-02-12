# Usamos una versión "slim" de Python para reducir el tamaño de la imagen de ~1GB a ~150MB
FROM python:3.11-slim

# Variables de entorno para optimizar Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalamos dependencias del sistema mínimas para compilación de librerías como scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copiamos solo los requisitos primero para cachear la instalación de librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r -r requirements.txt

# Copiamos el resto del código (incluyendo las carpetas 'modules' y 'data')
COPY . .

# Exponemos el puerto estándar de Streamlit
EXPOSE 8501

# Comando de ejecución con optimizaciones para producción
# Usamos 0.0.0.0 para que sea accesible externamente
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
