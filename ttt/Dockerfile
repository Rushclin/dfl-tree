FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le service
COPY node_service.py .

# Exposer le port de l'API Flask
EXPOSE 5000

CMD ["python", "node_service.py"]
