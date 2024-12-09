FROM python:3.9-slim

WORKDIR /app

COPY node_service.py .
# RUN pip install paho-mqtt
RUN pip install "paho-mqtt<2.0.0"


CMD ["python", "node_service.py"]
