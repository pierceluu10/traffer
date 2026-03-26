FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir httpx==0.26.0 redis==5.0.0 opencv-python-headless==4.9.0.80
COPY traffic/ traffic/
