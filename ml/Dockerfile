FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

RUN pip install --no-cache-dir \
    ultralytics \
    flask

COPY yolo_backend/model.py /app/model.py

ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MODEL_DIR=/app/models
ENV YOLO_MODEL=best.pt
ENV FORCE_CPU=true

EXPOSE 9090

CMD ["python", "/app/model.py"]