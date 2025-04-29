FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app
COPY . .

# Установка Python-зависимостей вручную (с точной фиксацией numpy)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir git+https://github.com/roboflow/inference.git@main \
 && pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.19.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    opencv-python-headless==4.5.5.62 \
    paddleocr==2.6.1.3 \
    paddlepaddle==2.6.1 \
    requests==2.31.0 \
    python-dotenv==1.0.1 \
    Pillow==9.2.0 \
    pyyaml==6.0.1 \
    matplotlib==3.6.3 \
    seaborn==0.11.2 \
    python-multipart==0.0.9 \
    numpy==1.23.5

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
