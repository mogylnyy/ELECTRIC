# app/main.py

import os
import io
import cv2
import numpy as np
import torch
import re
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# === Загрузка переменных окружения
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "digitdetector-unbek/1"

# === Инициализация FastAPI
app = FastAPI()

# === Инициализация моделей
print("🚀 Загрузка моделей...")
model_v5 = torch.hub.load('yolov5', 'custom', path='app/yolov5_model/best.pt', source='local')
ocr = PaddleOCR(det=False, use_angle_cls=False, lang='en')
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)
print("✅ Модели загружены успешно.")

# === Функция затемнения ROI
def darken(image, factor=0.75):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("📥 Получение изображения...")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("❌ Ошибка: не удалось декодировать изображение.")
            return "Fail"

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # === 1. ROI через YOLOv5
        print("🔍 Поиск ROI через YOLOv5...")
        results = model_v5(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()
        if len(boxes) == 0:
            print("❌ Ошибка: YOLOv5 не нашёл ROI.")
            return "Fail"

        x1, y1, x2, y2 = map(int, boxes[0][:4])
        roi = img_bgr[y1:y2, x1:x2]
        print("✅ ROI найден.")

        # === 2. Затемнение ROI
        roi_darker = darken(roi)
        print("🌑 ROI затемнён.")

        # === 3. Roboflow API (детекция цифр)
        print("📡 Отправка ROI в Roboflow...")
        _, img_encoded = cv2.imencode('.jpg', roi_darker)
        response = client.infer(file=img_encoded.tobytes(), model_id=ROBOFLOW_MODEL_ID)
        preds = response.get("predictions", [])
        print("📡 Ответ получен от Roboflow.")

        if preds:
            median_y = np.median([p["y"] for p in preds])
            preds = [p for p in preds if p["y"] >= median_y * 0.9]
        else:
            print("❌ Ошибка: нет предсказаний от Roboflow.")
            return "Fail"

        # === 4. Вырезка и сортировка цифр
        print("✂️ Вырезка и сортировка цифр...")
        digit_imgs = []
        for p in sorted(preds, key=lambda b: b["x"]):
            x, y = int(p["x"]), int(p["y"])
            w, h = int(p["width"] // 2), int(p["height"] // 2)
            x1, y1 = max(x - w, 0), max(y - h, 0)
            x2, y2 = min(x + w, roi.shape[1]), min(y + h, roi.shape[0])
            digit_crop = roi[y1:y2, x1:x2]
            if digit_crop.shape[0] < 10 or digit_crop.shape[1] < 10:
                continue
            resized = cv2.resize(digit_crop, (32, 64))
            digit_imgs.append(resized)

        if not digit_imgs:
            print("❌ Ошибка: нет валидных цифр после вырезки.")
            return "Fail"

        row = cv2.hconcat(digit_imgs)
        print("🧵 Склейка цифр в одну строку завершена.")

        # === 5. PaddleOCR (распознавание строки)
        print("📖 Запуск PaddleOCR...")
        img_rgb_row = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)
        results = ocr.ocr(img_rgb_row, det=False)

        if results and isinstance(results[0], list) and len(results[0]) > 0:
            raw_text = results[0][0][0]
            clean = re.sub(r"[^0-9]", "", raw_text).strip()

            if len(clean) == 8 and clean.startswith("1"):
                clean = clean[1:]

            if clean:
                print(f"✅ PaddleOCR успешно распознал: {clean}")
                return clean
            else:
                print("❌ Ошибка: строка после очистки пуста.")
                return "Fail"
        else:
            print("❌ Ошибка: PaddleOCR не распознал строку.")
            return "Fail"

    except Exception as e:
        print(f"❌ Ошибка в /predict: {e}")
        return "Fail"
