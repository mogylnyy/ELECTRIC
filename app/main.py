import os
import cv2
import numpy as np
import torch
import re
import base64
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# === Загрузка переменных окружения
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "digitdetector-unbek/1"

app = FastAPI()

print("🚀 Загрузка моделей...")
model_v5 = torch.hub.load('yolov5', 'custom', path='app/yolov5_model/best.pt', source='local')
ocr = PaddleOCR(
    det=False,
    use_angle_cls=False,
    lang='en',
    rec_model_dir=None  # <=== Важно: загружает heavy (PP-OCRv3)
)
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)
print("✅ Модели загружены успешно.")


def darken(image, factor=0.75):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"result": "Fail"}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = model_v5(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()
        if len(boxes) == 0:
            return {"result": "Fail"}

        x1, y1, x2, y2 = map(int, boxes[0][:4])
        roi = img_bgr[y1:y2, x1:x2]
        roi_darker = darken(roi)

        tmp_filename = "tmp.jpg"
        cv2.imwrite(tmp_filename, roi_darker)
        response = client.infer(tmp_filename, model_id=ROBOFLOW_MODEL_ID)
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        preds = response.get("predictions", [])
        if not preds:
            return {"result": "Fail"}

        median_y = np.median([p["y"] for p in preds])
        preds = [p for p in preds if p["y"] >= median_y * 0.9]

        digit_imgs = []
        for p in sorted(preds, key=lambda b: b["x"]):
            x, y = int(p["x"]), int(p["y"])
            w, h = int(p["width"] // 2), int(p["height"] // 2)
            x1, y1 = max(x - w, 0), max(y - h, 0)
            x2, y2 = min(x + w, roi.shape[1]), min(y + h, roi.shape[0])
            digit_crop = roi_darker[y1:y2, x1:x2]
            if digit_crop.shape[0] < 10 or digit_crop.shape[1] < 10:
                continue
            resized = cv2.resize(digit_crop, (32, 64))
            digit_imgs.append(resized)

        if not digit_imgs:
            return {"result": "Fail"}

        row = cv2.hconcat(digit_imgs)
        img_rgb_row = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)
        ocr_results = ocr.ocr(img_rgb_row, det=False)

        if ocr_results and isinstance(ocr_results[0], list) and len(ocr_results[0]) > 0:
            raw_text = ocr_results[0][0][0]
            clean = re.sub(r"[^0-9]", "", raw_text).strip()

            if len(clean) == 8 and clean.startswith("1"):
                clean = clean[1:]

            if clean and len(clean) <= 8:
                return {"result": clean}
            else:
                return {"result": "Fail"}
        else:
            return {"result": "Fail"}

    except Exception:
        return {"result": "Fail"}
