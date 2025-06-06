import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Загрузка YOLO модели (предположим, она в models/yolo_face.pt)
face_model = YOLO("models/yolo_face2.pt")

def detect_and_blur_faces(image: Image.Image, blur_radius=15) -> Image.Image:
    """
    Детектирует лица и блюрит их на изображении.

    Args:
        image (PIL.Image): Входное изображение.
        blur_radius (int): Радиус размытия Гаусса.

    Returns:
        PIL.Image: Изображение с заблюренными лицами.
    """
    results = face_model.predict(image, conf=0.15, save=False, imgsz=640)

    img_np = np.array(image.convert("RGB"))
    img_pil = Image.fromarray(img_np)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Вырезаем лицо и применяем блюр с заданным радиусом
        face_crop = img_pil.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img_pil.paste(face_crop, (x1, y1))

    return img_pil
