import torch
from PIL import Image
from ultralytics import YOLO

# === Загрузка YOLO моделей по типу среза ===
yolo_models = {
    "axial": YOLO("models/yolo_axial.pt"),
    "sagittal": YOLO("models/yolo_sag.pt"),
    "coronal": YOLO("models/yolo_coronal.pt"),
}

def ensure_rgb(image: Image.Image) -> Image.Image:
    """Преобразует изображение в RGB формат."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def detect_tumor(image: Image.Image, slice_type: str) -> Image.Image:
    """
    Детекция опухоли с использованием соответствующей YOLO модели.
    
    Параметры:
        image (PIL.Image): Входное изображение.
        slice_type (str): Тип среза ('axial', 'sagittal' или 'coronal').

    Возвращает:
        PIL.Image: Изображение с аннотациями обнаруженных опухолей.
    """
    image = ensure_rgb(image)
    
    slice_type = slice_type.lower()
    if slice_type not in yolo_models:
        raise ValueError(f"Unsupported slice type: {slice_type}. Choose from 'axial', 'sagittal', or 'coronal'.")
    
    model = yolo_models[slice_type]
    results = model.predict(image, conf=0.3, save=False, imgsz=512)
    annotated_image = results[0].plot()  # Получаем изображение с боксами
    return Image.fromarray(annotated_image)