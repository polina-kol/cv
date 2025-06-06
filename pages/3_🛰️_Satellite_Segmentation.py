import streamlit as st
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from utils.unet_utils import load_model, preprocess, overlay_mask_on_image, target_size

st.title("🛰️ Сегментация изображения (U-Net)")

img_pil = None
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')

url = st.text_input("Или введите URL изображения:")
if url and not img_pil:
    try:
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")

if img_pil:
    original_size = img_pil.size  # Сохраняем оригинальный размер изображения

    # Предобработка изображения для модели
    input_tensor = preprocess(img_pil)

    # Отображение оригинального изображения
    st.image(img_pil, caption="Оригинал", use_container_width=True)

    # Загрузка и применение модели
    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)

    # Отображение результатов
    st.subheader("Результаты сегментации")
    st.image(mask, caption="Вероятностная маска", use_container_width=True, clamp=True)
    st.image(binary_mask * 255, caption="Бинарная маска", use_container_width=True)

    # Наложение маски на оригинал
    overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4, color=(255, 0, 0))
    st.image(overlay, caption="Изображение с наложенной маской", use_container_width=True)
else:
    st.info("Загрузите изображение или вставьте ссылку.")
