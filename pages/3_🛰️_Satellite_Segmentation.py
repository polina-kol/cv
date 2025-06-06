import streamlit as st
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests

from utils.unet_utils import load_model, preprocess, overlay_mask_on_image

st.title("🛰️ Сегментация спутниковых изображений (U-Net)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображения:", ["Загрузка файла", "Ссылка (URL)"])

    img_pil = None

    if source == "Загрузка файла":
        uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file).convert('RGB')

    else:
        url = st.text_input("Введите URL изображения:")
        if url:
            try:
                response = requests.get(url)
                img_pil = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                st.error(f"Ошибка загрузки изображения: {e}")

    if img_pil:
        original_size = img_pil.size
        input_tensor = preprocess(img_pil)

        st.image(img_pil, caption="Оригинал", use_container_width=True)

        model = load_model()
        with torch.no_grad():
            output = model(input_tensor)
            mask = output.squeeze().cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8)

        st.subheader("Результаты сегментации")
        st.image(mask, caption="Вероятностная маска", use_container_width=True, clamp=True)
        st.image(binary_mask * 255, caption="Бинарная маска", use_container_width=True)

        overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4, color=(255, 0, 0))
        st.image(overlay, caption="Изображение с наложенной маской", use_container_width=True)
    else:
        st.info("Пожалуйста, загрузите изображение или введите URL.")

with tab2:
    st.subheader("U-Net модель сегментации растительности")
    st.markdown("- Обучена на спутниковых снимках")
    st.markdown("- Классификация: **растительность / не-растительность**")
    st.markdown("- Активация: `Sigmoid`, метки — бинарные")

    with st.expander("📊 Метрики модели"):
        st.markdown("- **Точность (Accuracy)**: 0.94")
        st.markdown("- **Полнота (Recall)**: 0.91")
        st.markdown("- **IoU (Intersection over Union)**: 0.87")
        st.markdown("- **Функция потерь**: BCEWithLogitsLoss")

        st.image("assets/loss_curve.png", caption="Кривая потерь")
        st.image("assets/iou_curve.png", caption="IoU по эпохам")
