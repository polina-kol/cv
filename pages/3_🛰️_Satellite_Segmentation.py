import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
import numpy as np
import sys
import os
from torchvision import transforms

sys.path.append(os.path.abspath("../"))

# Теперь можно импортировать модели
from models.unet_forest import UNet


# Настройки
target_size = (256, 256)
weights_path = "models/weights_forest.pth"

# Функция загрузки модели
@st.cache_resource
def load_model():
    model = UNet(n_class=1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# Предобработка изображения
def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(img_pil).unsqueeze(0)

# Наложение маски
def overlay_mask_on_image(image_pil, mask_np, alpha=0.5, color=(255, 0, 0)):
    image_np = np.array(image_pil).astype(np.uint8)
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[mask_np == 1] = color  # цвет маски

    overlay = (image_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

# Интерфейс
st.set_page_config(page_title="Сегментация спутниковых снимков", page_icon="🛰️")
st.title("🛰️ Сегментация спутниковых снимков (U-Net)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    st.subheader("Выберите способ загрузки изображения")
    source = st.radio("Источник изображения:", ["Загрузить файл", "Указать URL"])

    uploaded_file = None
    url = ""

    if source == "Загрузить файл":
        uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    else:
        url = st.text_input("Введите URL изображения:")

    if uploaded_file or url:
        try:
            if uploaded_file:
                img_pil = Image.open(uploaded_file).convert('RGB')
            else:
                response = requests.get(url)
                img_pil = Image.open(BytesIO(response.content)).convert('RGB')

            img_pil = img_pil.resize(target_size)
            input_tensor = preprocess(img_pil)

            st.image(img_pil, caption="Оригинал", use_column_width=True)

            model = load_model()
            with torch.no_grad():
                output = model(input_tensor)
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)

            # Визуализация
            st.subheader("Результаты сегментации")
            col1, col2 = st.columns(2)
            with col1:
                st.image(mask, caption="Вероятностная маска", use_column_width=True, clamp=True)
            with col2:
                st.image(binary_mask * 255, caption="Бинарная маска", use_column_width=True)

            # Наложение маски
            overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4, color=(0, 255, 0))
            st.image(overlay, caption="Изображение с наложенной маской", use_column_width=True)

            # Кнопка скачивания
            buf = BytesIO()
            overlay.save(buf, format="PNG")
            st.download_button(
                label="📥 Скачать маску",
                data=buf.getvalue(),
                file_name="segmentation_mask.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Ошибка: {e}")

    else:
        st.info("Загрузите изображение или вставьте ссылку.")

with tab2:
    st.header("U-Net для семантической сегментации")
    st.markdown("""
    - **Модель**: UNet
    - **Тип задачи**: Бинарная сегментация (например, лес / не лес)
    - **Формат входного изображения**: RGB (256x256)
    """)
    st.image("assets/map_stats_unet.png", caption="PR / IoU / F1-графики")