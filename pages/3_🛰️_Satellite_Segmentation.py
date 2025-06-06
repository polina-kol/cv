import streamlit as st
from PIL import Image
import requests
import torch
from io import BytesIO
from utils.unet_utils import load_model, preprocess, postprocess_mask, overlay_mask_on_image

# Настройки
st.set_page_config(page_title="Сегментация спутниковых снимков", page_icon="🛰️")
st.title("🛰️ Сегментация спутниковых снимков (U-Net)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    st.subheader("Выберите способ загрузки изображения")
    source = st.radio("Источник изображения:", ["Загрузить файл", "Указать URL"])

    img_pil = None
    if source == "Загрузить файл":
        uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Введите URL изображения:")
        if url:
            try:
                response = requests.get(url)
                img_pil = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения: {e}")

    if img_pil:
        st.image(img_pil, caption="Оригинал", use_container_width=True)
        input_tensor = preprocess(img_pil)
        model = load_model()

        with st.spinner("Обработка..."):
            with torch.no_grad():
                output = model(input_tensor)
                binary_mask, binary_mask_img = postprocess_mask(output, img_pil.size)

            # Показываем маски
            st.subheader("Результаты сегментации")
            col1, col2 = st.columns(2)
            with col1:
                st.image(binary_mask_img, caption="Бинарная маска", use_column_width=True)
            with col2:
                overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4, color=(0, 255, 0))
                st.image(overlay, caption="Изображение с наложенной маской", use_column_width=True)

            # Кнопка загрузки
            buf = BytesIO()
            binary_mask_img.save(buf, format="PNG")
            st.download_button(
                label="📥 Скачать маску",
                data=buf.getvalue(),
                file_name="mask.png",
                mime="image/png"
            )
    else:
        st.info("Загрузите изображение или вставьте URL.")

with tab2:
    st.header("Информация о модели")
    st.markdown("""
    - **Архитектура**: U-Net (кастомная)
    - **Тип**: Бинарная сегментация (например, лес / не лес)
    - **Входной размер**: 256×256
    - **Язык**: PyTorch
    """)
    st.image("assets/map_stats_unet.png", caption="PR / IoU / F1-графики")
