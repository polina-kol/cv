import streamlit as st
from PIL import Image
import requests
import torch
from io import BytesIO
from utils.unet_utils import load_model, preprocess, postprocess_mask, overlay_mask_on_image

# Конфиг
st.set_page_config(page_title="Сегментация спутниковых снимков", page_icon="🛰️")
st.title("🛰️ Сегментация спутниковых снимков (U-Net)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    st.subheader("Загрузка изображения")
    source = st.radio("Источник:", ["Файл", "URL"])

    img_pil = None
    if source == "Файл":
        uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Введите URL изображения:")
        if url:
            try:
                response = requests.get(url)
                img_pil = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")

    if img_pil:
        st.image(img_pil, caption="Оригинал", use_container_width=True)
        input_tensor = preprocess(img_pil)
        st.write("Текущая рабочая директория:", os.getcwd())
        st.write("Содержимое директории:", os.listdir())
        st.write("Содержимое models/:", os.listdir("models") if os.path.exists("models") else "❌ models/ не найдена")

        model = load_model()

        with st.spinner("Обработка..."):
            with torch.no_grad():
                output = model(input_tensor)
                binary_mask, binary_mask_img = postprocess_mask(output, img_pil.size)

        st.subheader("Результаты")
        col1, col2 = st.columns(2)
        with col1:
            st.image(binary_mask_img, caption="Бинарная маска", use_column_width=True)
        with col2:
            overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4)
            st.image(overlay, caption="Маска на изображении", use_column_width=True)

        buf = BytesIO()
        binary_mask_img.save(buf, format="PNG")
        st.download_button("📥 Скачать маску", data=buf.getvalue(), file_name="mask.png", mime="image/png")

    else:
        st.info("Пожалуйста, загрузите изображение или введите URL.")

with tab2:
    st.markdown("""
    ### Модель
    - Архитектура: U-Net
    - Тип сегментации: бинарная
    - Размер входа: 256×256
    - Фреймворк: PyTorch

    ![PR графики](assets/map_stats_unet.png)
    """)
