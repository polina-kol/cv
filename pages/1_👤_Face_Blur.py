import streamlit as st
from utils.face_utils import detect_and_blur_faces
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Детекция лиц и блюринг", page_icon="👤")
st.title("👤 Детекция лиц и блюринг")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображений:", ["Загрузка файлов", "Ссылка (URL)"])

    # Ползунок для выбора силы блюра
    blur_radius = st.slider("Сила блюра", min_value=1, max_value=50, value=15, step=1)

    if source == "Загрузка файлов":
        uploaded_files = st.file_uploader(
            "Загрузите изображение(я)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                result = detect_and_blur_faces(image, blur_radius=blur_radius)
                
                st.image(result, caption=f"Обработано: {file.name}", use_container_width=True)
                
                buf = BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    label="📥 Скачать обработанное изображение",
                    data=buf.getvalue(),
                    file_name=f"blurred_{file.name}",
                    mime="image/png"
                )

    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке"):
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                result = detect_and_blur_faces(image, blur_radius=blur_radius)

                st.image(result, caption="Обработанное изображение", use_container_width=True)

                buf = BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    label="📥 Скачать обработанное изображение",
                    data=buf.getvalue(),
                    file_name="blurred_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения: {e}")

with tab2:
    st.header("YOLOv5 face detector + OpenCV Blur")
    st.markdown("""
    - **Модель**: YOLOv5s, дообученная на [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
    - **Метрика mAP**: 0.89
    - **Эпох**: 50  
    """)
    st.image("assets/pr_curve_face.png", caption="PR-кривая")
