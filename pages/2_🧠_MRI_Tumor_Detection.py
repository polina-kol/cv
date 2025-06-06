import streamlit as st
from utils.brain_utils import detect_tumor  # Теперь classify_slice не используется
from PIL import Image
import requests
from io import BytesIO

st.title("🧠 Срезы мозга и детекция опухоли")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображений:", ["Загрузка файлов", "Ссылка (URL)"])

    # Выбор типа среза один раз (можно сделать для каждого файла отдельно, если нужно)
    slice_type = st.selectbox("Выберите тип среза", ["Axial", "Sagittal", "Coronal"])
    slice_type = slice_type.lower()  # Нужно для совместимости с detect_tumor()

    if source == "Загрузка файлов":
        uploaded_files = st.file_uploader("Загрузите MRI срезы", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        for file in uploaded_files:
            image = Image.open(file)
            detection_image = detect_tumor(image, slice_type)
            st.image(
                detection_image,
                caption=f"{slice_type.upper()} срез — результат YOLO",
                use_container_width=True
            )
    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке"):
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                detection_image = detect_tumor(image, slice_type)
                st.image(
                    detection_image,
                    caption=f"{slice_type.upper()} срез — результат YOLO",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Ошибка при загрузке или обработке изображения: {e}")

with tab2:
    st.subheader("YOLOv12m модели для детекции опухоли")
    st.markdown("- 3 отдельных YOLOv12m модели (по типу среза)")
    st.markdown("- **Эпох**: 50 (у каждой)")

    with st.expander("📊 Axial — Метрики и описание датасета"):
        st.image("assets/ax_data.png", caption="Data — Axial")
        st.image("assets/confusion_ax.png", caption="Confusion Matrix — Axial")
        st.image("assets/pr_ax.png", caption="PR Curve — Axial")

    with st.expander("📊 Coronal — Метрики и описание датасета"):
        st.image("assets/data_cor.png", caption="Data — Coronal")
        st.image("assets/confusion_cor.png", caption="Confusion Matrix — Coronal")
        st.image("assets/pr_cor.png", caption="PR Curve — Coronal")

    with st.expander("📊 Sagittal — Метрики и описание датасета"):
        st.image("assets/data_sag.png", caption="Data — Sagittal")
        st.image("assets/conf_sag.png", caption="Confusion Matrix — Sagittal")
        st.image("assets/pr_sag.png", caption="PR Curve — Sagittal")
