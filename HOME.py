import streamlit as st

st.set_page_config(page_title="Computer Vision project", page_icon="🤖", layout="wide")

# Вставка CSS для увеличения шрифтов
st.markdown("""
<style>
    h1 {
        font-size: 42px !important;
    }
    .big-text {
        font-size: 22px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Computer Vision project")
st.markdown('<p class="big-text">Добро пожаловать! Здесь вы можете:</p>', unsafe_allow_html=True)
st.markdown("""
<ul class="big-text">
<li>👤 <b>Блюрить лица на изображениях</b></li>
<li>🧠 <b>Детектировать опухоли мозга с учётом типа среза</b></li>
<li>🛰️ <b>Выполнять сегментацию аэроснимков</b></li>
</ul>
""", unsafe_allow_html=True)