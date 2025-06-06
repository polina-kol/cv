from models.unet_forest import UNet
import streamlit as st
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms



@st.cache_resource
def load_model(weights_path='models/unet_forest.pth'):
    model = UNet(1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

target_size = (256, 256)

def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(img_pil).unsqueeze(0)

def overlay_mask_on_image(image_pil, mask_np, alpha=0.5, color=(255, 0, 0)):
    """Наложение бинарной маски на изображение PIL."""
    image_np = np.array(image_pil).astype(np.uint8)
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[mask_np == 1] = color  # цвет маски

    overlay = (image_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

# Интерфейс
img_pil = None
st.title("U-Net: Сегментация и наложение маски")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')

url = st.text_input("Или введите ссылку на изображение")
if url and not img_pil:
    try:
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")

if img_pil:
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
    st.image(mask, caption="Вероятностная маска", use_column_width=True, clamp=True)
    st.image(binary_mask * 255, caption="Бинарная маска", use_column_width=True)

    # Наложение маски
    overlay = overlay_mask_on_image(img_pil, binary_mask, alpha=0.4, color=(255, 0, 0))
    st.image(overlay, caption="Изображение с наложенной маской", use_column_width=True)
else:
    st.info("Загрузите изображение или вставьте ссылку.")