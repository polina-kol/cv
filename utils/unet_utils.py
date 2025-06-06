import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import streamlit as st
from models.unet_forest import UNet

target_size = (256, 256)

def preprocess(img_pil, target_size=(256, 256)):
    img_pil = img_pil.convert("RGB") 
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(img_pil).unsqueeze(0)


def overlay_mask_on_image(image_pil, mask_np, alpha=0.5, color=(255, 0, 0)):
    """Наложение бинарной маски на изображение PIL."""
    image_np = np.array(image_pil).astype(np.uint8)

    # Масштабируем маску к размеру изображения, если нужно
    if mask_np.shape != (image_pil.height, image_pil.width):
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)
        mask_np = np.array(mask_pil) // 255

    mask_rgb = np.zeros_like(image_np)
    mask_rgb[mask_np == 1] = color  # Цвет маски

    overlay = (image_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

@st.cache_resource
def load_model(weights_path=None):
    if weights_path is None:
        # Автоматически определяет путь до весов
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "..", "models", "unet_forest.pth")
    model = UNet(1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model
