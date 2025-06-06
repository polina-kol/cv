import torch
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.unet_forest import UNet
import streamlit as st

# Размер входа модели
TARGET_SIZE = (256, 256)

# Кэшированная загрузка модели

def load_model(weights_path='models/unet_forest.pth'):
    model = UNet(n_class=1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# Предобработка изображения
def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
    ])
    return transform(img_pil).unsqueeze(0)  # [1, C, H, W]

# Постобработка маски
def postprocess_mask(output_tensor, original_size):
    mask = torch.sigmoid(output_tensor).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    binary_mask_resized = Image.fromarray(binary_mask * 255).resize(original_size)
    return binary_mask, binary_mask_resized

# Наложение маски
def overlay_mask_on_image(image_pil, mask_np, alpha=0.4, color=(0, 255, 0)):
    image_np = np.array(image_pil).astype(np.uint8)
    if mask_np.shape != image_np[:2]:
        mask_np = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize(image_np.shape[:2][::-1]))
        mask_np = (mask_np > 127).astype(np.uint8)
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[mask_np == 1] = color
    overlay = (image_np * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return Image.fromarray(overlay)
