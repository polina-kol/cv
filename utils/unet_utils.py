import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from models.unet_forest import UNet  # Импортируем твою модель, если она в другом файле

def load_model(path="models/unet_weights.pth"):
    model = UNet(n_class=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def preprocess(image_pil):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image_pil).unsqueeze(0)

def postprocess_mask(output_tensor, orig_size):
    pred = output_tensor.squeeze().detach().cpu().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(orig_size)
    return mask, mask_img

def overlay_mask_on_image(image, mask_array, alpha=0.4, color=(0, 255, 0)):
    image_np = np.array(image).copy()
    mask_resized = Image.fromarray(mask_array).resize(image.size)
    mask_bin = np.array(mask_resized) > 0

    overlay = image_np.copy()
    overlay[mask_bin] = (
        overlay[mask_bin] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)

    return Image.fromarray(overlay)
