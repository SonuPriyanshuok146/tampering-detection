import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .model import build_model


def generate_gradcam(model, input_tensor, target_layer=None, device="cpu"):
    """
    Args:
        model: trained model (in eval mode)
        input_tensor: single image tensor, shape (1, 3, H, W)
        target_layer: conv layer to hook (defaults to last ResNet block)

    Returns:
        numpy array (H, W, 3) — the ELA image with the Grad-CAM heatmap overlaid
    """
    if target_layer is None:
        target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.to(device).eval()