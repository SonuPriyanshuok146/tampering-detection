import numpy as np
from PIL import Image, ImageChops
import io


def compute_ela(image_path, quality=90, scale=15):
    """
    Compute the Error Level Analysis image for a given input image.

    Args:
        image_path (str): path to the input image
        quality (int): JPEG re-save quality (90 is a common default)
        scale (int): amplification factor for the difference, for visibility

    Returns:
        PIL.Image: RGB ELA image, same size as the input
    """
    original = Image.open(image_path).convert("RGB")

    # Re-save the image at a known JPEG quality, in memory
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Pixel-wise difference between original and re-saved version
    diff = ImageChops.difference(original, resaved)

    # Amplify the difference so tampered regions are visible
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale_factor = 255.0 / max_diff
    diff = Image.eval(diff, lambda px: min(255, int(px * scale_factor)))

    return diff


def ela_to_array(ela_image, size=(128, 128)):
    """Resize an ELA PIL image and convert it to a normalized numpy array."""
    ela_image = ela_image.resize(size)
    arr = np.array(ela_image).astype("float32") / 255.0
    return arr