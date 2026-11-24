from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from .config import IMAGENET_MEAN, IMAGENET_STD, MODEL_IMAGE_SIZE, MODEL_SCALE_SIZE


def _prepare_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and composite transparency onto neutral gray."""
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def _smallest_max_size(image: Image.Image, max_size: int) -> Image.Image:
    """Mirror albumentations SmallestMaxSize for a single image."""
    width, height = image.size
    smallest_side = min(width, height)
    if smallest_side <= 0:
        raise ValueError("L’image possède des dimensions invalides.")
    scale = max_size / smallest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        raise ValueError("L’image est trop petite après redimensionnement.")
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_image(image_bytes: bytes) -> tuple[object, Image.Image]:
    """Return a normalized CHW tensor-like array and the prepared preview image."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = _prepare_rgb(source)
            image = _smallest_max_size(image, MODEL_SCALE_SIZE)
            image = _center_crop(image, MODEL_IMAGE_SIZE)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError("Le fichier ne contient pas une image exploitable.") from exc

    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return array, image
