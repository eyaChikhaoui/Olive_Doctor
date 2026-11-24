from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"

MAX_UPLOAD_SIZE_BYTES = int(os.getenv("OLIVE_DOCTOR_MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
MODEL_IMAGE_SIZE = 384
MODEL_SCALE_SIZE = int(MODEL_IMAGE_SIZE * 1.12)
MODEL_DROP_RATE = 0.35
MODEL_DROP_PATH_RATE = 0.30
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
