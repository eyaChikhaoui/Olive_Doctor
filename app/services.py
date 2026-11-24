from __future__ import annotations

from .config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_UPLOAD_SIZE_BYTES
from .model_loader import ModelRegistry, ModelLoadError
from .preprocessing import preprocess_image


class InputValidationError(ValueError):
    """Raised for an invalid user-uploaded file."""


class AnalysisService:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def _validate_upload(self, content: bytes, filename: str, content_type: str | None) -> None:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_MIME_TYPES:
            raise InputValidationError("Format non pris en charge. Utilisez JPG, PNG ou WebP.")
        if len(content) == 0:
            raise InputValidationError("Le fichier importé est vide.")
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise InputValidationError(f"L’image dépasse la taille maximale de {max_mb} Mo.")

    def analyze(self, specialist_key: str, content: bytes, filename: str, content_type: str | None):
        self._validate_upload(content, filename, content_type)
        try:
            image_array, _ = preprocess_image(content)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc

        specialist = self.registry.get(specialist_key)
        try:
            probabilities = specialist.predict(image_array)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise RuntimeError("Une erreur technique est survenue pendant l’inférence du modèle.") from exc

        spec = specialist.spec
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        predicted_class = spec.class_ids[predicted_index]
        return {
            "specialist": spec.key,
            "specialist_display_name": spec.display_name,
            "technical_model": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
            "predicted_class": predicted_class,
            "predicted_label_fr": spec.labels_fr[predicted_class],
            "confidence": probabilities[predicted_index],
            "probabilities": [
                {
                    "class_id": class_id,
                    "label_fr": spec.labels_fr[class_id],
                    "probability": probability,
                }
                for class_id, probability in zip(spec.class_ids, probabilities)
            ],
            "device": specialist.device_name or "inconnu",
            "filename": filename,
        }
