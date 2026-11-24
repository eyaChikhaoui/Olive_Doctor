from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .config import (
    MODEL_DIR,
    MODEL_DROP_PATH_RATE,
    MODEL_DROP_RATE,
    MODEL_IMAGE_SIZE,
    MODEL_NAME,
)


@dataclass(frozen=True)
class SpecialistSpec:
    key: str
    display_name: str
    checkpoint_name: str
    class_ids: tuple[str, ...]
    labels_fr: dict[str, str]

    @property
    def checkpoint_path(self) -> Path:
        return MODEL_DIR / self.checkpoint_name


SPECIALISTS: dict[str, SpecialistSpec] = {
    "leaf": SpecialistSpec(
        key="leaf",
        display_name="Leaf Specialist",
        checkpoint_name="leaf_specialist_best.pth",
        class_ids=(
            "Fumagina",
            "Healthy_Leaf",
            "Mite_Damage",
            "Nutritional_Deficiency",
            "Peacock_Spot",
            "Virosis",
        ),
        labels_fr={
            "Fumagina": "Fumagine",
            "Healthy_Leaf": "Feuille saine",
            "Mite_Damage": "Dégâts d’acariens",
            "Nutritional_Deficiency": "Carence nutritionnelle",
            "Peacock_Spot": "Œil de paon",
            "Virosis": "Virose",
        },
    ),
    "branch": SpecialistSpec(
        key="branch",
        display_name="Branch Specialist",
        checkpoint_name="branch_specialist_best.pth",
        class_ids=("Black_Scale", "Fumagina", "Healthy_Branch", "Knot_Disease"),
        labels_fr={
            "Black_Scale": "Cochenille noire",
            "Fumagina": "Fumagine",
            "Healthy_Branch": "Branche saine",
            "Knot_Disease": "Tuberculose de l’olivier",
        },
    ),
}


class ModelLoadError(RuntimeError):
    """Raised when a real checkpoint or its runtime cannot be used."""


class SpecialistModel:
    def __init__(self, spec: SpecialistSpec):
        self.spec = spec
        self._model: Any | None = None
        self._torch: Any | None = None
        self.device_name: str | None = None
        self.error: str | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        try:
            import timm
            import torch
        except Exception as exc:
            raise ModelLoadError(
                "Le moteur PyTorch/timm est indisponible. Installez les dépendances de requirements.txt."
            ) from exc

        checkpoint_path = self.spec.checkpoint_path
        if not checkpoint_path.is_file():
            raise ModelLoadError(
                f"Le checkpoint {checkpoint_path.name} est introuvable dans le dossier models/."
            )

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ModelLoadError(f"Impossible de lire le checkpoint {checkpoint_path.name}.") from exc
        if not isinstance(checkpoint, dict):
            raise ModelLoadError(f"Le checkpoint {checkpoint_path.name} n’a pas un format de bundle reconnu.")

        checkpoint_model_name = checkpoint.get("model_name")
        checkpoint_classes = tuple(checkpoint.get("class_names", ()))
        checkpoint_image_size = checkpoint.get("img_size")
        if checkpoint_model_name != MODEL_NAME:
            raise ModelLoadError(
                f"Architecture inattendue dans {checkpoint_path.name}: {checkpoint_model_name!r}."
            )
        if checkpoint_classes != self.spec.class_ids:
            raise ModelLoadError(
                f"Ordre de classes inattendu dans {checkpoint_path.name}; aucun réordonnancement automatique ne sera fait."
            )
        if checkpoint_image_size != MODEL_IMAGE_SIZE:
            raise ModelLoadError(
                f"Résolution inattendue dans {checkpoint_path.name}: {checkpoint_image_size!r}."
            )

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ModelLoadError(f"Le checkpoint {checkpoint_path.name} ne contient pas model_state_dict.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=len(self.spec.class_ids),
            drop_rate=MODEL_DROP_RATE,
            drop_path_rate=MODEL_DROP_PATH_RATE,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ModelLoadError(f"Les poids de {checkpoint_path.name} ne correspondent pas à l’architecture attendue.") from exc
        model.to(device)
        model.eval()

        self._model = model
        self._torch = torch
        self.device_name = str(device)

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                try:
                    self._load()
                    self.error = None
                except ModelLoadError as exc:
                    self.error = str(exc)
                    raise

    def predict(self, image_array: object) -> list[float]:
        self.ensure_loaded()
        assert self._torch is not None
        assert self._model is not None
        device = next(self._model.parameters()).device
        tensor = self._torch.as_tensor(image_array, dtype=self._torch.float32, device=device)
        with self._torch.inference_mode():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        return [float(value) for value in probabilities]


class ModelRegistry:
    def __init__(self) -> None:
        self._models = {key: SpecialistModel(spec) for key, spec in SPECIALISTS.items()}

    def get(self, key: str) -> SpecialistModel:
        try:
            return self._models[key]
        except KeyError as exc:
            raise KeyError(f"Spécialiste inconnu: {key}") from exc

    def status(self) -> list[dict[str, object]]:
        runtime_available = True
        runtime_error: str | None = None
        try:
            import torch  # noqa: F401
            import timm  # noqa: F401
        except Exception as exc:
            runtime_available = False
            runtime_error = f"Runtime indisponible: {exc}"

        statuses = []
        for key, spec in SPECIALISTS.items():
            model = self._models[key]
            statuses.append(
                {
                    "key": key,
                    "display_name": spec.display_name,
                    "technical_name": MODEL_NAME,
                    "checkpoint": spec.checkpoint_name,
                    "checkpoint_available": spec.checkpoint_path.is_file(),
                    "loaded": model.loaded,
                    "runtime_available": runtime_available,
                    "device": model.device_name,
                    "error": model.error or runtime_error,
                }
            )
        return statuses
