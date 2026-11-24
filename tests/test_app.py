from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import main
from app.model_loader import ModelRegistry


client = TestClient(main.app)


def image_bytes(mode: str = "RGB") -> bytes:
    image = Image.new(mode, (640, 520), (70, 105, 58, 255) if mode == "RGBA" else (70, 105, 58))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_route_manifest():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["service"] == "Olive Doctor"
    manifest = client.get("/manus-routes.json")
    assert manifest.status_code == 200
    assert manifest.json()["routes"][0]["path"] == "/"


def test_model_status_reports_both_independent_checkpoints():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    payload = response.json()
    assert {item["key"] for item in payload["models"]} == {"leaf", "branch"}
    assert all(item["checkpoint"].endswith(".pth") for item in payload["models"])


def test_invalid_file_is_rejected_before_model_loading():
    response = client.post(
        "/api/analyze/leaf",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Format" in response.json()["detail"]


def test_corrupt_image_is_rejected():
    response = client.post(
        "/api/analyze/branch",
        files={"file": ("branch.png", b"broken", "image/png")},
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_frontend_contains_required_french_interactions():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
    script = (root / "frontend" / "script.js").read_text(encoding="utf-8")
    assert "Leaf Specialist" in html and "Branch Specialist" in html
    assert "Glissez-déposez" in html and "Nouvelle analyse" in html
    assert "/api/analyze/${state.model}" in script
    assert "application/json" not in html


@pytest.mark.skipif(os.getenv("OLIVE_DOCTOR_RUN_MODEL_TESTS") != "1", reason="test lourd désactivé par défaut")
def test_real_specialists_return_consistent_probabilities():
    leaf = client.post("/api/analyze/leaf", files={"file": ("leaf.png", image_bytes(), "image/png")})
    branch = client.post("/api/analyze/branch", files={"file": ("branch.png", image_bytes(), "image/png")})
    assert leaf.status_code == 200, leaf.text
    assert branch.status_code == 200, branch.text
    for response, expected_count, expected_key in [(leaf, 6, "leaf"), (branch, 4, "branch")]:
        payload = response.json()
        assert payload["specialist"] == expected_key
        assert len(payload["probabilities"]) == expected_count
        assert abs(sum(item["probability"] for item in payload["probabilities"]) - 1.0) < 1e-5
        assert payload["predicted_class"] in {item["class_id"] for item in payload["probabilities"]}


def test_model_registry_has_separate_instances():
    registry = ModelRegistry()
    assert registry.get("leaf") is not registry.get("branch")
    assert registry.get("leaf").spec.class_ids != registry.get("branch").spec.class_ids
