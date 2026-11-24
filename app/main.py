from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_ASSETS_DIR, FRONTEND_DIR
from .model_loader import ModelLoadError
from .schemas import AnalysisResponse, HealthResponse, ModelsStatusResponse
from .services import AnalysisService, InputValidationError

app = FastAPI(
    title="Olive Doctor",
    description="Analyse indépendante des feuilles et branches d’olivier.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

analysis_service = AnalysisService()

if FRONTEND_ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "indisponible"
    return HealthResponse(status="ok", service="Olive Doctor", device=device)


@app.get("/api/models/status", response_model=ModelsStatusResponse)
def models_status() -> ModelsStatusResponse:
    return ModelsStatusResponse(models=analysis_service.registry.status())


async def _analyze(specialist_key: str, file: UploadFile) -> AnalysisResponse:
    content = await file.read()
    try:
        result = analysis_service.analyze(
            specialist_key,
            content,
            file.filename or "image",
            file.content_type,
        )
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnalysisResponse(**result)


@app.post("/api/analyze/leaf", response_model=AnalysisResponse)
async def analyze_leaf(file: UploadFile = File(...)) -> AnalysisResponse:
    return await _analyze("leaf", file)


@app.post("/api/analyze/branch", response_model=AnalysisResponse)
async def analyze_branch(file: UploadFile = File(...)) -> AnalysisResponse:
    return await _analyze("branch", file)


@app.get("/manus-routes.json")
def routes_manifest():
    manifest_path = FRONTEND_DIR / "manus-routes.json"
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}")
def static_frontend(path: str) -> FileResponse:
    candidate = (FRONTEND_DIR / path).resolve()
    frontend_root = FRONTEND_DIR.resolve()
    if candidate.is_file() and frontend_root in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(FRONTEND_DIR / "index.html")
