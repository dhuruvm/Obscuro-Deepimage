"""
Obscuro Deepimage — FastAPI backend + HTML frontend server.

Endpoints:
  GET  /                      — Main UI (HTML single-page app)
  POST /api/analyse           — Analyse image/video → ForensicVerdict + forensic report
  GET  /api/health            — Liveness/readiness check
  GET  /api/stats             — Self-improvement accuracy report
  POST /api/feedback          — Submit ground-truth label
  POST /api/train/start       — Start autonomous training
  GET  /api/train/status      — Training status and progress
  POST /api/train/stop        — Stop training
  POST /api/evolve/start      — Start algorithm evolution
  GET  /api/evolve/status     — Algorithm evolver status
  POST /api/evolve/stop       — Stop evolution
  POST /api/evolve/reset      — Reset to default config
  GET  /api/evolve/config     — Get active algorithm config
"""
import logging
import time
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from typing import Optional

from app.schemas import (
    ForensicVerdict, HealthResponse,
    TrainingStartRequest, EvolutionStartRequest,
)
from app.models import detector
from app.agents import coordinator, self_improvement
from app.agents import auto_trainer, algorithm_evolver
from app.agents.forensic_writer import generate_forensic_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Obscuro Deepimage starting — loading models...")
    ok = detector.load_models()
    if not ok:
        logger.warning("Model loading incomplete — some signals may be unavailable.")
    # Apply any previously evolved algorithm config
    config = algorithm_evolver.get_active_config()
    if config.get("weights"):
        algorithm_evolver._apply_config_to_fusion(config)
    yield
    logger.info("Obscuro Deepimage shutting down.")


app = FastAPI(
    title="Obscuro Deepimage",
    description="Agentic Multi-Modal Deepfake Detection System — Forensic Specialist Edition.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the main UI."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(), status_code=200)
    return HTMLResponse("<h1>Obscuro Deepimage</h1><p>UI file not found.</p>", 500)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Minimal 1×1 transparent ICO so browsers don't log 404s
    ico = (
        b'\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00'
        b'\x30\x00\x00\x00\x16\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00'
        b'\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x04\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x1a\x1a\x2b\xff\x00\x00\x00\x00'
    )
    return Response(content=ico, media_type="image/x-icon")


# ─── Detection ────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(status="ok", models_loaded=detector.is_loaded())


@app.post("/api/analyse", response_model=ForensicVerdict, tags=["detection"])
async def analyse(
    file: UploadFile = File(..., description="Image (JPG/PNG/WebP) or video (MP4/AVI/MOV)"),
):
    """Analyse an image or video for deepfake content. Returns full forensic verdict + specialist report."""
    max_size_mb = 100
    data = await file.read()

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {max_size_mb}MB.")

    filename = file.filename or "upload"
    logger.info("Analysis request: file=%s, size=%.1f KB", filename, len(data) / 1024)

    try:
        verdict = coordinator.route(data, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # Log prediction for self-improvement + training
    input_hash = self_improvement.make_input_hash(data)
    self_improvement.log_prediction(input_hash, verdict)

    # Generate forensic specialist report
    verdict_dict = verdict.model_dump()
    verdict_dict["agent_results"] = [ar.model_dump() for ar in verdict.agent_results]
    forensic_report = generate_forensic_report(verdict_dict, filename=filename, input_hash=input_hash)

    # Return verdict with report embedded
    return ForensicVerdict(
        verdict=verdict.verdict,
        deepfake_probability=verdict.deepfake_probability,
        confidence_in_verdict=verdict.confidence_in_verdict,
        agent_results=verdict.agent_results,
        fusion_weights=verdict.fusion_weights,
        rationale=verdict.rationale,
        forensic_report=forensic_report,
        processing_time_s=verdict.processing_time_s,
        media_type=verdict.media_type,
        warnings=verdict.warnings,
    )


@app.get("/api/stats", tags=["system"])
async def stats():
    report = self_improvement.compute_accuracy_report()
    recal = self_improvement.recalibrate_weights_if_needed()
    labelled = auto_trainer.get_labelled_count()
    return {
        "accuracy_report": report,
        "weight_recalibration": recal,
        "labelled_samples": labelled,
    }


@app.post("/api/feedback", tags=["system"])
async def feedback(
    input_hash: str = Form(...),
    ground_truth: str = Form(..., description="'fake' or 'real'"),
):
    if ground_truth not in ("fake", "real"):
        raise HTTPException(status_code=400, detail="ground_truth must be 'fake' or 'real'.")

    from app.schemas import ForensicVerdict as FV
    dummy = FV(
        verdict="UNKNOWN",
        deepfake_probability=0.5,
        confidence_in_verdict=0.0,
        agent_results=[],
        fusion_weights={},
        rationale="Feedback record.",
    )
    self_improvement.log_prediction(input_hash, dummy, ground_truth=ground_truth)
    return {"status": "feedback_recorded", "input_hash": input_hash, "ground_truth": ground_truth}


# ─── Training ─────────────────────────────────────────────────────────────────

@app.post("/api/train/start", tags=["training"])
async def train_start(req: TrainingStartRequest):
    """Start autonomous model fine-tuning in the background."""
    result = auto_trainer.start_training(
        epochs=req.epochs,
        samples_per_class=req.samples_per_class,
        lr=req.lr,
    )
    return result


@app.get("/api/train/status", tags=["training"])
async def train_status():
    """Get current training status and progress."""
    return auto_trainer.get_training_status()


@app.post("/api/train/stop", tags=["training"])
async def train_stop():
    """Request graceful training stop."""
    return auto_trainer.stop_training()


# ─── Algorithm Evolution ──────────────────────────────────────────────────────

@app.post("/api/evolve/start", tags=["evolution"])
async def evolve_start(req: EvolutionStartRequest):
    """Start autonomous algorithm evolution."""
    result = algorithm_evolver.start_evolution(
        generations=req.generations,
        population_size=req.population_size,
    )
    return result


@app.get("/api/evolve/status", tags=["evolution"])
async def evolve_status():
    """Get algorithm evolver status."""
    return algorithm_evolver.get_evolver_status()


@app.post("/api/evolve/stop", tags=["evolution"])
async def evolve_stop():
    return algorithm_evolver.stop_evolution()


@app.post("/api/evolve/reset", tags=["evolution"])
async def evolve_reset():
    """Reset algorithm configuration to default weights."""
    return algorithm_evolver.reset_to_default()


@app.get("/api/evolve/config", tags=["evolution"])
async def evolve_config():
    """Get the currently active algorithm configuration."""
    return algorithm_evolver.get_active_config()
