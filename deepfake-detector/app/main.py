"""
Obscuro Deepimage — FastAPI backend.

Endpoints:
  POST /api/analyse   — accepts image or video upload, returns ForensicVerdict JSON
  GET  /api/health    — liveness/readiness check
  GET  /api/stats     — self-improvement agent accuracy report
  POST /api/feedback  — submit ground-truth label for a previous prediction
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from app.schemas import ForensicVerdict, HealthResponse
from app.models import detector
from app.agents import coordinator, self_improvement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Obscuro Deepimage starting — loading models...")
    ok = detector.load_models()
    if not ok:
        logger.warning("Model loading incomplete — some detection signals may be unavailable.")
    yield
    logger.info("Obscuro Deepimage shutting down.")


app = FastAPI(
    title="Obscuro Deepimage",
    description=(
        "Agentic Multi-Modal Deepfake Detection System. "
        "Detects deepfake images and videos using multiple independent forensic signals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=detector.is_loaded(),
    )


@app.post("/api/analyse", response_model=ForensicVerdict, tags=["detection"])
async def analyse(
    file: UploadFile = File(..., description="Image (JPG/PNG/WebP) or video (MP4/AVI/MOV)"),
):
    """
    Analyse an image or video for deepfake content.

    Returns a ForensicVerdict with:
    - Overall verdict and deepfake probability
    - Per-agent signal breakdown  
    - Fusion weights (including quantum-inspired calibration)
    - Structured prose rationale
    """
    max_size_mb = 100
    data = await file.read()

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(data) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size_mb}MB.",
        )

    filename = file.filename or "upload"
    logger.info(
        "Analysis request: file=%s, size=%.1f KB",
        filename, len(data) / 1024,
    )

    try:
        verdict = coordinator.route(data, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # Log to self-improvement agent
    input_hash = self_improvement.make_input_hash(data)
    self_improvement.log_prediction(input_hash, verdict)

    return verdict


@app.get("/api/stats", tags=["system"])
async def stats():
    """
    Return accuracy statistics and self-improvement agent report.
    Only meaningful when labelled feedback has been submitted via /api/feedback.
    """
    report = self_improvement.compute_accuracy_report()
    recal = self_improvement.recalibrate_weights_if_needed()
    return {"accuracy_report": report, "weight_recalibration": recal}


@app.post("/api/feedback", tags=["system"])
async def feedback(
    input_hash: str = Form(...),
    ground_truth: str = Form(..., description="'fake' or 'real'"),
):
    """
    Submit a ground-truth label for a previous prediction.
    This feeds the Self-Improvement Agent's recalibration loop.
    """
    if ground_truth not in ("fake", "real"):
        raise HTTPException(status_code=400, detail="ground_truth must be 'fake' or 'real'.")

    # Re-log with ground truth attached (simplified — in production, update existing record)
    from app.schemas import AgentResult, ForensicVerdict
    dummy_verdict = ForensicVerdict(
        verdict="UNKNOWN",
        deepfake_probability=0.5,
        confidence_in_verdict=0.0,
        agent_results=[],
        fusion_weights={},
        rationale="Feedback record only.",
    )
    self_improvement.log_prediction(input_hash, dummy_verdict, ground_truth=ground_truth)
    return {"status": "feedback_recorded", "input_hash": input_hash, "ground_truth": ground_truth}
