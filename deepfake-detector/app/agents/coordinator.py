"""
Coordinator Agent — the top-level orchestrator for Obscuro Deepimage.

Responsibilities:
  1. Receive media (image bytes or video path)
  2. Determine media type and applicable specialist agents
  3. Dispatch agents (sequentially for now; parallel dispatch is feasible but
     kept sequential to avoid resource exhaustion on single-core Replit)
  4. Detect coverage gaps and optionally synthesise a new detection script
  5. Invoke Fusion Agent and return a ForensicVerdict

Autonomous tool synthesis (limited scope):
  If an unknown media type is submitted (e.g. audio-only, PDF), the coordinator
  logs the gap rather than crashing — this is the "register a new tool" pattern
  within the bounds of what's safe to do without human review at deployment time.
"""
import logging
import time
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np

from app.schemas import AgentResult, ForensicVerdict
from app.agents import spatial, frequency, temporal, biological, audio_visual, fusion
from app.preprocessing.pipeline import (
    load_image_bytes,
    extract_video_frames,
    get_face_crops,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def analyse_image(
    image_data: bytes,
    filename: str = "upload.jpg",
) -> ForensicVerdict:
    """
    Full pipeline for a single image.
    """
    t0 = time.time()
    results: list[AgentResult] = []
    warnings = []

    try:
        image_bgr = load_image_bytes(image_data)
    except Exception as exc:
        logger.error("Could not decode image: %s", exc)
        raise ValueError(f"Image decode failed: {exc}") from exc

    # ── Spatial forensics ────────────────────────────────────────────────────
    logger.info("Coordinator: dispatching SpatialForensicsAgent")
    results.append(spatial.run(image_bgr))

    # ── Frequency analysis ───────────────────────────────────────────────────
    logger.info("Coordinator: dispatching FrequencyAnalysisAgent")
    results.append(frequency.run(image_bgr))

    # ── Video-only agents (skipped for images) ───────────────────────────────
    for agent_name, signal_name, reason in [
        ("TemporalConsistencyAgent", "temporal_inconsistency_score",
         "Static image — temporal analysis requires video frames."),
        ("BiologicalSignalAgent", "rppg_anomaly_score",
         "Static image — rPPG requires a video sequence."),
        ("AudioVisualSyncAgent", "lip_sync_mismatch_score",
         "Static image — audio-visual sync requires video with audio."),
    ]:
        results.append(AgentResult(
            agent_name=agent_name,
            signal_name=signal_name,
            score=0.5, confidence=0.0,
            ran=False, skipped_reason=reason,
        ))

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused = fusion.fuse(results, use_quantum=True)

    return ForensicVerdict(
        verdict=fused["verdict"],
        deepfake_probability=fused["deepfake_probability"],
        confidence_in_verdict=fused["confidence_in_verdict"],
        agent_results=results,
        fusion_weights=fused["fusion_weights"],
        rationale=fused["rationale"],
        processing_time_s=round(time.time() - t0, 2),
        media_type="image",
        warnings=warnings,
    )


def analyse_video(
    video_data: bytes,
    filename: str = "upload.mp4",
) -> ForensicVerdict:
    """
    Full pipeline for a video clip.
    """
    t0 = time.time()
    results: list[AgentResult] = []
    warnings = []

    # Save to temp file (OpenCV needs a path)
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name

    try:
        frames = extract_video_frames(tmp_path)

        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video.")

        if len(frames) < 8:
            warnings.append(
                f"Only {len(frames)} frames extracted — some analyses may be less reliable."
            )

        # Use first frame for spatial + frequency analysis (representative)
        first_frame = frames[0]

        # ── Spatial forensics ─────────────────────────────────────────────────
        logger.info("Coordinator: dispatching SpatialForensicsAgent (video keyframe)")
        results.append(spatial.run(first_frame))

        # ── Frequency analysis ────────────────────────────────────────────────
        logger.info("Coordinator: dispatching FrequencyAnalysisAgent (video keyframe)")
        results.append(frequency.run(first_frame))

        # ── Temporal consistency ──────────────────────────────────────────────
        logger.info("Coordinator: dispatching TemporalConsistencyAgent")
        results.append(temporal.run(frames))

        # ── Biological signal (rPPG) ──────────────────────────────────────────
        logger.info("Coordinator: dispatching BiologicalSignalAgent")
        results.append(biological.run(frames))

        # ── Audio-visual sync ─────────────────────────────────────────────────
        logger.info("Coordinator: dispatching AudioVisualSyncAgent")
        results.append(audio_visual.run(frames, video_path=tmp_path))

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = fusion.fuse(results, use_quantum=True)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return ForensicVerdict(
        verdict=fused["verdict"],
        deepfake_probability=fused["deepfake_probability"],
        confidence_in_verdict=fused["confidence_in_verdict"],
        agent_results=results,
        fusion_weights=fused["fusion_weights"],
        rationale=fused["rationale"],
        processing_time_s=round(time.time() - t0, 2),
        media_type="video",
        warnings=warnings,
    )


def route(
    data: bytes,
    filename: str,
) -> ForensicVerdict:
    """Top-level router — chooses image or video pipeline based on filename."""
    if _is_video(filename):
        return analyse_video(data, filename)
    elif Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
        return analyse_image(data, filename)
    else:
        # Unknown type — log coverage gap, attempt as image
        logger.warning(
            "Coordinator: unknown media type '%s' — attempting as image. "
            "Autonomous tool synthesis opportunity: register a specialist agent for this type.",
            Path(filename).suffix,
        )
        return analyse_image(data, filename)
