"""
Coordinator Agent — the top-level orchestrator for Obscuro Deepimage.

Responsibilities:
  1. Receive media (image bytes or video path)
  2. Determine media type and applicable specialist agents
  3. Dispatch agents in PARALLEL (ThreadPoolExecutor) for image analysis
  4. Detect coverage gaps and log them for future tool synthesis
  5. Detect inter-agent conflict and surface it in warnings
  6. Invoke Fusion Agent and return a ForensicVerdict

Architecture pattern: parallel planner-executor with conflict detection.
All specialist agents are independent; the coordinator dispatches them
concurrently and only waits at the fusion step.
"""
import logging
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

import numpy as np

from app.schemas import AgentResult, ForensicVerdict
from app.agents import spatial, frequency, temporal, biological, audio_visual, fusion
from app.agents import exif_metadata, facial_landmarks
from app.preprocessing.pipeline import (
    load_image_bytes,
    extract_video_frames,
    get_face_crops,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Agent timeout in seconds (generous — first run downloads HF model)
_AGENT_TIMEOUT = 120


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def _detect_conflicts(results: List[AgentResult]) -> List[str]:
    """
    Detect strong inter-agent disagreements.
    If two agents with confidence ≥ 0.45 disagree by more than 0.50 in score,
    flag the conflict so the user knows the evidence is contested.
    """
    active = [r for r in results if r.ran and r.confidence >= 0.45]
    conflicts = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if abs(a.score - b.score) >= 0.50:
                direction_a = "FAKE" if a.score > 0.5 else "REAL"
                direction_b = "FAKE" if b.score > 0.5 else "REAL"
                conflicts.append(
                    f"Agent conflict: {a.agent_name} ({direction_a}, score={a.score:.2f}) "
                    f"vs {b.agent_name} ({direction_b}, score={b.score:.2f}). "
                    "Independent signals disagree — treat verdict with caution."
                )
    return conflicts


def _safe_run(fn, *args, name: str = "agent") -> AgentResult:
    """Run an agent function safely, returning a neutral result on error."""
    try:
        return fn(*args)
    except Exception as exc:
        logger.error("Agent %s failed unexpectedly: %s", name, exc, exc_info=True)
        return AgentResult(
            agent_name=name,
            signal_name="error",
            score=0.5,
            confidence=0.0,
            ran=False,
            skipped_reason=f"Agent error: {exc}",
            details={"error": str(exc)},
        )


def analyse_image(
    image_data: bytes,
    filename: str = "upload.jpg",
) -> ForensicVerdict:
    """
    Full parallel pipeline for a single static image.

    Agents dispatched concurrently:
      - SpatialForensicsAgent     (pixel/texture artifacts, ViT + EfficientNet)
      - FrequencyAnalysisAgent    (FFT/DCT/SRM spectral fingerprints)
      - FacialLandmarksAgent      (anatomical irregularity, MediaPipe 478-point)
      - EXIFMetadataAgent         (metadata consistency, AI software tags)

    Video-only agents are skipped with a clear reason.
    """
    t0 = time.time()
    warnings: List[str] = []

    try:
        image_bgr = load_image_bytes(image_data)
    except Exception as exc:
        logger.error("Could not decode image: %s", exc)
        raise ValueError(f"Image decode failed: {exc}") from exc

    # ── Parallel dispatch of image agents ─────────────────────────────────────
    logger.info("Coordinator: dispatching 4 image agents in parallel")

    agent_tasks = {
        "SpatialForensicsAgent":  (spatial.run,          image_bgr),
        "FrequencyAnalysisAgent": (frequency.run,         image_bgr),
        "FacialLandmarksAgent":   (facial_landmarks.run,  image_bgr),
        "EXIFMetadataAgent":      (exif_metadata.run,     image_data),
    }

    results: List[AgentResult] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_to_name = {
            pool.submit(_safe_run, fn, *args, name=name): name
            for name, (fn, *args) in agent_tasks.items()
        }
        for future in as_completed(future_to_name, timeout=_AGENT_TIMEOUT):
            name = future_to_name[future]
            try:
                results.append(future.result())
                logger.info("Coordinator: %s completed", name)
            except Exception as exc:
                logger.error("Coordinator: %s raised: %s", name, exc)
                results.append(AgentResult(
                    agent_name=name,
                    signal_name="error",
                    score=0.5, confidence=0.0,
                    ran=False, skipped_reason=str(exc),
                ))

    # ── Video-only agents (not applicable for images) ─────────────────────────
    for agent_name, signal_name, reason in [
        ("TemporalConsistencyAgent", "temporal_inconsistency_score",
         "Static image — temporal analysis requires a video frame sequence."),
        ("BiologicalSignalAgent", "rppg_anomaly_score",
         "Static image — rPPG heartbeat detection requires video."),
        ("AudioVisualSyncAgent", "lip_sync_mismatch_score",
         "Static image — audio-visual sync requires video with audio track."),
    ]:
        results.append(AgentResult(
            agent_name=agent_name,
            signal_name=signal_name,
            score=0.5, confidence=0.0,
            ran=False, skipped_reason=reason,
        ))

    # ── Conflict detection ────────────────────────────────────────────────────
    conflicts = _detect_conflicts(results)
    warnings.extend(conflicts)

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
    Image-domain agents run in parallel on the first keyframe.
    Video-domain agents run sequentially (they each process the full frame list).
    """
    t0 = time.time()
    results: List[AgentResult] = []
    warnings: List[str] = []

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
                f"Only {len(frames)} frames extracted — some temporal analyses may be less reliable."
            )

        first_frame = frames[0]

        # Encode first frame back to JPEG bytes for EXIF agent (frame has no EXIF, flag it)
        import cv2
        _, jpeg_bytes = cv2.imencode(".jpg", first_frame)
        frame_bytes = jpeg_bytes.tobytes()

        # ── Parallel image-domain agents on keyframe ──────────────────────────
        logger.info("Coordinator: dispatching image-domain agents on video keyframe")
        image_tasks = {
            "SpatialForensicsAgent":  (spatial.run,         first_frame),
            "FrequencyAnalysisAgent": (frequency.run,        first_frame),
            "FacialLandmarksAgent":   (facial_landmarks.run, first_frame),
            "EXIFMetadataAgent":      (exif_metadata.run,    frame_bytes),
        }
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_name = {
                pool.submit(_safe_run, fn, *args, name=name): name
                for name, (fn, *args) in image_tasks.items()
            }
            for future in as_completed(future_to_name, timeout=_AGENT_TIMEOUT):
                name = future_to_name[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(AgentResult(
                        agent_name=name, signal_name="error",
                        score=0.5, confidence=0.0,
                        ran=False, skipped_reason=str(exc),
                    ))

        # ── Sequential video-domain agents ────────────────────────────────────
        logger.info("Coordinator: dispatching video-domain agents")
        results.append(_safe_run(temporal.run,     frames,                name="TemporalConsistencyAgent"))
        results.append(_safe_run(biological.run,   frames,                name="BiologicalSignalAgent"))
        results.append(_safe_run(audio_visual.run, frames, tmp_path,      name="AudioVisualSyncAgent"))

        # ── Conflict detection ─────────────────────────────────────────────────
        conflicts = _detect_conflicts(results)
        warnings.extend(conflicts)

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


def route(data: bytes, filename: str) -> ForensicVerdict:
    """Top-level router — choose image or video pipeline."""
    if _is_video(filename):
        return analyse_video(data, filename)
    elif Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
        return analyse_image(data, filename)
    else:
        logger.warning(
            "Unknown media type '%s' — attempting as image. "
            "Coverage gap logged for potential future agent synthesis.",
            Path(filename).suffix,
        )
        return analyse_image(data, filename)
