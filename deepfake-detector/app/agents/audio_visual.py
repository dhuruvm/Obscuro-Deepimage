"""
Audio-Visual Sync Agent — lip-sync mismatch detection.

Cross-modal deepfakes (audio swap, talking-head synthesis) introduce subtle
desynchronisation between lip movement and audio energy.

Method: correlate lip-region visual motion energy with audio amplitude envelope.
High mismatch between visual lip activity and audio energy → suspicious.

NOTE: requires audio track extraction (ffmpeg). If no audio, agent is skipped.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import cv2

from app.schemas import AgentResult
from app.preprocessing.pipeline import detect_and_crop_face

logger = logging.getLogger(__name__)


def _extract_audio_amplitude(video_path: str, duration_s: float) -> Optional[np.ndarray]:
    """
    Extract mono audio amplitude envelope from video using ffmpeg.
    Returns amplitude array or None if no audio / ffmpeg unavailable.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            tmp_path = f.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                      # No video
                "-acodec", "pcm_s16le",
                "-ar", "1000",              # Downsample to 1000 Hz for efficiency
                "-ac", "1",                 # Mono
                "-f", "s16le",
                tmp_path,
            ],
            capture_output=True, timeout=30,
        )

        if result.returncode != 0:
            logger.debug("ffmpeg audio extraction returned non-zero: %s", result.stderr[:200])
            return None

        audio_data = np.frombuffer(Path(tmp_path).read_bytes(), dtype=np.int16).astype(np.float32)
        os.unlink(tmp_path)

        if len(audio_data) < 100:
            return None

        # Compute amplitude envelope (RMS in 40ms windows at 1000Hz = 40 samples)
        window = 40
        n_windows = len(audio_data) // window
        rms = np.array([
            np.sqrt(np.mean(audio_data[i * window:(i + 1) * window] ** 2))
            for i in range(n_windows)
        ])
        return rms

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as exc:
        logger.debug("Audio extraction failed: %s", exc)
        return None


def _extract_lip_motion(frames: List[np.ndarray]) -> np.ndarray:
    """
    Estimate lip motion energy per frame from lower-face region.
    Returns array of per-frame motion magnitudes.
    """
    motion_energies = []
    prev_lip = None

    for frame in frames:
        crop, _ = detect_and_crop_face(frame)
        work = crop if crop is not None else cv2.resize(frame, (224, 224))
        h, w = work.shape[:2]
        # Lower third of face = lip region
        lip_region = cv2.cvtColor(
            work[int(h * 0.6):, w // 4:3 * w // 4],
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        if prev_lip is not None and lip_region.shape == prev_lip.shape:
            diff = np.abs(lip_region - prev_lip)
            motion_energies.append(float(diff.mean()))
        else:
            motion_energies.append(0.0)

        prev_lip = lip_region

    return np.array(motion_energies, dtype=np.float32)


def _correlation_score(audio_env: np.ndarray, lip_motion: np.ndarray) -> Tuple[float, float]:
    """
    Compute normalised cross-correlation between audio amplitude and lip motion.
    Returns (correlation, anomaly_score).
    """
    n = min(len(audio_env), len(lip_motion))
    if n < 8:
        return 0.0, 0.5

    # Resample to same length
    if len(audio_env) != n:
        indices = np.linspace(0, len(audio_env) - 1, n).astype(int)
        audio_env = audio_env[indices]
    lip_motion_trimmed = lip_motion[:n]

    # Normalise
    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    corr = float(np.dot(norm(audio_env), norm(lip_motion_trimmed)) / n)

    # Real speech: correlation > 0.4; silent video: both near zero (acceptable)
    # Deepfake with audio swap: low/negative correlation
    audio_active = audio_env.mean() > audio_env.std() * 0.5

    if not audio_active:
        anomaly = 0.2  # Silent video — inconclusive
    elif corr > 0.5:
        anomaly = 0.1  # Good lip-audio sync
    elif corr > 0.2:
        anomaly = 0.35
    elif corr > 0.0:
        anomaly = 0.55
    else:
        anomaly = 0.75  # Poor or negative correlation — suspicious

    return corr, anomaly


def run(frames: List[np.ndarray], video_path: Optional[str] = None) -> AgentResult:
    """Run audio-visual sync analysis."""
    if video_path is None or not Path(video_path).exists():
        return AgentResult(
            agent_name="AudioVisualSyncAgent",
            signal_name="lip_sync_mismatch_score",
            score=0.5,
            confidence=0.05,
            ran=False,
            skipped_reason="No video path provided — audio-visual sync analysis unavailable for images.",
        )

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_s = total_frames / fps

        lip_motion = _extract_lip_motion(frames)
        audio_env = _extract_audio_amplitude(video_path, duration_s)

        if audio_env is None or audio_env.mean() < 1.0:
            return AgentResult(
                agent_name="AudioVisualSyncAgent",
                signal_name="lip_sync_mismatch_score",
                score=0.5,
                confidence=0.05,
                ran=False,
                skipped_reason="No audio track detected or audio is silent — lip-sync analysis skipped.",
            )

        corr, anomaly = _correlation_score(audio_env, lip_motion)
        confidence = 0.5  # Moderate confidence

        details: Dict[str, Any] = {
            "audio_lip_correlation": round(corr, 4),
            "audio_frames_analysed": len(audio_env),
            "video_frames_analysed": len(frames),
            "method": "RMS audio envelope vs lip-region motion energy cross-correlation",
            "caveat": "Audio-visual sync analysis is approximate; works best on clear speech at ≥ 15fps.",
        }

        return AgentResult(
            agent_name="AudioVisualSyncAgent",
            signal_name="lip_sync_mismatch_score",
            score=float(np.clip(anomaly, 0.0, 1.0)),
            confidence=confidence,
            details=details,
        )

    except Exception as exc:
        logger.error("AudioVisualSyncAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="AudioVisualSyncAgent",
            signal_name="lip_sync_mismatch_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
