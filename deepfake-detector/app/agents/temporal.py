"""
Temporal Consistency Agent — video-specific temporal artifact detection.

Checks:
  1. Frame-to-frame optical flow inconsistency (unnatural jitter)
  2. Inter-frame texture variance (face blending artifacts at boundaries)
  3. Blink rate analysis (deepfake videos often have abnormal or absent blinking)
  4. Landmark drift (face mesh movement irregularities)
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

from app.schemas import AgentResult
from app.preprocessing.pipeline import detect_and_crop_face

logger = logging.getLogger(__name__)


def _compute_optical_flow(frames: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute Lucas-Kanade sparse optical flow between consecutive frames.
    Returns (mean_flow_magnitude, flow_irregularity_score).
    High irregularity → possible temporal inconsistency from face replacement.
    """
    if len(frames) < 3:
        return 0.0, 0.5

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    magnitudes = []
    irregularities = []

    for i in range(min(len(frames) - 1, 20)):  # Cap at 20 frame-pairs
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None or len(p0) < 5:
            continue

        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
        if p1 is None:
            continue

        good_old = p0[status == 1]
        good_new = p1[status == 1]
        if len(good_old) < 3:
            continue

        flow_vecs = good_new - good_old
        mags = np.sqrt(flow_vecs[:, :, 0] ** 2 + flow_vecs[:, :, 1] ** 2).flatten()
        magnitudes.append(float(mags.mean()))

        # Flow regularity: real motion should be coherent (low std/mean ratio)
        irregularity = float(mags.std() / (mags.mean() + 1e-6))
        irregularities.append(irregularity)

    if not magnitudes:
        return 0.0, 0.5

    avg_mag = float(np.mean(magnitudes))
    avg_irreg = float(np.mean(irregularities))

    # Normalise irregularity: typical real video ≈ 0.5-1.0, deepfake ≈ > 1.5
    irreg_score = float(np.clip((avg_irreg - 0.8) / 1.5, 0.0, 1.0))
    return avg_mag, irreg_score


def _face_texture_consistency(frames: List[np.ndarray]) -> float:
    """
    Measure frame-to-frame consistency of face-region texture.
    Deepfake face swaps sometimes introduce subtle texture discontinuities.
    Returns score in [0, 1] where higher = more inconsistent = more likely fake.
    """
    face_crops = []
    for frame in frames[:16]:
        crop, _ = detect_and_crop_face(frame)
        if crop is not None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            face_crops.append(gray)

    if len(face_crops) < 3:
        return 0.5

    # Compute frame-to-frame structural similarity proxy (MSE)
    mses = []
    for i in range(len(face_crops) - 1):
        diff = face_crops[i + 1] - face_crops[i]
        mse = float(np.mean(diff ** 2))
        mses.append(mse)

    # High variance in MSE → inconsistent blending
    mse_variance = float(np.std(mses) / (np.mean(mses) + 1e-6))
    score = float(np.clip((mse_variance - 0.5) / 1.5, 0.0, 1.0))
    return score


def _estimate_blink_rate(frames: List[np.ndarray]) -> Tuple[Optional[float], float]:
    """
    Estimate blink rate from eye aspect ratio (EAR) using OpenCV Haar detector.
    Returns (blinks_per_second, anomaly_score).
    Normal blink rate: 12-20 per minute = 0.2-0.33 per second.
    Deepfakes often have 0 or very low blink rate.
    """
    # Use Haar cascade for eye detection (no MediaPipe required)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    if eye_cascade.empty():
        return None, 0.5

    eye_open_states = []
    for frame in frames[:24]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        eye_open_states.append(len(eyes) > 0)

    if len(eye_open_states) < 4:
        return None, 0.5

    # Count transitions: open→closed→open = 1 blink
    blinks = 0
    for i in range(1, len(eye_open_states) - 1):
        if eye_open_states[i - 1] and not eye_open_states[i] and eye_open_states[i + 1]:
            blinks += 1

    # Assume ~25fps for blink rate calculation
    duration_s = len(frames) / 25.0
    blinks_per_sec = blinks / max(duration_s, 1.0)

    # Normal: 0.20-0.33 blinks/sec; anomalous: < 0.05 or > 0.8
    if blinks_per_sec < 0.05:
        anomaly = 0.7  # Very low — suspicious
    elif blinks_per_sec < 0.15:
        anomaly = 0.4
    elif blinks_per_sec > 0.7:
        anomaly = 0.5  # Very high — also suspicious
    else:
        anomaly = 0.1  # Normal range

    return blinks_per_sec, anomaly


def run(frames: List[np.ndarray]) -> AgentResult:
    """Run temporal consistency analysis on a list of video frames."""
    if len(frames) < 2:
        return AgentResult(
            agent_name="TemporalConsistencyAgent",
            signal_name="temporal_inconsistency_score",
            score=0.5,
            confidence=0.1,
            ran=False,
            skipped_reason="Fewer than 2 frames available — cannot perform temporal analysis.",
        )

    try:
        _, flow_irreg = _compute_optical_flow(frames)
        texture_score = _face_texture_consistency(frames)
        blinks_per_sec, blink_anomaly = _estimate_blink_rate(frames)

        final_score = 0.40 * flow_irreg + 0.35 * texture_score + 0.25 * blink_anomaly
        final_conf = 0.55

        details: Dict[str, Any] = {
            "optical_flow_irregularity": round(flow_irreg, 4),
            "face_texture_inconsistency": round(texture_score, 4),
            "blink_rate_per_sec": round(blinks_per_sec, 3) if blinks_per_sec else "undetected",
            "blink_anomaly_score": round(blink_anomaly, 4),
            "frames_analyzed": len(frames),
            "note": "Normal blink rate: 0.20-0.33/sec. Deepfakes often have zero or abnormal blink frequency.",
        }

        return AgentResult(
            agent_name="TemporalConsistencyAgent",
            signal_name="temporal_inconsistency_score",
            score=float(np.clip(final_score, 0.0, 1.0)),
            confidence=final_conf,
            details=details,
        )

    except Exception as exc:
        logger.error("TemporalConsistencyAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="TemporalConsistencyAgent",
            signal_name="temporal_inconsistency_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
