"""
Facial Landmark & Anatomical Consistency Agent — Obscuro Deepimage.

Deepfake-generated faces often show subtle anatomical irregularities:
  - Asymmetric eye aspect ratios beyond natural variation
  - Face symmetry violations (blend boundary artifacts)
  - Unnatural facial proportion indices
  - Contour jitter at synthesis/blend boundaries
  - Iris shape anomalies (AI images often have malformed irises)

Methods:
  1. MediaPipe Face Mesh (478 landmarks, required) for full analysis
  2. OpenCV Haar cascade fallback for basic eye-region proportions

References:
  - Yang et al., "Exposing DeepFakes Using Inconsistent Head Poses", ICASSP 2019
  - Li et al., "Exposing DeepFake Videos By Detecting Face Warping Artifacts", CVPR 2019
  - Matern et al., "Exploiting Visual Artifacts to Expose Deepfakes", WACV 2019
"""
import logging
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List

from app.schemas import AgentResult

logger = logging.getLogger(__name__)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _ear(landmarks: np.ndarray) -> float:
    """Eye Aspect Ratio from 6 (x,y) points [corner, top1, top2, corner, bot2, bot1]."""
    v1 = float(np.linalg.norm(landmarks[1] - landmarks[5]))
    v2 = float(np.linalg.norm(landmarks[2] - landmarks[4]))
    h = float(np.linalg.norm(landmarks[0] - landmarks[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


def _contour_smoothness(pts: np.ndarray) -> float:
    """
    Measure direction-change variance along a landmark contour.
    Real faces: smooth curves. Blend artifacts: local direction spikes.
    Returns normalised jitter in [0, 1], higher = more jagged.
    """
    if len(pts) < 3:
        return 0.0
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    changes = np.diff(angles)
    # Wrap to [-π, π]
    changes = (changes + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(np.std(np.abs(changes)) / 1.0, 0.0, 1.0))


def _face_symmetry(pts: np.ndarray, center_x: float, face_w: float) -> float:
    """
    Compare left/right jaw contours by mirroring.
    Returns normalised asymmetry [0,1]; higher = more asymmetric.
    """
    left_jaw  = pts[[234, 93, 132, 58, 172, 136, 150, 149, 176, 148]]
    right_jaw = pts[[454, 323, 361, 288, 397, 365, 379, 378, 400, 377]]
    mirrored_x = 2 * center_x - left_jaw[:, 0]
    diff = np.abs(mirrored_x - right_jaw[:, 0])
    return float(np.clip(diff.mean() / (face_w + 1e-6), 0.0, 1.0))


# ── MediaPipe analysis ─────────────────────────────────────────────────────────

def _run_mediapipe(image_bgr: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    import mediapipe as mp
    details: Dict[str, Any] = {}

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_mesh = mp.solutions.face_mesh
    with mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.45,
    ) as mesh:
        res = mesh.process(rgb)

    if not res.multi_face_landmarks:
        details["face_detected"] = False
        details["note"] = "No face detected by MediaPipe Face Mesh."
        return 0.5, 0.1, details

    details["face_detected"] = True
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)  # (478, 2)

    # ── Eye Aspect Ratios ──────────────────────────────────────────────────────
    # MediaPipe iris-refined landmarks
    # Left eye: 33, 160, 158, 133, 153, 144  Right: 362, 385, 387, 263, 373, 380
    left_eye_pts  = pts[[33, 160, 158, 133, 153, 144]]
    right_eye_pts = pts[[362, 385, 387, 263, 373, 380]]
    ear_l = _ear(left_eye_pts)
    ear_r = _ear(right_eye_pts)
    ear_asymmetry = abs(ear_l - ear_r)
    details["ear_left"]       = round(float(ear_l), 4)
    details["ear_right"]      = round(float(ear_r), 4)
    details["ear_asymmetry"]  = round(float(ear_asymmetry), 4)

    # ── Face Symmetry ──────────────────────────────────────────────────────────
    face_w = float(np.abs(pts[:, 0].max() - pts[:, 0].min()))
    nose_bridge_x = float(pts[6, 0])   # landmark 6 = nose bridge
    sym_score = _face_symmetry(pts, nose_bridge_x, face_w)
    details["face_symmetry_deviation"] = round(float(sym_score), 4)

    # ── Facial Proportions ─────────────────────────────────────────────────────
    # Inter-pupil distance using iris centers (landmarks 468, 473 with refined)
    try:
        ipd = float(np.linalg.norm(pts[468] - pts[473]))
    except IndexError:
        ipd = float(np.linalg.norm(left_eye_pts.mean(0) - right_eye_pts.mean(0)))

    face_h = float(np.linalg.norm(pts[10] - pts[152]))   # forehead → chin
    ipd_ratio = ipd / (face_h + 1e-6)
    # Natural range ~0.28–0.44; outside = suspicious
    proportion_anomaly = float(np.clip(abs(ipd_ratio - 0.36) / 0.12, 0.0, 1.0))
    details["ipd_face_ratio"]       = round(float(ipd_ratio), 4)
    details["proportion_anomaly"]   = round(float(proportion_anomaly), 4)

    # ── Jaw Contour Smoothness ─────────────────────────────────────────────────
    # Landmark indices 0-16: left-to-right jaw contour
    jaw_pts = pts[list(range(0, 17))]
    jaw_jitter = _contour_smoothness(jaw_pts)
    details["jaw_contour_jitter"] = round(float(jaw_jitter), 4)

    # ── Iris regularity (with refined landmarks 468-477) ──────────────────────
    try:
        left_iris  = pts[468:473]   # 5 iris pts left
        right_iris = pts[473:478]   # 5 iris pts right
        def iris_circularity(iris_pts: np.ndarray) -> float:
            centre = iris_pts.mean(0)
            radii = np.linalg.norm(iris_pts - centre, axis=1)
            return float(np.std(radii) / (radii.mean() + 1e-6))
        iris_l = iris_circularity(left_iris)
        iris_r = iris_circularity(right_iris)
        iris_anomaly = (iris_l + iris_r) / 2.0
        details["iris_irregularity"]   = round(float(iris_anomaly), 4)
    except Exception:
        iris_anomaly = 0.0

    # ── Ensemble ───────────────────────────────────────────────────────────────
    # Calibration targets (typical real images):
    #   ear_asymmetry ≤ 0.08    → score 0
    #   sym_score     ≤ 0.04    → score 0
    #   proportion    ≤ 0.25    → score 0  (already normalised)
    #   jaw_jitter    ≤ 0.25    → score 0
    ear_s    = float(np.clip(ear_asymmetry / 0.10, 0.0, 1.0))
    sym_s    = float(np.clip(sym_score     / 0.06, 0.0, 1.0))
    prop_s   = proportion_anomaly
    jaw_s    = float(np.clip(jaw_jitter    / 0.40, 0.0, 1.0))
    iris_s   = float(np.clip(iris_anomaly  / 0.30, 0.0, 1.0))

    combined = (
        0.25 * ear_s
        + 0.25 * sym_s
        + 0.20 * prop_s
        + 0.15 * jaw_s
        + 0.15 * iris_s
    )

    details["ear_anomaly_score"]  = round(float(ear_s), 4)
    details["symmetry_score"]     = round(float(sym_s), 4)
    details["iris_anomaly_score"] = round(float(iris_s), 4)
    details["note"] = (
        f"MediaPipe 478-landmark analysis. "
        f"EAR asymmetry={ear_asymmetry:.3f}, "
        f"face symmetry deviation={sym_score:.3f}, "
        f"iris irregularity={iris_anomaly:.3f}."
    )

    confidence = 0.62  # MediaPipe results are reliable when face is detected
    return float(np.clip(combined, 0.0, 1.0)), confidence, details


# ── OpenCV Haar fallback ───────────────────────────────────────────────────────

def _run_opencv_fallback(image_bgr: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    """
    Minimal fallback using OpenCV eye detection for EAR asymmetry only.
    Much less accurate than MediaPipe.
    """
    details: Dict[str, Any] = {}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        details["face_detected"] = False
        details["note"] = "No face detected (Haar fallback)."
        return 0.5, 0.1, details

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y + fh, x:x + fw]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20, 20))

    details["face_detected"] = True
    details["eyes_found"] = int(len(eyes))
    details["note"] = "OpenCV Haar fallback (MediaPipe unavailable). Limited accuracy."

    if len(eyes) < 2:
        return 0.5, 0.15, details

    # Sort eyes by x position
    eyes = sorted(eyes, key=lambda e: e[0])
    ey1, ey2 = eyes[0], eyes[1]
    # Compare eye widths as a crude EAR proxy
    w_ratio = float(ey1[2]) / (float(ey2[2]) + 1e-6)
    asymmetry = abs(1.0 - w_ratio)
    details["eye_width_ratio"] = round(float(w_ratio), 3)
    score = float(np.clip(asymmetry / 0.3, 0.0, 1.0))
    return score, 0.20, details


# ── Public entry point ─────────────────────────────────────────────────────────

def run(image_bgr: np.ndarray) -> AgentResult:
    """
    Run facial landmark & anatomical consistency analysis.
    Returns AgentResult with anatomical_anomaly_score in [0, 1].
    Higher score = more anatomical irregularities = more likely deepfake.
    """
    try:
        try:
            score, confidence, details = _run_mediapipe(image_bgr)
        except ImportError:
            logger.warning("MediaPipe unavailable — using OpenCV Haar fallback.")
            score, confidence, details = _run_opencv_fallback(image_bgr)

        return AgentResult(
            agent_name="FacialLandmarksAgent",
            signal_name="anatomical_anomaly_score",
            score=score,
            confidence=confidence,
            details=details,
        )

    except Exception as exc:
        logger.error("FacialLandmarksAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="FacialLandmarksAgent",
            signal_name="anatomical_anomaly_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
