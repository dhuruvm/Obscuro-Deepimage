"""
Spatial Forensics Agent — pixel-level artifact detection.

Uses:
  1. HuggingFace deepfake-specific ViT model (primary, if available)
  2. timm EfficientNet-B4 feature extractor (fallback / secondary signal)
  3. JPEG blocking artifact analysis (low-level texture heuristic)
  4. GAN fingerprint analysis via autocorrelation residuals
"""
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any

from app.schemas import AgentResult
from app.models import detector
from app.preprocessing.pipeline import (
    preprocess_image_for_model,
    detect_and_crop_face,
    FACE_CROP_SIZE,
)

logger = logging.getLogger(__name__)


def _jpeg_artifact_score(image_bgr: np.ndarray) -> float:
    """
    Estimate JPEG blocking artifacts. High-quality synthetic images often lack
    the DCT-block structure of real JPEG-compressed photographs.
    Returns score in [0, 1] where higher = more artifact-free (suspicious for fake).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    block_diffs = []
    for y in range(8, h - 8, 8):
        row_diff = np.abs(gray[y, :] - gray[y - 1, :]).mean()
        block_diffs.append(row_diff)
    for x in range(8, w - 8, 8):
        col_diff = np.abs(gray[:, x] - gray[:, x - 1]).mean()
        block_diffs.append(col_diff)

    if not block_diffs:
        return 0.5

    # Low blocking variance → image may be synthetically generated
    avg_block = float(np.mean(block_diffs))
    # Typical real JPEG: 3-8; synthetics often < 2 or very high if over-compressed
    # Normalize: close to 0 = very clean (suspicious), mid-range = natural
    score = float(np.clip(1.0 - (avg_block / 6.0), 0.0, 1.0))
    return score * 0.6  # Low weight — weak signal alone


def _gan_fingerprint_score(image_bgr: np.ndarray) -> float:
    """
    Simplified GAN fingerprint detection via high-frequency residual autocorrelation.
    GAN generators introduce periodic patterns in the noise floor.
    Returns score in [0, 1] where higher = more GAN-like.
    Reference: Frank et al., "Leveraging Frequency Analysis for Deep Fake
    Image Recognition", ICML 2020.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Extract noise residual via Gaussian subtraction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred

    # Compute autocorrelation of residual
    residual_norm = residual - residual.mean()
    autocorr = np.fft.fft2(residual_norm)
    power = np.abs(autocorr) ** 2
    power_log = np.log1p(np.fft.fftshift(power))

    h, w = power_log.shape
    center_region = power_log[h // 4:3 * h // 4, w // 4:3 * w // 4]
    edge_region = power_log - 0
    center_energy = center_region.mean()
    total_energy = power_log.mean() + 1e-6

    # High center-to-total energy ratio → more natural image
    # Low ratio + strong edge peaks → potential GAN fingerprint
    ratio = center_energy / total_energy
    score = float(np.clip(1.0 - ratio * 1.5, 0.0, 1.0))
    return score


def run(image_bgr: np.ndarray) -> AgentResult:
    """
    Run spatial forensics on a single BGR image.
    Returns an AgentResult with deepfake_score in [0, 1].
    """
    details: Dict[str, Any] = {}

    try:
        face_crop, bbox = detect_and_crop_face(image_bgr)
        details["face_detected"] = bbox is not None
        work_img = face_crop if face_crop is not None else image_bgr

        # ── Signal 1: HuggingFace ViT deepfake classifier ──────────────────
        pil_img = Image.fromarray(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        hf_result = detector.spatial_score_from_hf(pil_img)

        if hf_result is not None:
            hf_score, hf_conf = hf_result
            details["hf_deepfake_score"] = round(hf_score, 4)
            details["hf_confidence"] = round(hf_conf, 4)
            details["model"] = "ViT (HuggingFace dima806/deepfake_vs_real_image_detection)"
        else:
            hf_score, hf_conf = None, 0.0
            details["model"] = "EfficientNet-B4 (timm, ImageNet pretrained — not fine-tuned on deepfakes)"

        # ── Signal 2: timm EfficientNet texture features ────────────────────
        tensor = preprocess_image_for_model(work_img)
        timm_score, timm_conf = detector.spatial_score_from_timm(tensor)
        details["timm_texture_score"] = round(timm_score, 4)
        details["timm_confidence"] = round(timm_conf, 4)

        # ── Signal 3: JPEG artifact analysis ────────────────────────────────
        jpeg_score = _jpeg_artifact_score(work_img)
        details["jpeg_artifact_score"] = round(jpeg_score, 4)

        # ── Signal 4: GAN fingerprint ────────────────────────────────────────
        gan_score = _gan_fingerprint_score(work_img)
        details["gan_fingerprint_score"] = round(gan_score, 4)

        # ── Ensemble within spatial agent ────────────────────────────────────
        if hf_score is not None:
            # HF model is reliable — weight it most heavily
            final_score = (
                0.55 * hf_score
                + 0.20 * timm_score
                + 0.10 * jpeg_score
                + 0.15 * gan_score
            )
            final_conf = 0.6 + 0.4 * hf_conf
        else:
            # Fallback ensemble without HF model
            final_score = (
                0.35 * timm_score
                + 0.25 * jpeg_score
                + 0.40 * gan_score
            )
            final_conf = 0.4  # Lower confidence without specialist model

        details["ensemble_note"] = (
            "Primary: HuggingFace ViT + secondary texture/frequency heuristics"
            if hf_score is not None
            else "WARNING: Specialist deepfake model unavailable — heuristics only; accuracy limited."
        )

        return AgentResult(
            agent_name="SpatialForensicsAgent",
            signal_name="pixel_artifact_score",
            score=float(np.clip(final_score, 0.0, 1.0)),
            confidence=float(np.clip(final_conf, 0.0, 1.0)),
            details=details,
        )

    except Exception as exc:
        logger.error("SpatialForensicsAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="SpatialForensicsAgent",
            signal_name="pixel_artifact_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
