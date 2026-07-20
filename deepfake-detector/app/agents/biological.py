"""
Biological Signal Agent — remote photoplethysmography (rPPG) analysis.

Real faces contain subtle colour pulsations from blood flow (heartbeat signal)
detectable in the green channel of video frames. Deepfake-generated faces
typically lack this physiological signal or present it with anomalous periodicity.

Method: CHROM-based rPPG (Haan & Jeanne, TBME 2013) simplified to spatial
colour channel variance tracking across face ROI over time.

NOTE: rPPG is a weak signal in short clips and low-resolution video.
Confidence is intentionally kept low unless a strong periodic signal is found.
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

from app.schemas import AgentResult
from app.preprocessing.pipeline import detect_and_crop_face

logger = logging.getLogger(__name__)

MIN_FRAMES_FOR_RPPG = 15
EXPECTED_HR_RANGE_BPM = (45, 180)   # Physiologically plausible range
ASSUMED_FPS = 25.0


def _extract_face_rgb_signal(frames: List[np.ndarray]) -> np.ndarray:
    """
    Extract mean R, G, B values from face ROI across frames.
    Returns array of shape (N, 3).
    """
    rgb_vals = []
    for frame in frames:
        crop, _ = detect_and_crop_face(frame)
        work = crop if crop is not None else cv2.resize(frame, (224, 224))
        # Use centre 50% of face to avoid edges / hair
        h, w = work.shape[:2]
        roi = work[h // 4:3 * h // 4, w // 4:3 * w // 4]
        mean_rgb = roi.reshape(-1, 3).mean(axis=0)  # B, G, R in OpenCV
        rgb_vals.append([mean_rgb[2], mean_rgb[1], mean_rgb[0]])  # R, G, B
    return np.array(rgb_vals, dtype=np.float64)


def _chrom_rppg(rgb_signal: np.ndarray, fps: float) -> Tuple[float, float, float]:
    """
    CHROM-based rPPG signal extraction.
    Returns (dominant_freq_hz, signal_snr, anomaly_score).
    """
    if len(rgb_signal) < MIN_FRAMES_FOR_RPPG:
        return 0.0, 0.0, 0.5

    R, G, B = rgb_signal[:, 0], rgb_signal[:, 1], rgb_signal[:, 2]

    # Normalise channels
    def norm_channel(c):
        return (c - c.mean()) / (c.std() + 1e-8)

    Rn, Gn, Bn = norm_channel(R), norm_channel(G), norm_channel(B)

    # CHROM chrominance channels
    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    alpha = X.std() / (Y.std() + 1e-8)
    pulse = X - alpha * Y

    # Bandpass filter to heartbeat frequencies
    fft_pulse = np.fft.rfft(pulse)
    freqs = np.fft.rfftfreq(len(pulse), d=1.0 / fps)
    low, high = EXPECTED_HR_RANGE_BPM[0] / 60.0, EXPECTED_HR_RANGE_BPM[1] / 60.0
    band_mask = (freqs >= low) & (freqs <= high)
    out_of_band_mask = ~band_mask

    power_in_band = np.abs(fft_pulse[band_mask]) ** 2
    power_out_band = np.abs(fft_pulse[out_of_band_mask]) ** 2

    if power_in_band.sum() == 0:
        return 0.0, 0.0, 0.7  # No heartbeat signal — suspicious

    # SNR: ratio of heartbeat band power to noise floor
    snr = float(power_in_band.sum() / (power_out_band.sum() + 1e-8))

    # Dominant frequency
    peak_idx = np.argmax(power_in_band)
    dom_freq = float(freqs[band_mask][peak_idx])

    # Anomaly scoring:
    # Real face: SNR > 2.0, dom_freq in 0.75-3.0 Hz (45-180 bpm)
    # Deepfake: SNR < 0.5 (no signal) or SNR very high with perfect periodicity
    if snr < 0.3:
        anomaly = 0.75  # Very weak signal — likely synthetic
    elif snr < 1.0:
        anomaly = 0.45
    elif snr > 20.0:
        anomaly = 0.60  # Suspiciously perfect — possibly injected signal
    else:
        anomaly = 0.15  # Normal physiological signal present

    return dom_freq, snr, anomaly


def run(frames: List[np.ndarray]) -> AgentResult:
    """Run rPPG-based biological signal analysis."""
    if len(frames) < MIN_FRAMES_FOR_RPPG:
        return AgentResult(
            agent_name="BiologicalSignalAgent",
            signal_name="rppg_anomaly_score",
            score=0.5,
            confidence=0.1,
            ran=False,
            skipped_reason=(
                f"Requires ≥{MIN_FRAMES_FOR_RPPG} frames; got {len(frames)}. "
                "rPPG analysis needs ~0.6s of video at 25fps."
            ),
        )

    try:
        rgb_signal = _extract_face_rgb_signal(frames)
        dom_freq, snr, anomaly = _chrom_rppg(rgb_signal, fps=ASSUMED_FPS)

        bpm_estimate = dom_freq * 60.0 if dom_freq > 0 else None

        # Confidence is intentionally low — rPPG is noisy in short clips
        confidence = float(np.clip(0.2 + 0.3 * min(snr / 5.0, 1.0), 0.1, 0.5))

        details: Dict[str, Any] = {
            "dominant_freq_hz": round(dom_freq, 3),
            "estimated_bpm": round(bpm_estimate, 1) if bpm_estimate else "not detected",
            "signal_snr": round(snr, 3),
            "frames_used": len(frames),
            "method": "CHROM-based rPPG (Haan & Jeanne 2013)",
            "caveat": (
                "rPPG is a weak signal and unreliable for clips < 3s or resolution < 480p. "
                "This score carries intentionally low confidence weight."
            ),
        }

        return AgentResult(
            agent_name="BiologicalSignalAgent",
            signal_name="rppg_anomaly_score",
            score=float(np.clip(anomaly, 0.0, 1.0)),
            confidence=confidence,
            details=details,
        )

    except Exception as exc:
        logger.error("BiologicalSignalAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="BiologicalSignalAgent",
            signal_name="rppg_anomaly_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
