"""
EXIF & Metadata Forensics Agent — Obscuro Deepimage.

AI-generated images systematically differ from real photographs in their
embedded metadata:
  - Real cameras produce EXIF with make/model, focal length, aperture, ISO, GPS
  - AI generators produce images with absent or minimal EXIF
  - Photoshop / generation tools leave identifiable software tags
  - Metadata inconsistencies (impossible dates, non-physical settings) are red flags
  - PNG images from generative tools carry "parameters" blocks in their text chunks

References:
  - Agarwal & Farid, "Photo Forensics From JPEG Grid Inconsistencies", WIFS 2017
  - Corvi et al., ICASSP 2023 (metadata of diffusion-model outputs)
"""
import logging
from io import BytesIO
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

from app.schemas import AgentResult

# ── Known AI generation software keywords ─────────────────────────────────────
_AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e",
    "generative", "ai generated", "ai-generated", "kling",
    "sora", "flux", "firefly", "imagen", "runway", "pika",
    "synthesia", "deepfake", "faceswap", "roop", "artbreeder",
    "nightcafe", "wombo", "jasper", "fotor", "canva ai",
    "adobe firefly", "invoke ai", "automatic1111", "comfyui",
    "novelai", "dreamstudio", "leonardo.ai", "seaart",
]

# Real camera makes (partial match, lowercase)
_REAL_CAMERA_MAKES = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "phase one", "pentax", "ricoh",
    "apple", "google", "huawei", "xiaomi", "oneplus", "motorola",
    "samsung", "lg electronics", "htc", "dji",
]

# EXIF tag IDs we care about
_TAGS = {
    271: "make",
    272: "model",
    305: "software",
    306: "datetime",
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso_speed",
    37386: "focal_length",
    34853: "gps_info",
    36867: "datetime_original",
    37500: "maker_note",
    37510: "user_comment",
    36868: "datetime_digitized",
    40961: "color_space",
    41495: "sensing_method",
    41728: "file_source",   # 3 = Digital Still Camera
    41729: "scene_type",
}


def _parse_exif(img: Image.Image) -> Dict[str, Any]:
    """Extract EXIF tag dict from a Pillow image."""
    parsed: Dict[str, Any] = {}
    exif_data = None

    try:
        exif_data = img._getexif()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass

    if not exif_data:
        try:
            exif_obj = img.getexif()
            if exif_obj:
                exif_data = dict(exif_obj)
        except Exception:
            pass

    if not exif_data:
        return parsed

    for tag_id, name in _TAGS.items():
        if tag_id in exif_data:
            val = exif_data[tag_id]
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="replace").strip("\x00").strip()
                except Exception:
                    val = repr(val)
            parsed[name] = val

    return parsed


def _analyse_png_metadata(img: Image.Image, details: Dict[str, Any]) -> float:
    """
    Check PNG text chunks for AI generation parameters.
    Tools like Automatic1111 / ComfyUI embed full prompt text in PNG metadata.
    Returns extra fake score contribution (0.0–0.90).
    """
    try:
        info = img.info or {}
        for key in ("parameters", "prompt", "workflow", "generation", "comment"):
            val = info.get(key, "")
            if not val:
                continue
            val_lower = str(val).lower()
            for kw in _AI_SOFTWARE_KEYWORDS + ["steps:", "cfg scale", "sampler", "lora", "checkpoint"]:
                if kw in val_lower:
                    details["png_ai_parameter_found"] = f"key='{key}': {str(val)[:120]}"
                    return 0.92
    except Exception:
        pass
    return 0.0


def run(image_bytes: bytes) -> AgentResult:
    """
    Analyse EXIF and file metadata for signs of AI generation.
    Returns AgentResult with metadata_anomaly_score in [0, 1].
    Higher = more suspicious.
    """
    details: Dict[str, Any] = {}

    try:
        img = Image.open(BytesIO(image_bytes))
        fmt = img.format or "unknown"
        details["format"] = fmt
        details["size"] = f"{img.width}x{img.height}"
        details["mode"] = img.mode

        # ── PNG-specific chunk analysis ────────────────────────────────────
        if fmt == "PNG":
            png_score = _analyse_png_metadata(img, details)
            if png_score > 0:
                details["note"] = "PNG metadata contains AI generation parameters — strong synthetic indicator."
                return AgentResult(
                    agent_name="EXIFMetadataAgent",
                    signal_name="metadata_anomaly_score",
                    score=png_score,
                    confidence=0.85,
                    details=details,
                )

        # ── Parse EXIF ─────────────────────────────────────────────────────
        parsed = _parse_exif(img)
        details["exif_fields_found"] = list(parsed.keys())

        # ── Signal A: No EXIF at all ───────────────────────────────────────
        if not parsed:
            details["exif_present"] = False
            if fmt == "PNG":
                # PNG commonly has no EXIF but no AI signatures either
                details["note"] = "No EXIF. PNG format — absence is common but not conclusive."
                return AgentResult(
                    agent_name="EXIFMetadataAgent",
                    signal_name="metadata_anomaly_score",
                    score=0.52,
                    confidence=0.30,
                    details=details,
                )
            else:
                details["note"] = "No EXIF metadata in JPEG/WebP — unusual for genuine camera photos."
                return AgentResult(
                    agent_name="EXIFMetadataAgent",
                    signal_name="metadata_anomaly_score",
                    score=0.68,
                    confidence=0.55,
                    details=details,
                )

        details["exif_present"] = True

        # ── Signal B: AI software tag ──────────────────────────────────────
        for field in ("software", "make", "model", "user_comment"):
            if field not in parsed:
                continue
            val_lower = str(parsed[field]).lower()
            for kw in _AI_SOFTWARE_KEYWORDS:
                if kw in val_lower:
                    details["ai_keyword"] = f"{field}: {parsed[field]}"
                    details["note"] = f"AI software tag detected in EXIF '{field}' field."
                    return AgentResult(
                        agent_name="EXIFMetadataAgent",
                        signal_name="metadata_anomaly_score",
                        score=0.93,
                        confidence=0.88,
                        details=details,
                    )

        # ── Signal C: Real camera fingerprint ─────────────────────────────
        has_camera_make = "make" in parsed
        has_camera_model = "model" in parsed
        has_exposure = "exposure_time" in parsed
        has_fnumber = "f_number" in parsed
        has_iso = "iso_speed" in parsed
        has_focal = "focal_length" in parsed
        has_datetime = "datetime" in parsed or "datetime_original" in parsed
        has_gps = "gps_info" in parsed
        has_maker_note = "maker_note" in parsed   # Very strong indicator — camera-specific
        has_file_source = "file_source" in parsed

        if has_camera_make:
            make_lower = str(parsed["make"]).lower()
            is_known_brand = any(m in make_lower for m in _REAL_CAMERA_MAKES)
            details["camera_make"] = str(parsed["make"])
            details["is_known_camera_brand"] = is_known_brand
        if has_camera_model:
            details["camera_model"] = str(parsed["model"])

        # Physical exposure settings: impossible to fabricate accurately at scale
        physical_count = sum([has_exposure, has_fnumber, has_iso, has_focal])
        completeness_items = [
            has_camera_make,
            has_camera_model,
            has_exposure,
            has_fnumber,
            has_iso,
            has_focal,
            has_datetime,
            has_maker_note,
        ]
        completeness = sum(completeness_items) / len(completeness_items)
        details["exif_completeness_score"] = round(completeness, 3)
        details["physical_settings_count"] = physical_count

        # ── Signal D: DateTime sanity ──────────────────────────────────────
        dt_str = str(parsed.get("datetime_original", parsed.get("datetime", "")))
        if dt_str and (dt_str.startswith("0000") or dt_str.startswith("1970-01-01")):
            details["datetime_anomaly"] = f"Suspicious timestamp: {dt_str}"
            completeness -= 0.15

        # ── Signal E: file_source check ────────────────────────────────────
        if has_file_source:
            # Value 3 = Digital Still Camera; other values unusual
            fs = parsed.get("file_source")
            if fs == 3 or str(fs) == "3":
                details["file_source"] = "Digital Still Camera (authentic indicator)"
                completeness = min(1.0, completeness + 0.10)

        # ── Combine ────────────────────────────────────────────────────────
        # More complete, camera-consistent EXIF → lower fake score
        fake_score = float(np.clip(1.0 - completeness, 0.0, 1.0))

        # Confidence scales with completeness extremes (very complete or very empty)
        confidence = float(np.clip(abs(completeness - 0.5) * 1.6, 0.25, 0.70))

        if completeness >= 0.75:
            details["note"] = (
                "Rich EXIF with camera make/model and physical exposure settings — "
                "consistent with genuine camera capture."
            )
        elif completeness >= 0.40:
            details["note"] = (
                "Partial EXIF present. Some camera indicators found but metadata is incomplete — "
                "could indicate post-processing or partial stripping."
            )
        else:
            details["note"] = (
                "Minimal EXIF. Camera make/model and physical settings absent — "
                "consistent with AI-generated or heavily processed media."
            )

        return AgentResult(
            agent_name="EXIFMetadataAgent",
            signal_name="metadata_anomaly_score",
            score=fake_score,
            confidence=confidence,
            details=details,
        )

    except Exception as exc:
        logger.error("EXIFMetadataAgent error: %s", exc, exc_info=True)
        return AgentResult(
            agent_name="EXIFMetadataAgent",
            signal_name="metadata_anomaly_score",
            score=0.5,
            confidence=0.1,
            details={"error": str(exc)},
        )
