"""
Preprocessing pipeline for Obscuro Deepimage.
Handles: face detection & crop, frame extraction from video, resize & normalize.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Target size for spatial detector
FACE_CROP_SIZE = 224
# Max frames to sample from a video
MAX_FRAMES = 32
FRAME_SAMPLE_RATE = 1  # every N seconds


def detect_and_crop_face(
    image: np.ndarray,
    padding: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect largest face in image using OpenCV DNN + fall back to Haar.
    Returns (cropped_face_bgr, (x, y, w, h)) or (None, None) if no face found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try OpenCV Haar cascade (always available)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )

    if len(faces) == 0:
        # No face found — return the full image resized
        resized = cv2.resize(image, (FACE_CROP_SIZE, FACE_CROP_SIZE))
        return resized, None

    # Pick largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    face_crop = image[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
    return face_crop, (x, y, w, h)


def preprocess_image_for_model(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Return float32 CHW tensor in [0,1] range, normalized."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (FACE_CROP_SIZE, FACE_CROP_SIZE))
    arr = rgb.astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)  # HWC → CHW


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into BGR ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img


def extract_video_frames(
    video_path: str,
    max_frames: int = MAX_FRAMES,
) -> List[np.ndarray]:
    """
    Extract uniformly sampled frames from a video file.
    Returns list of BGR ndarrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        total_frames = max_frames

    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    logger.info("Extracted %d frames from %s (total=%d, fps=%.1f)",
                len(frames), video_path, total_frames, fps)
    return frames


def frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
    """Convert BGR ndarray frames to PIL RGB images."""
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def get_face_crops(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """Extract face crops from a list of frames. Returns (crops, faces_found)."""
    crops = []
    found = 0
    for frame in frames:
        crop, bbox = detect_and_crop_face(frame)
        if crop is not None:
            crops.append(crop)
            if bbox is not None:
                found += 1
    return crops, found
