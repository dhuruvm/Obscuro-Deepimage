# Obscuro Deepimage

Agentic multi-modal deepfake detection system. Upload an image or video and receive a multi-signal forensic verdict — not a black-box label — with per-agent score breakdowns, fusion weights, and structured prose rationale.

## Run & Operate

- **UI (Streamlit):** `cd deepfake-detector && python start_ui.py` — port 8080 (webview)
- **API (FastAPI):** `cd deepfake-detector && python start_api.py` — port 8000 (console)
- Workflows are pre-configured as "Deepimage UI" and "Deepimage API"

## Stack

- **Language:** Python 3.11
- **ML:** PyTorch 2.x + timm (EfficientNet-B4) + HuggingFace Transformers (ViT deepfake detector)
- **CV:** OpenCV, MediaPipe, scipy/numpy
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Quantum module:** PennyLane (classical simulation, clearly labelled experimental)

## Where Things Live

```
deepfake-detector/
├── app/
│   ├── main.py                    # FastAPI routes (/api/analyse, /api/health, /api/stats, /api/feedback)
│   ├── schemas.py                 # Pydantic: AgentResult, ForensicVerdict
│   ├── agents/
│   │   ├── coordinator.py         # Top-level orchestrator — routes image vs video
│   │   ├── spatial.py             # ViT / EfficientNet pixel-level artifacts
│   │   ├── frequency.py           # FFT + DCT spectral fingerprints
│   │   ├── temporal.py            # Optical flow, blink rate (video)
│   │   ├── biological.py          # CHROM rPPG heartbeat signal (video)
│   │   ├── audio_visual.py        # Lip-sync mismatch (video + audio)
│   │   ├── fusion.py              # Weighted ensemble + quantum-inspired calibration
│   │   └── self_improvement.py    # Prediction logging, accuracy tracking, recalibration
│   ├── models/detector.py         # Model loading (HF pipeline + timm fallback)
│   └── preprocessing/pipeline.py  # Face detection, frame extraction, normalisation
├── ui/streamlit_app.py            # Full Streamlit frontend
├── logs/predictions.jsonl         # Self-improvement prediction log
├── model_card.md                  # Model card (intended use, limitations, risks)
└── requirements.txt
```

## Architecture Decisions

- **Multi-agent, not monolithic:** Each forensic signal is an independent agent so failures are isolated and weights are interpretable. The Coordinator Agent routes to applicable agents per media type.
- **HuggingFace primary + timm fallback:** `dima806/deepfake_vs_real_image_detection` (ViT, 325MB) is the primary spatial classifier; EfficientNet-B4 (timm) is a secondary texture-feature signal. Both load at startup.
- **Quantum-inspired fusion:** PennyLane variational circuit adjusts ensemble weights — runs entirely on CPU simulation. Clearly labelled experimental; no quantum hardware or speedup claimed.
- **Self-improvement is bounded:** The Self-Improvement Agent logs predictions and recalibrates fusion weights when labelled ground truth is submitted via `/api/feedback`. No unbounded recursive modification.
- **Frequency analysis covers both GAN and diffusion:** FFT spectral peaks target GAN fingerprints; channel cross-correlation targets diffusion-model artifacts (Corvi et al. 2023).

## Product

A forensic tool that accepts an image or short video and returns:
- **Verdict**: LIKELY FAKE / LIKELY REAL / UNCERTAIN
- **Deepfake probability** (0–1)
- **Per-agent signal breakdown** with individual scores, confidence, and details
- **Fusion weights** (quantum-inspired calibration noted)
- **Structured prose rationale** readable by non-technical users
- **Downloadable JSON forensic report**
- **Accuracy dashboard** (once ground-truth feedback is submitted)
- **Model card** and **technical research report** tabs

## User Preferences

- Build real working systems, not demos — all signals use actual algorithms or pretrained models
- Honest framing: limitations, generalisation gaps, and the "quantum" module are explicitly labelled
- No mocked outputs

## Gotchas

- Models download from HuggingFace on first startup (~400MB total). Subsequent starts use the cache.
- The timm EfficientNet-B4 fallback is NOT fine-tuned on deepfakes — verdicts from it alone are uncalibrated. The UI and rationale warn about this.
- rPPG and temporal agents require video ≥15 frames; they skip gracefully with a `skipped_reason` if not enough frames.
- Audio-visual sync requires ffmpeg (installed) and a video with an audio track.
- Streamlit runs on port 8080; FastAPI on port 8000. The UI calls the API via `http://localhost:8000`.
