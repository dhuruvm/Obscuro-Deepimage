# Obscuro Deepimage

An agentic, multi-signal deepfake detection system. Upload an image and the system runs multiple specialist detection agents in parallel, fuses the signals, and returns a forensic verdict with confidence score and plain-language explanation.

## Architecture

- **Backend:** FastAPI (`deepfake-detector/app/main.py`) — serves REST API + built-in HTML UI
- **Agents:** Spatial Forensics, Frequency Analysis, Temporal Consistency, Biological Signal, Audio-Visual Sync, Fusion (quantum-inspired ensemble)
- **Self-Improvement:** Logs predictions, recalibrates fusion weights, tracks accuracy drift
- **Optional Streamlit UI:** `deepfake-detector/ui/streamlit_app.py`

## How to run

The primary workflow is **artifacts/api-server: Deepimage**:
```
cd /home/runner/workspace/deepfake-detector && uv run python start_api.py
```
This installs dependencies automatically via `uv` and starts the FastAPI server on `$PORT`.

The Streamlit UI can be started with the **Deepimage UI** workflow:
```
cd deepfake-detector && python start_ui.py
```

## Key files

| Path | Purpose |
|------|---------|
| `deepfake-detector/app/main.py` | FastAPI app + all API endpoints |
| `deepfake-detector/app/agents/` | Specialist detection agents |
| `deepfake-detector/app/models/detector.py` | Model loading (ViT/EfficientNet) |
| `deepfake-detector/app/static/index.html` | Drag-and-drop frontend UI |
| `deepfake-detector/ui/streamlit_app.py` | Alternative Streamlit frontend |
| `deepfake-detector/model_card.md` | Model details, limitations, ethics |
| `deepfake-detector/requirements.txt` | Python dependencies |

## Notes

- The HuggingFace ViT model (`dima806/deepfake_vs_real_image_detection`) downloads on first run; if unavailable it falls back to a minimal CNN.
- The quantum-inspired fusion module uses PennyLane on classical CPU — no quantum hardware, labeled experimental.
- No external secrets required to run.

## User preferences
