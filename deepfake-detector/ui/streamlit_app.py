"""
Obscuro Deepimage — Streamlit frontend.

Provides a drag-and-drop interface for image/video upload and displays:
  - Verdict badge and probability meter
  - Per-agent signal breakdown with confidence indicators
  - Fusion weights (including quantum-inspired calibration note)
  - Full structured forensic rationale
  - System stats and accuracy tracker
  - Model card and technical report tabs
"""
import os
import json
import time
from pathlib import Path
from typing import Optional
import streamlit as st
import httpx

# ─── Config ───────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Obscuro Deepimage",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; }
  .subtitle   { font-size: 1.05rem; color: #6B7280; margin-bottom: 1.5rem; }
  .verdict-fake  { background: #FEF2F2; border: 2px solid #EF4444;
                   border-radius: 12px; padding: 1.2rem; text-align: center; }
  .verdict-real  { background: #F0FDF4; border: 2px solid #22C55E;
                   border-radius: 12px; padding: 1.2rem; text-align: center; }
  .verdict-uncertain { background: #FFFBEB; border: 2px solid #F59E0B;
                        border-radius: 12px; padding: 1.2rem; text-align: center; }
  .verdict-label { font-size: 2rem; font-weight: 800; }
  .agent-card { background: #F9FAFB; border: 1px solid #E5E7EB;
                border-radius: 10px; padding: 0.9rem; margin: 0.5rem 0; }
  .skipped    { opacity: 0.5; font-style: italic; }
  .disclaimer { background: #FFF7ED; border-left: 4px solid #F59E0B;
                padding: 0.8rem 1rem; font-size: 0.88rem; border-radius: 4px; }
  .quantum-badge { background: #EDE9FE; border: 1px solid #8B5CF6;
                   border-radius: 6px; padding: 0.25rem 0.6rem;
                   font-size: 0.78rem; color: #6D28D9; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ─────────────────────────────────────────────────────────

def call_analyse(file_bytes: bytes, filename: str) -> Optional[dict]:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{API_URL}/api/analyse",
                files={"file": (filename, file_bytes, "application/octet-stream")},
            )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:400]}")
            return None
    except httpx.ConnectError:
        st.error(
            "⚠️ Cannot connect to the detection backend. "
            "Make sure the API server is running (port 8000)."
        )
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def get_stats() -> Optional[dict]:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_URL}/api/stats")
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def render_verdict(result: dict):
    verdict = result.get("verdict", "UNCERTAIN")
    prob = result.get("deepfake_probability", 0.5)
    conf = result.get("confidence_in_verdict", 0.0)
    media_type = result.get("media_type", "image")

    # ── Verdict badge ────────────────────────────────────────────────────────
    css_class = {
        "LIKELY FAKE": "verdict-fake",
        "LIKELY REAL": "verdict-real",
        "UNCERTAIN":   "verdict-uncertain",
    }.get(verdict, "verdict-uncertain")

    emoji = {"LIKELY FAKE": "🚨", "LIKELY REAL": "✅", "UNCERTAIN": "⚠️"}.get(verdict, "❓")
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="verdict-label">{emoji} {verdict}</div>'
        f'<div style="font-size:1.1rem; margin-top:0.3rem;">'
        f'Deepfake probability: <strong>{prob:.1%}</strong> | '
        f'Verdict confidence: <strong>{conf:.0%}</strong> | '
        f'Media: {media_type}'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    proc_time = result.get("processing_time_s", 0)
    st.caption(f"⏱ Analysis completed in {proc_time:.2f}s")

    # ── Warnings ─────────────────────────────────────────────────────────────
    for w in result.get("warnings", []):
        st.warning(w)

    st.divider()

    # ── Agent breakdown ───────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📊 Agent Signal Breakdown")
        fusion_weights = result.get("fusion_weights", {})
        for ar in result.get("agent_results", []):
            name = ar.get("agent_name", "Agent")
            score = ar.get("score", 0.5)
            conf_a = ar.get("confidence", 0.0)
            ran = ar.get("ran", True)
            signal = ar.get("signal_name", "")
            weight = fusion_weights.get(signal, 0.0)

            if not ran:
                st.markdown(
                    f'<div class="agent-card skipped">⏭ <strong>{name}</strong>'
                    f' — {ar.get("skipped_reason", "Skipped")}</div>',
                    unsafe_allow_html=True,
                )
                continue

            color = "#EF4444" if score > 0.55 else ("#22C55E" if score < 0.45 else "#F59E0B")
            direction = "🔴 FAKE signal" if score > 0.55 else ("🟢 REAL signal" if score < 0.45 else "🟡 Neutral")

            with st.expander(f"{name}  |  score: {score:.2f}  |  weight: {weight:.0%}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Score", f"{score:.3f}", delta=f"{direction}")
                c2.metric("Agent Confidence", f"{conf_a:.0%}")
                c3.metric("Fusion Weight", f"{weight:.0%}")

                st.progress(score, text=f"Fake signal strength: {score:.1%}")

                details = ar.get("details", {})
                if details:
                    with st.container():
                        for k, v in details.items():
                            if k not in ("note", "caveat", "error"):
                                st.text(f"  {k}: {v}")
                        if "note" in details:
                            st.info(f"ℹ️ {details['note']}")
                        if "caveat" in details:
                            st.warning(f"⚠️ {details['caveat']}")
                        if "error" in details:
                            st.error(f"Error: {details['error']}")

    with col2:
        st.subheader("⚖️ Fusion Weights")
        st.markdown(
            '<span class="quantum-badge">🔬 Experimental: quantum-inspired calibration (PennyLane, classical simulation)</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Weights adjusted by a variational quantum circuit simulated on CPU. "
            "No quantum hardware advantage is claimed — this is a research exploration only."
        )
        st.write("")
        for signal, w in fusion_weights.items():
            readable = signal.replace("_score", "").replace("_", " ").title()
            st.progress(float(w), text=f"{readable}: {w:.1%}")

        st.write("")
        st.subheader("📝 Probability Meter")
        st.progress(float(prob))
        st.caption(
            f"{'High confidence FAKE' if prob > 0.75 else 'Leaning FAKE' if prob > 0.55 else 'Leaning REAL' if prob < 0.45 else 'Low confidence FAKE' if prob < 0.25 else 'Inconclusive'}"
        )

    st.divider()

    # ── Rationale ─────────────────────────────────────────────────────────────
    st.subheader("🔎 Forensic Rationale")
    st.markdown(result.get("rationale", "No rationale available."))

    st.divider()

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "<strong>Important:</strong> Deepfake detection is probabilistic. "
        "This tool is intended for forensic research purposes and is <em>not</em> "
        "legally admissible evidence. False positives and false negatives occur. "
        "Do not rely on this output alone for consequential decisions."
        "</div>",
        unsafe_allow_html=True,
    )


def render_model_card():
    st.markdown("""
## 📄 Model Card — Obscuro Deepimage v1.0

### Intended Use
Forensic research and media authenticity assessment. Designed to assist analysts
in identifying potentially synthetic or manipulated imagery and video.
**Not intended for** legally binding determinations, surveillance, or use against
individuals without their knowledge.

### Training Data & Model Origins
| Agent | Model | Training Data |
|---|---|---|
| Spatial Forensics | ViT (dima806/deepfake_vs_real_image_detection) | Mixed real/fake dataset (faces) |
| Spatial Fallback | EfficientNet-B4 (timm) | ImageNet-1K (not fine-tuned on deepfakes) |
| Frequency Analysis | Signal-processing heuristics | No training required |
| Temporal Analysis | Optical flow + Haar cascades (OpenCV) | No training required |
| Biological Signal | CHROM rPPG algorithm | No training required |
| Audio-Visual Sync | Cross-correlation heuristic | No training required |
| Fusion Agent | Confidence-weighted ensemble + PennyLane circuit | Adaptive (logs) |

### Known Limitations & Generalisation Gaps
- **Generation method overfitting**: Detectors trained on GAN outputs may not generalise to diffusion-model fakes
  (Stable Diffusion, DALL-E 3, Midjourney). The frequency analysis module attempts to address this
  with diffusion-specific channel correlation checks, but performance on newest generators is unvalidated.
- **Compression robustness**: Heavy JPEG/H.264 compression degrades frequency and spatial signals.
  Performance drops significantly at JPEG quality < 50.
- **Resolution dependence**: Biological signal (rPPG) and temporal analysis require ≥480p video at ≥15fps.
  Low-resolution inputs receive lower-confidence scores.
- **Face detection failure**: No face detected → full-image analysis, which is less accurate.
- **Calibration**: Without fine-tuning on labelled deepfake datasets, the timm EfficientNet fallback
  outputs uncalibrated probabilities. The system explicitly labels this in verdicts.
- **Adversarial examples**: A sophisticated adversary who knows this system's detection signals could
  craft inputs that evade detection. This tool should not be the sole line of defence.

### Misuse Risks
- **False accusations**: Do not use this tool to publicly accuse individuals of creating deepfakes
  based solely on this output.
- **Chilling effect**: Probabilistic false positives on authentic media could suppress legitimate content.
- **Arms race dynamics**: Detection signal descriptions are intentionally kept at a level that does not
  constitute a "how to evade detection" checklist.

### Evaluation Status
Without access to labelled benchmark data (FaceForensics++, DFDC require registration),
production accuracy metrics are not available for this deployment. The Self-Improvement Agent
will compute accuracy, precision, recall, F1, and AUC once labelled feedback is submitted.

### Version & Date
v1.0.0 — July 2026
""")


def render_technical_report():
    st.markdown("""
## 📚 Technical Research Summary — Deepfake Detection (2024-2026 SOTA)

### Executive Summary
Deepfake detection is an active arms race. As of 2025-2026, the threat landscape has shifted
significantly from GAN-based face swaps to diffusion-model-generated imagery, requiring updated
detector strategies. Below is a synthesis of current approaches and their trade-offs.

---

### 1. Spatial / Pixel-Level Detection
**Current SOTA**: Vision Transformer (ViT) and hybrid CNN-ViT architectures outperform
pure CNNs on cross-dataset generalisation.

- **Key models**: CLIP-based detectors (UniFD, 2023), LSDA (Large-Scale Deepfake Analysis),
  Xception-based networks remain competitive on face swap benchmarks.
- **Limitation**: Heavily overfit to training distribution. A model trained on ProGAN fakes
  achieves ~98% AUC on that set but drops to 50-65% on DALL-E 3 or Stable Diffusion outputs.

### 2. Frequency-Domain Analysis
**Current SOTA**: Spectral peak detection (Frank et al., ICML 2020) works well for GAN outputs
but is weaker against diffusion models that don't use periodic upsampling.

- **Diffusion-specific frequency signatures** (Corvi et al., ICASSP 2023): focus on mid-frequency
  noise patterns and cross-channel correlation rather than spectral peaks.
- **DCT coefficient distribution**: Real JPEG images have a characteristic Laplacian distribution
  with quantisation nulls; synthetics often have smoother or anomalous distributions.

### 3. Temporal Consistency (Video)
- **Optical flow irregularity**: Face-swap deepfakes can introduce frame-to-frame jitter at blending boundaries.
- **Blink rate analysis**: Deepfakes often exhibit zero or abnormal blink rates due to training data biases
  (Li et al., CVPR 2018 — this gap has been partially closed in newer generators).
- **Biological signal (rPPG)**: Absence of physiological heartbeat signal in face ROI.
  Sensitive but noisy in short clips. Best combined with other signals.

### 4. Audio-Visual Coherence
- **SyncNet-based detection**: Cross-modal synchronisation check between lip landmarks and speech audio.
- **Limitation**: Works well for talking-head deepfakes but not for image-only or silent-video fakes.

### 5. Benchmark Datasets (Access Requirements)
| Dataset | Generation Types Covered | Access |
|---|---|---|
| FaceForensics++ | NeuralTextures, Deepfakes, Face2Face, FaceSwap | Public (registration) |
| DFDC | Multiple generators | Kaggle competition |
| Celeb-DF v2 | Face swaps | Public |
| DeeperForensics-1.0 | Face swap + perturbations | Public (registration) |
| WildDeepfake | In-the-wild | Public |
| **Gap**: | Diffusion-model fakes (SD, DALL-E) | No large public benchmark yet |

### 6. The Generalisation Challenge
The core unsolved problem: detectors trained on generation method A overfit and fail on
method B. Proposed mitigations:
- **Foundation model features**: Using CLIP, DINO, or similar large pretrained encoders as
  feature extractors produces more transferable representations.
- **Frequency augmentation**: Training with diverse frequency perturbations improves robustness.
- **Ensemble approaches**: Multiple independent signals with uncertainty quantification
  (the architecture Obscuro Deepimage implements) provide more reliable cross-distribution estimates.

### 7. Quantum-Inspired Module
The fusion weight optimisation step uses a PennyLane variational quantum circuit as a classical
simulation. **Honest assessment**: there is no peer-reviewed evidence of quantum ML outperforming
classical methods for deepfake detection. This module is implemented as a research exploration
of variational quantum circuits for ensemble meta-learning, not a performance claim. All circuits
run on CPU simulation; no quantum hardware is used.

---

*References: Frank et al. (2020), Li et al. (2018, 2020), Rossler et al. (2019 — FF++),
Corvi et al. (2023), Ojha et al. (2023 — UniFD), Tan et al. (2024).*
""")


# ─── Main app layout ───────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Obscuro Deepimage")
        st.markdown("*Agentic Multi-Modal Deepfake Detection*")
        st.divider()
        st.markdown("**Specialist agents:**")
        st.markdown("- 🖼️ Spatial Forensics (ViT/EfficientNet)")
        st.markdown("- 📡 Frequency Analysis (FFT/DCT)")
        st.markdown("- 🎬 Temporal Consistency (video)")
        st.markdown("- 💓 Biological Signal rPPG (video)")
        st.markdown("- 🎙️ Audio-Visual Sync (video)")
        st.markdown("- ⚛️ Quantum-inspired Fusion")
        st.divider()

        # System health check
        st.markdown("**System status:**")
        try:
            with httpx.Client(timeout=3.0) as client:
                h = client.get(f"{API_URL}/api/health")
            if h.status_code == 200:
                data = h.json()
                st.success("✅ API backend: online")
                if data.get("models_loaded"):
                    st.success("✅ ML models: loaded")
                else:
                    st.warning("⚠️ ML models: loading or partial")
            else:
                st.error("❌ API backend: error")
        except Exception:
            st.error("❌ API backend: offline")

        st.divider()
        st.caption(
            "⚠️ This tool is for forensic research only. "
            "Output is probabilistic and not legally admissible."
        )

    # Main area tabs
    tab_detect, tab_stats, tab_card, tab_report = st.tabs([
        "🔍 Detect", "📈 Stats & Calibration", "📄 Model Card", "📚 Technical Report"
    ])

    with tab_detect:
        st.markdown('<div class="main-title">🔍 Obscuro Deepimage</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Agentic Multi-Modal Deepfake Detection System — '
            'upload an image or video to receive a multi-signal forensic verdict.</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload image or video",
            type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv"],
            help="Images: JPG, PNG, WebP, BMP. Video: MP4, AVI, MOV (max 100MB).",
        )

        if uploaded is not None:
            file_bytes = uploaded.read()
            filename = uploaded.name
            size_mb = len(file_bytes) / (1024 * 1024)

            col_prev, col_meta = st.columns([1, 2])
            with col_prev:
                if any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
                    st.image(file_bytes, caption=f"{filename} ({size_mb:.1f}MB)", use_container_width=True)
                else:
                    st.video(file_bytes)
                    st.caption(f"{filename} ({size_mb:.1f}MB)")

            with col_meta:
                st.markdown(f"**Filename:** `{filename}`")
                st.markdown(f"**Size:** {size_mb:.2f} MB")
                st.markdown(f"**Type:** {'Video' if any(filename.lower().endswith(e) for e in ['.mp4', '.avi', '.mov', '.mkv']) else 'Image'}")

            if st.button("🔬 Run Forensic Analysis", type="primary", use_container_width=True):
                with st.spinner("Running multi-agent forensic analysis..."):
                    result = call_analyse(file_bytes, filename)

                if result:
                    st.success("Analysis complete.")
                    render_verdict(result)

                    # Download report
                    report_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download Forensic Report (JSON)",
                        data=report_json,
                        file_name=f"obscuro_report_{filename}.json",
                        mime="application/json",
                    )

                    # Feedback form
                    with st.expander("📬 Submit Ground Truth (helps self-improvement agent)"):
                        st.markdown(
                            "Know the true label for this sample? Submitting it helps the "
                            "system recalibrate its fusion weights over time."
                        )
                        input_hash = result.get("_hash", "")
                        # Compute hash client-side
                        import hashlib
                        client_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
                        gt = st.radio("Ground truth:", ["fake", "real"], horizontal=True)
                        if st.button("Submit feedback"):
                            try:
                                with httpx.Client(timeout=5.0) as client:
                                    fb_resp = client.post(
                                        f"{API_URL}/api/feedback",
                                        data={"input_hash": client_hash, "ground_truth": gt},
                                    )
                                if fb_resp.status_code == 200:
                                    st.success("Feedback recorded. Thank you.")
                                else:
                                    st.error(f"Feedback error: {fb_resp.text[:200]}")
                            except Exception as exc:
                                st.error(f"Could not submit feedback: {exc}")

    with tab_stats:
        st.markdown("## 📈 Accuracy & Calibration Dashboard")
        st.markdown(
            "This tab shows the Self-Improvement Agent's accuracy metrics, computed from "
            "labelled predictions submitted via the feedback form."
        )
        if st.button("🔄 Refresh Stats"):
            pass  # Re-runs the tab
        stats = get_stats()
        if stats:
            report = stats.get("accuracy_report", {})
            if "accuracy" in report:
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{report['accuracy']:.1%}")
                cols[1].metric("Precision", f"{report['precision']:.1%}")
                cols[2].metric("Recall", f"{report['recall']:.1%}")
                cols[3].metric("F1", f"{report['f1']:.1%}")
                if report.get("degradation_flag"):
                    st.error(
                        "⚠️ DEGRADATION FLAG: Accuracy below threshold. "
                        "Recalibration recommended."
                    )
            else:
                st.info(report.get("message", "No stats available yet."))
                st.markdown(
                    f"**Total predictions logged:** {report.get('total_predictions', 0)}  \n"
                    f"**Labelled predictions:** {report.get('labelled_predictions', 0)}"
                )

            recal = stats.get("weight_recalibration")
            if recal:
                st.markdown("### Suggested Weight Recalibration")
                st.json(recal)
        else:
            st.warning("Could not retrieve stats. Is the API backend running?")

    with tab_card:
        render_model_card()

    with tab_report:
        render_technical_report()


if __name__ == "__main__":
    main()
