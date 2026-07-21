# Model Card — Obscuro Deepimage v1.0

## Model Details

**System name:** Obscuro Deepimage  
**Version:** 1.0.0  
**Date:** July 2026  
**Task:** Multi-modal deepfake image and video detection  
**Architecture:** Multi-agent agentic pipeline with confidence-weighted ensemble fusion

## Intended Use

### Primary Use Cases
- Forensic research and media authenticity assessment
- Journalistic verification of suspected manipulated media
- Academic research into deepfake detection methods

### Out-of-Scope Uses
- Legally binding determinations of media authenticity
- Surveillance or monitoring without subject consent
- Automated mass moderation without human review
- Any use that relies on this system as sole evidence for consequential decisions

## Agents and Models

| Agent | Method | Backbone | Training Status |
|---|---|---|---|
| Spatial Forensics | Image classification | ViT (HuggingFace: dima806/deepfake_vs_real_image_detection) | Pre-trained on deepfake dataset |
| Spatial Fallback | Feature extraction | EfficientNet-B4 (timm, ImageNet) | **NOT fine-tuned on deepfakes** |
| Frequency Analysis | FFT + DCT heuristics | Signal processing (numpy/scipy) | No training |
| Temporal Consistency | Optical flow + blink detection | OpenCV | No training |
| Biological Signal | CHROM rPPG | Signal processing | No training |
| Audio-Visual Sync | Cross-correlation | Signal processing | No training |
| Fusion Agent | Confidence-weighted ensemble | PennyLane variational circuit (experimental) | Adaptive |

## Performance

Without access to labelled benchmark data (FaceForensics++, DFDC require registration),
pre-evaluated accuracy metrics are not available. Performance characteristics:

- **Expected AUC on FaceForensics++ (c23):** ~0.85–0.92 (for ViT-based spatial agent)
- **Expected AUC on DFDC:** ~0.70–0.78 (lower due to diverse generation methods)
- **Diffusion model fakes:** Validation pending; frequency and spatial agents modified to handle
  diffusion artifacts, but comprehensive evaluation not available

## Known Limitations

1. **Generalisation gap:** Performance drops on generation methods not in training data
2. **Compression robustness:** Heavy JPEG/H.264 compression degrades frequency signals
3. **Resolution requirement:** rPPG and temporal agents need ≥480p at ≥15fps
4. **Face detection dependency:** Accuracy lower when no face is detected
5. **Short video clips:** Biological and temporal agents less reliable on clips < 2 seconds
6. **Adversarial robustness:** Not tested against adversarial attacks targeting detection signals

## Ethical Considerations

- **False positives** can harm individuals by falsely labelling real content as fake
- **False negatives** can allow manipulated content to pass undetected
- Detection signals are described at a level that does not serve as a "guide to evade detection"
- No real person's likeness is bundled with this system; only public research models are used
- Output must be interpreted by a human expert before any consequential action

## Quantum-Inspired Module Disclosure

The fusion weight optimisation step uses a PennyLane variational quantum circuit simulated
entirely on classical CPU hardware. **There is no quantum hardware advantage, no quantum
speedup, and no peer-reviewed evidence that quantum ML outperforms classical methods for
this task.** This module is included as a research exploration only and is labelled accordingly
in the UI. Do not interpret "quantum-inspired" as a performance claim.

## Contact & Reporting

For issues, false positives, or misuse concerns, consult the system operator.
