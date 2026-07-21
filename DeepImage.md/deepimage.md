# Project Obscuro DeepImage: Agentic Deepfake Image Detection System — Master Prompt

## ROLE
You are a senior AI research architect specializing in two combined disciplines:
1. Computer vision forensics and deepfake/synthetic-image detection
2. Agentic AI system design (multi-tool orchestration, reasoning loops, evaluation)

## OBJECTIVE
Design a single **agentic AI system** — codename **Project Obscuro DeepImage** — whose sole job is to
determine whether a given **image** is authentic or AI-generated/manipulated (deepfake), and to
do so with high accuracy, generalization to unseen generation methods, and an explainable verdict.

This is not a request for a single classifier model. It is a request for an **agent**: a system
that can reason, call multiple specialized detection tools, weigh conflicting evidence, and
justify its conclusion — the way a human forensic analyst would.

## SCOPE (explicit boundaries)
- Input: a single static image (no video, no audio).
- Output: a verdict (Authentic / Likely Fake / Fake), a confidence score, and a plain-language
  explanation of the evidence.
- Out of scope: video, audio, real-time streaming detection, and legal/forensic certification.

## BACKGROUND CONTEXT (ground the research in this)
Deepfake images today are produced mainly via GANs and diffusion models. Detection research as
of 2026 shows:
- No single-signal detector generalizes well to unseen generators — multimodal/multi-signal
  fusion consistently outperforms single-method detectors.
- The most reliable signals are combinations of: frequency-domain artifacts, GAN/diffusion
  "fingerprints," inconsistencies in lighting/reflection/shadow physics, facial landmark and
  anatomical irregularities, compression-artifact mismatches, and metadata/EXIF inconsistencies.
- Lightweight, targeted forensic cues (rather than ever-larger black-box models) are showing
  better robustness and cross-generator generalization in recent 2026 research.
- Cross-domain/ensemble fusion architectures are the current state of the art for accuracy.

Use this as a starting baseline — but verify and expand it with your own research rather than
treating it as final.

## RESEARCH TASK (do this before designing anything)
1. Identify and compare the top-performing current (2025–2026) deepfake **image** detection
   methods, across: CNN-based classifiers, frequency/spectral analysis, diffusion-artifact
   detectors, ensemble/ fusion approaches, and foundation-model-based detectors.
   For each, note: accuracy, generalization to unseen generators, computational cost, and
   known weaknesses.
2. Identify the best current **agentic architecture patterns** for orchestrating multiple
   detection tools (e.g. planner-executor, tool-calling loop with a verifier/critic step,
   ensemble-voting agents, hierarchical agents). Compare tradeoffs.
3. Identify how leading real-world systems (e.g. Reality Defender, Hive, Intel FakeCatcher,
   Microsoft's tools, open-source projects) structure their pipelines, and what can be learned
   from their design choices.

## DESIGN TASK (after research, produce this)
Propose a full architecture for Project Obscuro DeepImage, including:
- **Agent loop**: how the agent receives an image, decides which detectors to run, interprets
  results, and resolves disagreement between tools.
- **Detection toolset**: the specific set of detection methods/models the agent should have
  access to as callable tools, and why each was chosen.
- **Fusion/reasoning layer**: how the agent combines multiple signals into one verdict
  (e.g. weighted scoring, learned meta-classifier, LLM-based reasoning over evidence).
- **Confidence & explainability**: how the agent produces a human-readable justification and
  a calibrated confidence score, not just a label.
- **Failure modes & limitations**: known edge cases (compression, low resolution, cropped
  faces, non-face deepfakes, adversarial evasion) and how the design should handle or flag them.
- **Evaluation plan**: which benchmarks/datasets to validate against (e.g. FaceForensics++,
  DFDC, Celeb-DF) and what generalization tests should be run before trusting the system.

## OUTPUT FORMAT
Respond in this structure:
1. Research Summary (methods + architectures, with sources)
2. Recommended Architecture for Project Obscuro DeepImage (diagram-in-words + component breakdown)
3. Tool/Model Shortlist (what to actually integrate, and why)
4. Limitations & Risks
5. Next Steps / Build Plan

## SUCCESS CRITERIA
The final design should be something a small team could realistically start building — not a
theoretical overview. Prioritize methods that are proven, not just novel; flag anything still
experimental as such.
