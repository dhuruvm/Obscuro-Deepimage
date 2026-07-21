"""
Fusion & Explanation Agent — Obscuro Deepimage.

Combines signals from all specialist agents into a single deepfake probability
using a confidence-weighted ensemble, with optional quantum-inspired
weight optimisation via PennyLane (classical simulation).

Outputs:
  - deepfake_probability (float, 0-1)
  - confidence_in_verdict (float, 0-1)
  - verdict string (LIKELY FAKE | UNCERTAIN | LIKELY REAL)
  - structured prose rationale
  - per-agent weight breakdown
  - conflict flag when agents strongly disagree

Quantum note: The PennyLane variational circuit runs on classical CPU simulation.
No quantum hardware is used. This module is labelled as experimental research.
"""
import logging
import numpy as np
from typing import List, Dict, Any

from app.schemas import AgentResult

logger = logging.getLogger(__name__)

# ── Signal prior weights ───────────────────────────────────────────────────────
# Calibrated for image analysis (video agents receive reduced base prior)
SIGNAL_PRIORS: Dict[str, float] = {
    "pixel_artifact_score":         0.28,   # Spatial / ViT — most reliable
    "spectral_artifact_score":      0.20,   # Frequency / FFT+SRM
    "anatomical_anomaly_score":     0.18,   # Facial landmarks (MediaPipe)
    "metadata_anomaly_score":       0.14,   # EXIF / metadata
    "temporal_inconsistency_score": 0.10,   # Video temporal (skipped for images)
    "rppg_anomaly_score":           0.05,   # rPPG biological (video only)
    "lip_sync_mismatch_score":      0.05,   # Audio-visual sync (video only)
}

# Conflict threshold: agents disagree if score delta ≥ this and both conf ≥ 0.40
_CONFLICT_DELTA = 0.45


# ── Quantum-inspired weight optimisation ──────────────────────────────────────

def _quantum_inspired_weight_update(
    scores: List[float],
    confidences: List[float],
    priors: List[float],
) -> List[float]:
    """
    Variational quantum circuit (PennyLane, classical CPU simulation) that
    adjusts ensemble weights based on per-agent confidence scores.

    EXPERIMENTAL LABEL: This runs on classical hardware — no quantum speedup.
    Included as a research exploration into variational quantum circuits for
    ensemble fusion weight calibration.
    """
    try:
        import pennylane as qml

        n = len(scores)
        if n < 2:
            return priors

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circuit(params):
            for i in range(n):
                qml.RY(params[i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        # Map prior weights [0,1] → rotation angles [0, π], scaled by confidence
        params = np.array([p * np.pi for p in priors], dtype=np.float64)
        adjusted = params * np.clip(confidences, 0.1, 1.0)

        z_vals = circuit(adjusted)
        weights = np.array([(1.0 + float(z)) / 2.0 for z in z_vals])

        # Blend: 35% circuit output + 65% confidence-adjusted priors
        conf_w = np.array(priors) * np.clip(confidences, 0.1, 1.0)
        blended = 0.35 * weights + 0.65 * (conf_w / (conf_w.sum() + 1e-8))
        normalised = blended / (blended.sum() + 1e-8)
        return normalised.tolist()

    except Exception as exc:
        logger.debug("Quantum-inspired optimisation unavailable (%s) — using priors.", exc)
        return priors


# ── Conflict detection ─────────────────────────────────────────────────────────

def _find_conflicts(
    active: List[AgentResult],
    signal_map: Dict[str, tuple],
) -> List[str]:
    """Return human-readable descriptions of strong inter-agent disagreements."""
    conflicts = []
    eligible = [
        ar for ar in active
        if ar.confidence >= 0.40 and ar.signal_name in signal_map
    ]
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            if abs(a.score - b.score) >= _CONFLICT_DELTA:
                dir_a = "FAKE" if a.score > 0.5 else "REAL"
                dir_b = "FAKE" if b.score > 0.5 else "REAL"
                conflicts.append(
                    f"{a.agent_name} ({dir_a} {a.score:.2f}) ↔ "
                    f"{b.agent_name} ({dir_b} {b.score:.2f})"
                )
    return conflicts


# ── Verdict generation ─────────────────────────────────────────────────────────

def _verdict_label(prob: float, confidence: float, has_conflict: bool) -> str:
    """
    Produce a three-tier verdict aligned with the spec:
      LIKELY FAKE | UNCERTAIN | LIKELY REAL
    Conflicts lower the confidence threshold, pushing toward UNCERTAIN.
    """
    if confidence < 0.28 or has_conflict:
        return "UNCERTAIN"
    if prob >= 0.60:
        return "LIKELY FAKE"
    if prob <= 0.40:
        return "LIKELY REAL"
    return "UNCERTAIN"


# ── Rationale builder ─────────────────────────────────────────────────────────

def _build_rationale(
    verdict: str,
    prob: float,
    agent_results: List[AgentResult],
    weights: Dict[str, float],
    conflicts: List[str],
) -> str:
    lines = []
    lines.append(f"**Verdict: {verdict}** — deepfake probability: {prob:.1%}\n")

    lines.append("**Forensic signal summary:**\n")
    for ar in agent_results:
        if not ar.ran:
            lines.append(
                f"- *{ar.agent_name}*: skipped — {ar.skipped_reason or 'not applicable'}"
            )
            continue
        w = weights.get(ar.signal_name, 0.0)
        direction = "FAKE" if ar.score > 0.55 else ("REAL" if ar.score < 0.45 else "neutral")
        lines.append(
            f"- *{ar.agent_name}* (weight {w:.0%}): score {ar.score:.2f} → "
            f"**{direction}** (agent confidence: {ar.confidence:.0%})"
        )
        if "note" in ar.details:
            lines.append(f"  ↳ {ar.details['note']}")
        if "caveat" in ar.details:
            lines.append(f"  ⚠️ {ar.details['caveat']}")

    if conflicts:
        lines.append("\n**⚠️ Agent conflicts detected:**")
        for c in conflicts:
            lines.append(f"  - {c}")
        lines.append(
            "  Conflicting signals mean multiple independent detectors disagree. "
            "The verdict should be treated with caution and reviewed by a human expert."
        )

    lines.append("\n**Interpretation:**")
    if verdict == "LIKELY FAKE":
        lines.append(
            "Multiple independent forensic signals converge on synthetic origin. "
            "Pixel-level, spectral, anatomical, and/or metadata evidence point toward "
            "AI generation or manipulation. "
        )
    elif verdict == "LIKELY REAL":
        lines.append(
            "No strong synthetic indicators were found. Forensic signals are broadly "
            "consistent with authentic, unmanipulated media. This is not a certificate "
            "of authenticity — sophisticated adversarial content may evade detection."
        )
    else:
        lines.append(
            "Evidence is inconclusive. Signals disagree or overall confidence is too "
            "low for a reliable determination. Manual expert review is recommended."
        )

    lines.append(
        "\n**⚠️ Caveat:** Deepfake detection is probabilistic. No detector generalises "
        "perfectly to all generation methods. This analysis is forensic evidence, "
        "not legal proof. False positives and false negatives occur."
    )

    return "\n".join(lines)


# ── Main fusion function ───────────────────────────────────────────────────────

def fuse(
    agent_results: List[AgentResult],
    use_quantum: bool = True,
) -> Dict[str, Any]:
    """
    Fuse agent results into a single verdict.
    Returns dict matching ForensicVerdict fields (minus agent_results).
    """
    active = [ar for ar in agent_results if ar.ran]

    if not active:
        return dict(
            verdict="UNCERTAIN",
            deepfake_probability=0.5,
            confidence_in_verdict=0.0,
            fusion_weights={},
            rationale="No agents produced usable results.",
        )

    signal_map = {ar.signal_name: (ar.score, ar.confidence) for ar in active}
    available_signals = [ar.signal_name for ar in active]

    # Normalised priors for available signals
    raw_priors = [SIGNAL_PRIORS.get(s, 0.08) for s in available_signals]
    total = sum(raw_priors) or 1.0
    prior_weights = [w / total for w in raw_priors]

    scores      = [signal_map[s][0] for s in available_signals]
    confidences = [signal_map[s][1] for s in available_signals]

    # Quantum-inspired calibration
    if use_quantum and len(scores) >= 2:
        weights = _quantum_inspired_weight_update(scores, confidences, prior_weights)
    else:
        # Confidence-scaled priors
        conf_scaled = [p * max(c, 0.1) for p, c in zip(prior_weights, confidences)]
        total_cs = sum(conf_scaled) or 1.0
        weights = [w / total_cs for w in conf_scaled]

    weighted_score = float(np.dot(weights, scores))
    weighted_conf  = float(np.dot(weights, confidences))

    # Conflict detection
    conflicts = _find_conflicts(active, signal_map)
    has_conflict = len(conflicts) > 0

    # If strong conflict exists, pull confidence down
    if has_conflict:
        weighted_conf *= 0.75

    verdict = _verdict_label(weighted_score, weighted_conf, has_conflict)
    fusion_weights = {s: round(float(w), 4) for s, w in zip(available_signals, weights)}
    rationale = _build_rationale(verdict, weighted_score, agent_results, fusion_weights, conflicts)

    return dict(
        verdict=verdict,
        deepfake_probability=float(np.clip(weighted_score, 0.0, 1.0)),
        confidence_in_verdict=float(np.clip(weighted_conf, 0.0, 1.0)),
        fusion_weights=fusion_weights,
        rationale=rationale,
    )
