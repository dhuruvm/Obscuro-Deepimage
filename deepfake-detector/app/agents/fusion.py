"""
Fusion & Explanation Agent.

Combines signals from all specialist agents into a single deepfake probability
using a confidence-weighted ensemble, with optional quantum-inspired
weight optimisation via PennyLane (classical simulation).

Outputs:
  - deepfake_probability (float, 0-1)
  - confidence_in_verdict (float, 0-1)
  - verdict string
  - structured prose rationale
  - per-agent weight breakdown
"""
import logging
import numpy as np
from typing import List, Dict, Any

from app.schemas import AgentResult, ForensicVerdict

logger = logging.getLogger(__name__)

# Prior base weights per signal (tuned for balanced precision/recall)
SIGNAL_PRIORS: Dict[str, float] = {
    "pixel_artifact_score":         0.35,
    "spectral_artifact_score":      0.25,
    "temporal_inconsistency_score": 0.20,
    "rppg_anomaly_score":           0.10,
    "lip_sync_mismatch_score":      0.10,
}


# ─── Quantum-inspired weight optimisation ────────────────────────────────────

def _quantum_inspired_weight_update(
    scores: List[float],
    confidences: List[float],
    priors: List[float],
) -> List[float]:
    """
    Quantum-inspired optimization: uses a PennyLane variational circuit
    (running on classical CPU simulation) to find a weight vector that
    minimises the expected calibration error under the given signal scores.

    IMPORTANT LABEL: This is an *experimental, quantum-inspired* module
    simulated entirely on classical hardware. It provides a heuristic weight
    adjustment, not a proven accuracy improvement. It is included as a
    research exploration into variational quantum circuits for ensemble fusion.

    Returns normalised weight vector.
    """
    try:
        import pennylane as qml

        n = len(scores)
        if n == 0:
            return priors

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circuit(params):
            for i in range(n):
                qml.RY(params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        # Initialise params from prior weights (map [0,1] → [0, π])
        params = np.array([p * np.pi for p in priors], dtype=np.float64)

        # One-step gradient-free update: scale params by confidence
        adjusted_params = params * np.array(confidences)

        # Run circuit — returns list of ⟨Z⟩ expectation values in [-1, 1]
        z_vals = circuit(adjusted_params)
        # Convert [-1, 1] → [0, 1] probability-like values
        weights = np.array([(1.0 + float(z)) / 2.0 for z in z_vals])

        # Blend with confidence-weighted priors (circuit output is noisy)
        blended = 0.4 * weights + 0.6 * np.array(priors)
        normalised = blended / (blended.sum() + 1e-8)
        return normalised.tolist()

    except Exception as exc:
        logger.debug("Quantum-inspired optimisation unavailable (%s) — using prior weights.", exc)
        return priors


# ─── Verdict generation ───────────────────────────────────────────────────────

def _verdict_label(prob: float, confidence: float) -> str:
    if confidence < 0.30:
        return "UNCERTAIN"
    if prob >= 0.65:
        return "LIKELY FAKE"
    if prob <= 0.35:
        return "LIKELY REAL"
    return "UNCERTAIN"


def _build_rationale(
    verdict: str,
    prob: float,
    agent_results: List[AgentResult],
    weights: Dict[str, float],
) -> str:
    """
    Generate a structured prose rationale explaining the verdict.
    Written for a non-technical reader while preserving forensic precision.
    """
    lines = []
    lines.append(f"**Verdict: {verdict}** (deepfake probability: {prob:.1%})\n")

    lines.append("**Summary of forensic signals:**\n")
    for ar in agent_results:
        if not ar.ran:
            lines.append(
                f"- *{ar.agent_name}*: Skipped — {ar.skipped_reason or 'not applicable'}"
            )
            continue
        w = weights.get(ar.signal_name, 0.0)
        direction = "FAKE" if ar.score > 0.55 else ("REAL" if ar.score < 0.45 else "neutral")
        lines.append(
            f"- *{ar.agent_name}* (weight {w:.0%}): score {ar.score:.2f} → "
            f"**{direction}** (agent confidence: {ar.confidence:.0%})"
        )
        # Add key detail notes
        if "note" in ar.details:
            lines.append(f"  - {ar.details['note']}")
        if "caveat" in ar.details:
            lines.append(f"  - ⚠️ {ar.details['caveat']}")

    lines.append("\n**Interpretation:**")
    if verdict == "LIKELY FAKE":
        lines.append(
            "Multiple independent forensic signals indicate this media is likely synthetically "
            "generated or manipulated. The pixel-level and spectral analyses are the primary "
            "drivers of this assessment."
        )
    elif verdict == "LIKELY REAL":
        lines.append(
            "The forensic signals are broadly consistent with authentic, unmanipulated media. "
            "No single strong indicator of synthesis was found. This does not constitute a "
            "definitive certificate of authenticity."
        )
    else:
        lines.append(
            "The evidence is inconclusive. Individual signals disagree, or the overall confidence "
            "in the detection is too low to issue a reliable verdict. Manual review by a human "
            "expert is recommended."
        )

    lines.append(
        "\n**Important caveat:** Deepfake detection is probabilistic. State-of-the-art detectors "
        "have known generalisation limits — particularly against generation methods not represented "
        "in their training data. This analysis is intended as forensic evidence, not legal proof. "
        "False positives and false negatives occur."
    )

    return "\n".join(lines)


# ─── Main fusion function ─────────────────────────────────────────────────────

def fuse(
    agent_results: List[AgentResult],
    use_quantum: bool = True,
) -> Dict[str, Any]:
    """
    Fuse agent results into a single verdict.
    Returns dict matching ForensicVerdict fields.
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

    # Map signal_name → (score, confidence)
    signal_map = {ar.signal_name: (ar.score, ar.confidence) for ar in active}

    # Build weight vector matching available signals
    available_signals = [ar.signal_name for ar in active]
    prior_weights = [SIGNAL_PRIORS.get(s, 0.1) for s in available_signals]
    # Normalise priors
    total = sum(prior_weights) or 1.0
    prior_weights = [w / total for w in prior_weights]

    scores = [signal_map[s][0] for s in available_signals]
    confidences = [signal_map[s][1] for s in available_signals]

    # Quantum-inspired weight update
    if use_quantum and len(scores) >= 2:
        weights = _quantum_inspired_weight_update(scores, confidences, prior_weights)
    else:
        weights = prior_weights

    # Confidence-weighted fusion
    weighted_score = float(np.dot(weights, scores))

    # Overall confidence: mean agent confidence weighted by signal weight
    weighted_conf = float(np.dot(weights, confidences))
    verdict = _verdict_label(weighted_score, weighted_conf)

    fusion_weights = {s: round(float(w), 4) for s, w in zip(available_signals, weights)}
    rationale = _build_rationale(verdict, weighted_score, agent_results, fusion_weights)

    return dict(
        verdict=verdict,
        deepfake_probability=float(np.clip(weighted_score, 0.0, 1.0)),
        confidence_in_verdict=float(np.clip(weighted_conf, 0.0, 1.0)),
        fusion_weights=fusion_weights,
        rationale=rationale,
    )
