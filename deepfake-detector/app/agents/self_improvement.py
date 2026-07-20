"""
Self-Improvement Agent — continual monitoring and recalibration.

Responsibilities:
  1. Log every prediction with timestamp, input hash, verdict, and per-signal scores.
  2. Track accuracy drift when ground-truth labels are provided (feedback loop).
  3. Periodically recalibrate Fusion Agent signal weights based on recent performance.
  4. Flag its own degradation when recent accuracy drops below threshold.

This is a *bounded, auditable* self-improvement loop — not unbounded recursive 
capability gain. Changes are logged and reversible.
"""
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from app.schemas import AgentResult, ForensicVerdict

logger = logging.getLogger(__name__)

LOG_PATH = Path("logs/predictions.jsonl")
RECAL_PATH = Path("logs/weight_recalibration.json")
MIN_SAMPLES_FOR_RECAL = 20
DEGRADATION_THRESHOLD = 0.60  # Accuracy below this triggers a warning


def _ensure_log_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(
    input_hash: str,
    verdict: ForensicVerdict,
    ground_truth: Optional[str] = None,  # "fake" | "real" | None
) -> None:
    """Append a prediction record to the JSONL log."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
        "verdict": verdict.verdict,
        "deepfake_probability": round(verdict.deepfake_probability, 4),
        "confidence": round(verdict.confidence_in_verdict, 4),
        "ground_truth": ground_truth,
        "agent_scores": {
            ar.agent_name: {
                "score": round(ar.score, 4),
                "confidence": round(ar.confidence, 4),
                "ran": ar.ran,
            }
            for ar in verdict.agent_results
        },
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_recent_logs(n: int = 100) -> List[Dict[str, Any]]:
    """Load last N prediction records from log."""
    _ensure_log_dir()
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().split("\n")
    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def compute_accuracy_report() -> Dict[str, Any]:
    """
    Compute accuracy metrics from labelled prediction logs.
    Returns report dict with accuracy, F1, false-positive/negative rates,
    and a degradation flag.
    """
    records = _load_recent_logs(200)
    labelled = [r for r in records if r.get("ground_truth") in ("fake", "real")]

    if len(labelled) < 5:
        return {
            "message": f"Insufficient labelled samples ({len(labelled)}) for accuracy report.",
            "total_predictions": len(records),
            "labelled_predictions": len(labelled),
            "degradation_flag": False,
        }

    tp = fp = tn = fn = 0
    for r in labelled:
        predicted_fake = r["deepfake_probability"] >= 0.5
        actual_fake = r["ground_truth"] == "fake"
        if predicted_fake and actual_fake:
            tp += 1
        elif predicted_fake and not actual_fake:
            fp += 1
        elif not predicted_fake and actual_fake:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / len(labelled)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    degradation_flag = accuracy < DEGRADATION_THRESHOLD
    if degradation_flag:
        logger.warning(
            "Self-Improvement Agent: DEGRADATION DETECTED — accuracy %.2f < threshold %.2f. "
            "Model recalibration recommended.",
            accuracy, DEGRADATION_THRESHOLD,
        )

    return {
        "total_predictions": len(records),
        "labelled_predictions": len(labelled),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "degradation_flag": degradation_flag,
        "degradation_threshold": DEGRADATION_THRESHOLD,
    }


def recalibrate_weights_if_needed() -> Optional[Dict[str, float]]:
    """
    If enough labelled data exists, compute per-signal correlations with ground
    truth and suggest adjusted fusion weights.
    Writes suggestions to RECAL_PATH.
    Returns suggested weights or None if insufficient data.
    """
    records = _load_recent_logs(200)
    labelled = [r for r in records if r.get("ground_truth") in ("fake", "real")]

    if len(labelled) < MIN_SAMPLES_FOR_RECAL:
        return None

    # Compute per-agent point-biserial correlation with ground truth
    agent_names = list(labelled[0].get("agent_scores", {}).keys())
    gt_binary = [1.0 if r["ground_truth"] == "fake" else 0.0 for r in labelled]

    suggestions = {}
    for agent in agent_names:
        agent_scores = [r["agent_scores"].get(agent, {}).get("score", 0.5) for r in labelled]
        if len(set(agent_scores)) < 2:
            suggestions[agent] = 0.1
            continue
        corr = float(abs(
            (len(gt_binary) * sum(a * b for a, b in zip(agent_scores, gt_binary))
             - sum(agent_scores) * sum(gt_binary))
            / (
                (len(gt_binary) * sum(s ** 2 for s in agent_scores) - sum(agent_scores) ** 2 + 1e-8)
                ** 0.5
                * (len(gt_binary) * sum(g ** 2 for g in gt_binary) - sum(gt_binary) ** 2 + 1e-8)
                ** 0.5
            )
        ))
        suggestions[agent] = round(max(0.05, corr), 4)

    # Normalise
    total = sum(suggestions.values()) or 1.0
    normalised = {k: round(v / total, 4) for k, v in suggestions.items()}

    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples_used": len(labelled),
        "suggested_weights": normalised,
    }
    with open(RECAL_PATH, "w") as f:
        json.dump(record, f, indent=2)

    logger.info("Self-Improvement Agent: weight recalibration complete → %s", normalised)
    return normalised


def make_input_hash(data: bytes) -> str:
    """Compute SHA-256 hash of raw input bytes for logging."""
    return hashlib.sha256(data).hexdigest()[:16]
