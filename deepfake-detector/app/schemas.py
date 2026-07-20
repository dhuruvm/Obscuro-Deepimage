"""
Pydantic schemas for Obscuro Deepimage API.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class AgentResult(BaseModel):
    agent_name: str
    signal_name: str
    score: float          # 0.0 = definitely real, 1.0 = definitely fake
    confidence: float     # How confident this agent is in its own score (0-1)
    details: Dict[str, Any] = {}
    ran: bool = True
    skipped_reason: Optional[str] = None


class ForensicVerdict(BaseModel):
    verdict: str          # "LIKELY FAKE", "LIKELY REAL", "UNCERTAIN"
    deepfake_probability: float  # 0-1
    confidence_in_verdict: float  # 0-1
    agent_results: list[AgentResult]
    fusion_weights: Dict[str, float]
    rationale: str        # Structured prose explanation
    heatmap_available: bool = False
    processing_time_s: float = 0.0
    media_type: str = "image"  # "image" | "video"
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"
