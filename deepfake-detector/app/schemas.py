"""Pydantic schemas for Obscuro Deepimage API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    agent_name: str
    signal_name: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    ran: bool = True
    skipped_reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ForensicVerdict(BaseModel):
    verdict: str
    deepfake_probability: float = Field(ge=0.0, le=1.0)
    confidence_in_verdict: float = Field(ge=0.0, le=1.0)
    agent_results: List[AgentResult]
    fusion_weights: Dict[str, float]
    rationale: str
    forensic_report: Optional[str] = None  # Full specialist report (Markdown)
    processing_time_s: float = 0.0
    media_type: str = "image"
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"


class TrainingStartRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=100)
    samples_per_class: int = Field(default=100, ge=10, le=1000)
    lr: float = Field(default=1e-4, ge=1e-6, le=1e-2)


class EvolutionStartRequest(BaseModel):
    generations: int = Field(default=20, ge=5, le=200)
    population_size: int = Field(default=10, ge=4, le=50)
