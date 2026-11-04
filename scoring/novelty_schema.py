
from pydantic import BaseModel, Field
from typing import List, Literal


class NoveltyAssessment(BaseModel):

    # 1. Separate Problem Novelty score (only 5 or 10)
    problem_novelty: Literal[5, 10] = Field(
        description="Independent problem novelty score. Must be either 5 or 10."
    )
    problem_explanation: str = Field(
        description="Explanation supporting the separate problem novelty score."
    )

    # 2. Methodological novelty (1-5)
    methodological_novelty: int = Field(
        ge=1,
        le=5,
        description="Novelty rating for the proposed methodology (1-5)."
    )
    methodological_explanation: str

    # 3. Technical contribution (1-5)
    technical_contribution: int = Field(
        ge=1,
        le=5,
        description="Novelty rating for technical contributions (1-5)."
    )
    technical_explanation: str

    # 4. Application novelty (1-5)
    application_novelty: int = Field(
        ge=1,
        le=5,
        description="Novelty rating for the application (1-5)."
    )
    application_explanation: str

    # 5. Prior-art differentiation (1-5)
    prior_art_differentiation: int = Field(
        ge=1,
        le=5,
        description="How strongly the idea differs from prior work (1-5)."
    )
    differentiation_explanation: str

    # Evidence and comparison
    overlapping_contributions: List[str] = Field(
        default_factory=list
    )
    novel_contributions: List[str] = Field(
        default_factory=list
    )
    supporting_paper_ids: List[str] = Field(
        default_factory=list
    )
    evidence_gaps: List[str] = Field(
        default_factory=list
    )

    # Overall assessment
    confidence: Literal["Low", "Medium", "High"]
    overall_reasoning: str
