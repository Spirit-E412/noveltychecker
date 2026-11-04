
import asyncio
import logging
from typing import Any
from decimal import Decimal, ROUND_HALF_UP

from noveltychecker.scoring.scoring_crew import run_scoring_crew


logger = logging.getLogger(__name__)


# =====================================================
# 1. SCORING WEIGHTS
# =====================================================

# Problem Novelty is intentionally excluded.
# These weights apply ONLY to the Overall Novelty Score.
# Python normalizes them using their total weight of 85.

WEIGHTS = {
    "methodological_novelty": 30,
    "technical_contribution": 30,
    "application_novelty": 15,
    "prior_art_differentiation": 10
}


# =====================================================
# 2. CALCULATE OVERALL NOVELTY SCORE
# =====================================================

def calculate_novelty_score(
    assessment: dict[str, Any]
) -> int:
    """
    Calculate the Overall Novelty Score using only four
    dimensions.

    Problem Novelty is not included in this calculation.

    Dimension ratings: 1-5
    Final score: 1-10
    """

    total_score = Decimal("0")

    total_weight = sum(
        Decimal(str(weight))
        for weight in WEIGHTS.values()
    )

    for dimension, weight in WEIGHTS.items():

        if dimension not in assessment:
            raise ValueError(
                f"Missing scoring dimension: {dimension}"
            )

        rating = assessment[dimension]

        if (
            isinstance(rating, bool)
            or not isinstance(rating, (int, float))
            or not 1 <= rating <= 5
        ):
            raise ValueError(
                f"Invalid rating for {dimension}: {rating}. "
                "Rating must be between 1 and 5."
            )

        total_score += (
            Decimal(str(rating)) * Decimal(str(weight))
        )

    # Normalize the four dimension weights to obtain
    # a weighted average in the 1-5 range.
    weighted_rating = total_score / total_weight

    # Convert the weighted average from 1-5 to 1-10.
    final_score = (
        Decimal("1")
        + (
            (weighted_rating - Decimal("1"))
            * Decimal("9")
            / Decimal("4")
        )
    ).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP
    )

    return int(final_score)


# =====================================================
# 3. EXTRACT EXISTING PIPELINE RESULTS
# =====================================================

def prepare_pipeline_context(
    pipeline_result: dict[str, Any]
) -> dict[str, Any]:

    if not isinstance(pipeline_result, dict):
        raise TypeError(
            "Pipeline result must be a dictionary."
        )

    outputs = pipeline_result.get("output", {})

    if not isinstance(outputs, dict):
        outputs = {}

    # Prefer the default experiment.
    experiment = outputs.get("default")

    # Fallback to another available experiment.
    if not isinstance(experiment, dict):

        experiment = next(
            (
                value
                for value in outputs.values()
                if isinstance(value, dict)
                and "evaluation_papers" in value
            ),
            {}
        )

    if not isinstance(experiment, dict):
        experiment = {}

    # Extract retrieved and ranked papers.
    papers = experiment.get("evaluation_papers") or []

    if not isinstance(papers, list):
        papers = []

    selected_papers = []

    for paper in papers[:10]:

        if not isinstance(paper, dict):
            continue

        selected_papers.append({

            "paperId": paper.get("paperId"),

            "title": paper.get("title"),

            "abstract": (
                paper.get("abstract") or ""
            )[:2000],

            "year": paper.get("year"),

            "venue": paper.get("venue"),

            "url": paper.get("url")

        })

    # Extract retrieval trace.
    trace = pipeline_result.get("trace", {})

    if not isinstance(trace, dict):
        trace = {}

    return {

        "existing_category": experiment.get("category"),

        "existing_review": experiment.get("review"),

        "existing_output": experiment.get(
            "output_novelty_text"
        ),

        "retrieved_papers": selected_papers,

        "retrieval_keywords": trace.get(
            "idea_keywords", []
        ),

        "retrieval_facets": trace.get(
            "idea_priority_facets", ""
        ),

        "number_of_retrieved_papers": len(
            selected_papers
        )

    }


# =====================================================
# 4. VALIDATE CREWAI ASSESSMENT
# =====================================================

def validate_assessment(
    assessment: dict[str, Any]
) -> None:

    if not isinstance(assessment, dict):
        raise TypeError(
            "CrewAI assessment must be a dictionary."
        )

    # Four dimensions used for Overall Novelty Score.
    required_fields = [

        *WEIGHTS.keys(),

        "problem_novelty",
        "problem_explanation",

        "methodological_explanation",
        "technical_explanation",
        "application_explanation",
        "differentiation_explanation",

        "overlapping_contributions",
        "novel_contributions",
        "supporting_paper_ids",
        "evidence_gaps",

        "confidence",
        "overall_reasoning"

    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in assessment
    ]

    if missing_fields:
        raise ValueError(
            f"Missing assessment fields: {missing_fields}"
        )

    # Validate the four Overall Novelty dimensions.
    for dimension in WEIGHTS:

        rating = assessment[dimension]

        if (
            isinstance(rating, bool)
            or not isinstance(rating, (int, float))
            or not 1 <= rating <= 5
        ):
            raise ValueError(
                f"Invalid rating for {dimension}: {rating}. "
                "Rating must be between 1 and 5."
            )

    # Validate separate Problem Novelty score.
    problem_score = assessment["problem_novelty"]

    if (
        isinstance(problem_score, bool)
        or not isinstance(problem_score, int)
        or problem_score not in (5, 10)
    ):
        raise ValueError(
            f"Invalid Problem Novelty score: {problem_score}. "
            "Problem Novelty must be either 5 or 10."
        )

    # Validate list fields.
    list_fields = [

        "overlapping_contributions",
        "novel_contributions",
        "supporting_paper_ids",
        "evidence_gaps"

    ]

    for field in list_fields:

        if not isinstance(assessment[field], list):
            raise ValueError(
                f"{field} must be a list."
            )

        if not all(
            isinstance(item, str)
            for item in assessment[field]
        ):
            raise ValueError(
                f"Every item in {field} must be a string."
            )

    if assessment["confidence"] not in [
        "Low", "Medium", "High"
    ]:
        raise ValueError(
            "Confidence must be Low, Medium, or High."
        )


# =====================================================
# 5. EVIDENCE VALIDATION
# =====================================================

def validate_evidence_coverage(
    context: dict[str, Any],
    assessment: dict[str, Any]
) -> dict[str, Any]:
    """
    Check whether the assessment references papers
    that actually exist in the retrieved-paper context.

    This validates reference coverage, not the scientific
    correctness of the LLM's claims.
    """

    retrieved_papers = context.get(
        "retrieved_papers", []
    )

    # Collect actual retrieved paper IDs.
    available_ids = {

        str(paper["paperId"])

        for paper in retrieved_papers

        if paper.get("paperId") is not None

    }

    # IDs supplied by the LLM.
    reported_ids = {

        str(paper_id)

        for paper_id in assessment.get(
            "supporting_paper_ids", []
        )

    }

    # Identify matching and unsupported references.
    matched_ids = available_ids.intersection(
        reported_ids
    )

    unmatched_ids = reported_ids.difference(
        available_ids
    )

    evidence_gaps = assessment.get(
        "evidence_gaps", []
    )

    overlaps = assessment.get(
        "overlapping_contributions", []
    )

    novel_contributions = assessment.get(
        "novel_contributions", []
    )

    retrieved_count = len(retrieved_papers)

    # Basic evidence coverage classification.
    if (
        retrieved_count < 2
        or not matched_ids
        or not overlaps
    ):

        evidence_status = "insufficient"

    elif (
        retrieved_count >= 5
        and len(matched_ids) >= 2
        and not evidence_gaps
        and not unmatched_ids
        and overlaps
        and novel_contributions
    ):

        evidence_status = "adequate"

    else:

        evidence_status = "partial"

    return {

        "evidence_status": evidence_status,

        "retrieved_paper_count": retrieved_count,

        "reported_supporting_paper_count": len(
            reported_ids
        ),

        "matched_supporting_paper_count": len(
            matched_ids
        ),

        "matched_supporting_paper_ids": sorted(
            matched_ids
        ),

        "unmatched_supporting_paper_ids": sorted(
            unmatched_ids
        ),

        "has_overlapping_contributions": bool(
            overlaps
        ),

        "has_novel_contributions": bool(
            novel_contributions
        ),

        "reported_evidence_gap_count": len(
            evidence_gaps
        )

    }


# =====================================================
# 6. DETERMINE FINAL CONFIDENCE
# =====================================================

def determine_confidence(
    evidence_validation: dict[str, Any]
) -> tuple[str, str]:
    """
    Determine confidence using evidence coverage.

    The LLM-reported confidence is not used as the
    final confidence value.
    """

    status = evidence_validation["evidence_status"]

    if status == "insufficient":

        return (
            "Low",
            (
                "Evidence coverage is insufficient. "
                "There are too few retrieved documents, "
                "no matched supporting references, or "
                "no identified overlapping contributions."
            )
        )

    if status == "adequate":

        return (
            "High",
            (
                "The initial evidence coverage checks are "
                "satisfied: multiple retrieved documents, "
                "matched supporting references, identified "
                "overlap and differentiated contributions, "
                "and no reported evidence gaps. This does "
                "not independently establish completeness "
                "of the literature search."
            )
        )

    return (
        "Medium",
        (
            "Some relevant prior-art evidence is available, "
            "but the evidence coverage checks are not "
            "sufficient for high confidence. Further "
            "literature comparison or evidence is needed."
        )
    )


# =====================================================
# 7. MAIN SCORING SERVICE
# =====================================================

async def score_pipeline_result(
    idea: str,
    pipeline_result: dict[str, Any]
) -> dict[str, Any]:

    if not idea or not idea.strip():
        raise ValueError(
            "Research idea cannot be empty."
        )

    # Prepare information from the existing pipeline.
    context = prepare_pipeline_context(
        pipeline_result
    )

    retrieved_count = context[
        "number_of_retrieved_papers"
    ]

    # Do not assign either score without prior-art evidence.
    if not context["retrieved_papers"]:

        return {

            "status": "insufficient_evidence",

            # Overall Novelty Score.
            "novelty_score": None,
            "maximum_score": 10,

            # Separate Problem Novelty Score.
            "problem_novelty_score": None,
            "problem_novelty_maximum": 10,

            "scoring_weights": WEIGHTS,

            "dimension_scores": None,

            "problem_novelty_explanation": None,

            "dimension_explanations": None,

            "overlapping_contributions": [],

            "novel_contributions": [],

            "supporting_paper_ids": [],

            "evidence_gaps": [
                "No retrieved prior-art papers available."
            ],

            "confidence": "Low",

            "llm_reported_confidence": None,

            "evidence_validation": {

                "evidence_status": "insufficient",

                "retrieved_paper_count": 0,

                "matched_supporting_paper_count": 0

            },

            "confidence_reason": (
                "No retrieved prior-art documents are "
                "available for comparison."
            ),

            "reasoning": (
                "Reliable overall and problem novelty "
                "assessments cannot be produced because "
                "no prior-art papers were available."
            )

        }

    logger.info(
        "Starting CrewAI assessment with %d papers.",
        retrieved_count
    )

    # Execute synchronous CrewAI inside a worker thread.
    assessment = await asyncio.to_thread(
        run_scoring_crew,
        idea,
        context
    )

    # Validate structured agent output.
    validate_assessment(assessment)

    # Calculate ONLY the four-dimension Overall Novelty Score.
    novelty_score = calculate_novelty_score(
        assessment
    )

    # Extract the independent Problem Novelty Score.
    problem_novelty_score = assessment["problem_novelty"]

    # Independently validate evidence references.
    evidence_validation = validate_evidence_coverage(
        context,
        assessment
    )

    # Determine final confidence using Python rules.
    final_confidence, confidence_reason = (
        determine_confidence(
            evidence_validation
        )
    )

    logger.info(
        "Final Overall Novelty Score: %d/10",
        novelty_score
    )

    logger.info(
        "Separate Problem Novelty Score: %d/10",
        problem_novelty_score
    )

    logger.info(
        "Final evidence-based confidence: %s",
        final_confidence
    )

    # Return the two independent assessments.
    return {

        "status": "completed",

        # SCORE 1: Overall Novelty (four dimensions only).
        "novelty_score": novelty_score,

        "maximum_score": 10,

        # SCORE 2: Independent Problem Novelty.
        "problem_novelty_score": problem_novelty_score,

        "problem_novelty_maximum": 10,

        # Only four dimensions are included here.
        "scoring_weights": WEIGHTS,

        "dimension_scores": {

            key: assessment[key]

            for key in WEIGHTS

        },

        # Explanations for the four overall dimensions.
        "dimension_explanations": {

            "methodological_novelty":
                assessment["methodological_explanation"],

            "technical_contribution":
                assessment["technical_explanation"],

            "application_novelty":
                assessment["application_explanation"],

            "prior_art_differentiation":
                assessment["differentiation_explanation"]

        },

        # Separate Problem Novelty explanation.
        "problem_novelty_explanation":
            assessment["problem_explanation"],

        "overlapping_contributions":
            assessment["overlapping_contributions"],

        "novel_contributions":
            assessment["novel_contributions"],

        "supporting_paper_ids":
            assessment["supporting_paper_ids"],

        "evidence_gaps":
            assessment["evidence_gaps"],

        "confidence":
            final_confidence,

        "llm_reported_confidence":
            assessment["confidence"],

        "confidence_reason":
            confidence_reason,

        "evidence_validation":
            evidence_validation,

        "reasoning":
            assessment["overall_reasoning"]

    }
