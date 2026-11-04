
from crewai import Task

from noveltychecker.scoring.novelty_schema import (
    NoveltyAssessment
)


def create_novelty_task(agent, idea, pipeline_results):

    description = f"""
    You are evaluating the novelty of a submitted research idea.

    ==================================================
    1. ORIGINAL RESEARCH IDEA
    ==================================================

    {idea}

    ==================================================
    2. EXISTING PIPELINE RESULTS
    ==================================================

    {pipeline_results}

    ==================================================
    3. YOUR EVALUATION RESPONSIBILITIES
    ==================================================

    A. Understand the research idea.

    Identify its:
    - Research problem
    - Proposed methodology
    - Technical mechanisms
    - Expected contributions
    - Intended application

    B. Analyze existing research papers.

    Examine the retrieved papers and available prior-art
    evidence to understand the existing research landscape.

    C. Identify overlapping contributions.

    Determine which technical aspects of the submitted idea
    have already been addressed by previous research.

    D. Identify differentiated contributions.

    Explain what is technically different from existing work.

    ==================================================
    4. NOVELTY ASSESSMENT
    ==================================================

    Produce TWO INDEPENDENT novelty assessments.

    ASSESSMENT A: OVERALL NOVELTY

    Evaluate ONLY the following four dimensions:

    1. Methodological Novelty

       Evaluate whether the proposed methodology introduces
       a new approach or meaningfully improves existing methods.

    2. Technical Contribution

       Evaluate the uniqueness and technical significance
       of the proposed technical mechanisms and contributions.

    3. Application Novelty

       Evaluate whether the proposed application introduces
       meaningful new technical value.

    4. Prior-Art Differentiation

       Evaluate how clearly the proposed idea differs from
       the retrieved prior research.

    Each of these four dimensions must receive an integer
    rating from 1 to 5.

    ASSESSMENT B: PROBLEM NOVELTY

    Evaluate the research problem independently of the
    four overall novelty dimensions.

    Assign ONLY one of these two scores:

    5 = The research problem is already established or has
        been addressed in existing research.

    10 = The research problem itself is genuinely new,
         supported by the available prior-art evidence.

    Provide a separate explanation for this assessment.

    Never assign Problem Novelty a value other than 5 or 10.

    IMPORTANT:

    Problem Novelty must NOT be included in the overall
    novelty assessment or influence its four dimension ratings.

    ==================================================
    5. RATING GUIDELINES
    ==================================================

    For the four overall novelty dimensions, use only
    integer ratings from 1 to 5.

    1 = Extremely low novelty.
        The idea has near-exact overlap with existing research
        and no meaningful differentiation.

    2 = Very limited novelty.
        The idea has substantial overlap with prior work
        and introduces only minor differences.

    3 = Moderate novelty.
        The idea introduces meaningful incremental
        improvements over existing research.

    4 = Strong novelty.
        The idea demonstrates substantial technical
        differentiation from retrieved prior work.

    5 = Very high novelty.
        The idea introduces highly distinctive technical
        contributions with strong evidence of differentiation.

    Do not assign a rating of 0.

    For Problem Novelty, use only 5 or 10 as defined above.

    ==================================================
    6. IMPORTANT EVALUATION RULES
    ==================================================

    - Compare actual technical contributions, not just words.
    - Do not consider a different title as novelty.
    - Do not automatically award novelty for a new application.
    - Do not invent paper references.
    - Use actual paper IDs from the supplied evidence.
    - Do not assume missing papers prove novelty.
    - If evidence is insufficient, explicitly report it.
    - Separate overlapping contributions from differentiated
      contributions.
    - Give explanations for all five assessments.
    - Do not give a high rating simply because a contribution
      is not found in the retrieved papers.
    - Distinguish genuine technical novelty from incremental
      improvements.
    - Base every rating on the available retrieved evidence.
    - Do not use the number or count of extracted contributions
      as a factor when assigning any novelty rating.
    - Evaluate contribution quality and distinctiveness rather
      than contribution quantity.
    - Continue using atomic contributions for technical
      comparison with prior-art documents.

    ==================================================
    7. EVIDENCE AND CONFIDENCE
    ==================================================

    Return:
    - Overlapping contributions.
    - Differentiated or novel contributions.
    - Supporting paper IDs.
    - Evidence gaps.
    - Confidence level.
    - Overall reasoning.

    Use only the supplied research idea and retrieved papers.
    Do not invent supporting evidence.

    If no prior-art papers are retrieved, the system must not
    claim that the idea is novel or produce a completed score.
    The Python scoring service will handle this case by
    returning null with status insufficient_evidence.

    ==================================================
    8. EXPECTED OUTPUT
    ==================================================

    Return a structured NoveltyAssessment containing:

    A. Overall Novelty dimensions:
       - Methodological novelty (1-5).
       - Methodological explanation.
       - Technical contribution (1-5).
       - Technical explanation.
       - Application novelty (1-5).
       - Application explanation.
       - Prior-art differentiation (1-5).
       - Differentiation explanation.

    B. Separate Problem Novelty:
       - Problem novelty score (only 5 or 10).
       - Problem explanation.

    C. Evidence and assessment:
       - Overlapping contributions.
       - Novel contributions.
       - Supporting paper IDs.
       - Evidence gaps.
       - Confidence.
       - Overall reasoning.

    ==================================================
    9. SCORING RESPONSIBILITIES
    ==================================================

    Do not calculate the final overall novelty score.

    Do not calculate or apply numerical weights.

    Python will independently calculate the Overall Novelty
    Score using the four overall novelty dimensions and
    predefined weights.

    Problem Novelty will be reported as a separate score.
    It will NOT contribute to the Overall Novelty Score.

    Do not combine these two scores.

    Never calculate or invent either final score yourself.
    """

    task = Task(
        description=description,

        expected_output=(
            "A structured NoveltyAssessment containing four "
            "overall novelty dimension ratings from 1 to 5, "
            "a separate problem novelty score of either 5 or 10, "
            "explanations for all five assessments, overlapping "
            "contributions, novel contributions, supporting "
            "paper IDs, evidence gaps, confidence and "
            "overall reasoning."
        ),

        agent=agent,

        output_pydantic=NoveltyAssessment
    )

    return task
