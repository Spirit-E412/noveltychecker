
import asyncio
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from noveltychecker.utils.load_env import load_env

load_env()

from noveltychecker.models.idea_novelty_checker.paper_collection import (
    get_most_relevant_papers,
)
from noveltychecker.scoring.scoring_service import score_pipeline_result
from noveltychecker.utils.s2_api import get_paper_data


logger = logging.getLogger(__name__)


async def run_ideanoveltychecker(
    idea,
    use_retrieval=True,
    input_papers_ids=None,
    input_papers=None,
    ablation=False,
):
    """
    Run the research novelty assessment pipeline.

    Existing functionality:
        - Research paper retrieval
        - Paper ranking
        - Retrieval trace generation

    New functionality:
        - Overall Novelty Score (four dimensions, out of 10)
        - Separate Problem Novelty Score (5 or 10)
        - Evidence-based novelty assessment

    Binary Novel / Not Novel assessment has been removed.
    """

    if input_papers_ids is None:
        input_papers_ids = []

    seed_papers = {}

    retrieval_trace = {
        "idea_keywords": [],
        "title_keywords": [],
        "idea_priority_facets": "",
        "most_relevant_papers": [],
        "embedding_ranked": [],
        "snippet_papers": [],
        "keyword_papers": [],
    }

    limit = int(os.getenv("NOVELTY_CHECK_TOPkPapers", 10))

    # =====================================================
    # 1. PREPARE INPUT PAPERS
    # =====================================================

    # Preserve user-provided input papers.
    if input_papers is None:

        if len(input_papers_ids) != 0:

            input_papers = await get_paper_data(
                input_papers_ids,
                id_type="paper_id",
                batch_wise=True,
            )

            input_papers = pd.DataFrame(input_papers)

        else:
            input_papers = []

    elif isinstance(input_papers, list):
        input_papers = pd.DataFrame(input_papers)

    # =====================================================
    # 2. PAPER RETRIEVAL
    # =====================================================

    if use_retrieval:

        retrieval_trace = await get_most_relevant_papers(
            idea,
            input_papers,
            max_papers_specter=100,
        )

    retrieval_trace["input_most_relevant_papers"] = input_papers

    # =====================================================
    # 3. INITIALIZE PIPELINE OUTPUT
    # =====================================================

    output = {
        "input": {
            "idea": idea,
            "input_papers_ids": input_papers_ids,
            "use_retrieval": use_retrieval,
        },
        "trace": retrieval_trace,
        "output": {},
    }

    most_relevant_papers = (
        retrieval_trace["most_relevant_papers"].copy()
    )

    # =====================================================
    # 4. DEFINE EXPERIMENTS
    # =====================================================

    if ablation:

        snippet_only_ablation = most_relevant_papers[
            most_relevant_papers["source"].apply(
                lambda x: "snippet" in x
            )
        ]

        keyword_papers_ablation = most_relevant_papers[
            most_relevant_papers["source"].apply(
                lambda x: "keyword" in x
            )
        ]

        experiments = [
            (
                "default",
                retrieval_trace["most_relevant_papers"][:limit],
            ),
            (
                "snippet_only",
                snippet_only_ablation[:limit],
            ),
            (
                "keyword_only",
                keyword_papers_ablation[:limit],
            ),
            (
                "norankGPT",
                retrieval_trace["embedding_ranked"][:limit],
            ),
            (
                "groundtruth_only",
                (
                    retrieval_trace["input_most_relevant_papers"][:limit]
                    if len(
                        retrieval_trace["input_most_relevant_papers"]
                    ) != 0
                    else []
                ),
            ),
            (
                "groundtruth_and_retrieved",
                (
                    pd.concat(
                        [
                            retrieval_trace["input_most_relevant_papers"],
                            retrieval_trace["most_relevant_papers"],
                        ]
                    )[:limit]
                    if (
                        len(
                            retrieval_trace["input_most_relevant_papers"]
                        ) != 0
                        and len(
                            retrieval_trace["most_relevant_papers"]
                        ) != 0
                    )
                    else []
                ),
            ),
        ]

    else:

        if use_retrieval:

            experiments = [
                (
                    "default",
                    retrieval_trace["most_relevant_papers"][:limit],
                )
            ]

        else:

            experiments = [
                (
                    "groundtruth_only",
                    (
                        retrieval_trace["input_most_relevant_papers"][:limit]
                        if len(
                            retrieval_trace["input_most_relevant_papers"]
                        ) != 0
                        else []
                    ),
                )
            ]

    # =====================================================
    # 5. STORE RETRIEVED PAPERS
    # =====================================================
    #
    # Binary novelty assessment has been removed.
    #
    # We only store the retrieved papers for each experiment.
    # The structured scoring layer will use these papers.
    # =====================================================

    for experiment_type, evaluation_papers in tqdm(
        experiments,
        total=len(experiments),
        desc="collecting_retrieved_papers...",
    ):

        if evaluation_papers is None or len(evaluation_papers) == 0:
            continue

        # Work on a copy so the original retrieval data
        # is not modified.
        papers_to_store = evaluation_papers.copy()

        if isinstance(papers_to_store, pd.DataFrame):

            papers_to_store = papers_to_store.drop(
                columns=["embedding"],
                errors="ignore",
            )

            papers_to_store = papers_to_store.to_dict(
                orient="records"
            )

        output["output"][experiment_type] = {
            "evaluation_papers": papers_to_store,
        }

    # =====================================================
    # 6. STRUCTURED NOVELTY SCORING
    # =====================================================
    #
    # SCORE 1:
    # Overall Novelty Score (out of 10)
    #
    # Based ONLY on:
    #   - Methodological Novelty
    #   - Technical Contribution
    #   - Application Novelty
    #   - Prior-Art Differentiation
    #
    # SCORE 2:
    # Separate Problem Novelty Score (5 or 10)
    #
    # Problem Novelty is NOT included in the
    # Overall Novelty Score calculation.
    #
    # Python calculates the weighted overall score.
    # CrewAI provides dimension ratings and explanations.
    # =====================================================

    scoring_experiment = None

    if "default" in output["output"]:
        scoring_experiment = "default"

    elif "groundtruth_only" in output["output"]:
        scoring_experiment = "groundtruth_only"

    if scoring_experiment is not None:

        try:

            print(
                "\nStarting structured novelty assessment..."
            )

            # scoring_service expects the primary experiment
            # under the key "default". For ground-truth-only
            # execution, create a temporary compatible context.
            scoring_input = output

            if scoring_experiment != "default":

                scoring_input = {
                    **output,
                    "output": {
                        **output["output"],
                        "default": output["output"][scoring_experiment],
                    },
                }

            novelty_assessment = await score_pipeline_result(
                idea=idea,
                pipeline_result=scoring_input,
            )

            # Store the structured assessment under the
            # actual experiment used for scoring.
            output["output"][scoring_experiment][
                "novelty_assessment"
            ] = novelty_assessment

            print(
                "\nOverall Novelty Score:",
                novelty_assessment.get("novelty_score"),
                "/10",
            )

            print(
                "Problem Novelty Score:",
                novelty_assessment.get("problem_novelty_score"),
                "/10",
            )

            print(
                "Assessment Status:",
                novelty_assessment.get("status"),
            )

            print(
                "Evidence Confidence:",
                novelty_assessment.get("confidence"),
            )

        except Exception as e:

            print(
                "\nStructured novelty scoring failed:",
                str(e),
            )

            logger.exception(
                "Structured novelty scoring failed."
            )

            output["output"][scoring_experiment][
                "novelty_assessment"
            ] = {
                "status": "scoring_failed",
                "novelty_score": None,
                "maximum_score": 10,
                "problem_novelty_score": None,
                "problem_novelty_maximum": 10,
                "scoring_weights": None,
                "dimension_scores": None,
                "dimension_explanations": None,
                "problem_novelty_explanation": None,
                "overlapping_contributions": [],
                "novel_contributions": [],
                "supporting_paper_ids": [],
                "evidence_gaps": [],
                "confidence": "Low",
                "llm_reported_confidence": None,
                "evidence_validation": None,
                "confidence_reason": (
                    "Structured novelty scoring could not "
                    "be completed."
                ),
                "reasoning": None,
                "error": str(e),
            }

    else:

        logger.warning(
            "No retrieved or supplied papers available for "
            "structured novelty scoring."
        )

    return output


# =====================================================
# 7. LOCAL EXECUTION
# =====================================================

if __name__ == "__main__":

    os.makedirs(
        "ablation_outputs",
        exist_ok=True,
    )

    idea_name = "htm"

    idea = (
        "Hierarchical Topic Models (HTMs) are useful for "
        "discovering topic hierarchies in a collection of "
        "documents. However, traditional HTMs often produce "
        "hierarchies where lowerlevel topics are unrelated "
        "and not specific enough to their higher-level topics. "
        "Additionally, these methods can be computationally "
        "expensive. We present HyHTM - a Hyperbolic geometry "
        "based Hierarchical Topic Models - that addresses "
        "these limitations by incorporating hierarchical "
        "information from hyperbolic geometry to explicitly "
        "model hierarchies in topic models. Experimental "
        "results with four baselines show that HyHTM can "
        "better attend to parent-child relationships among "
        "topics. HyHTM produces coherent topic hierarchies "
        "that specialise in granularity from generic "
        "higher-level topics to specific lowerlevel topics. "
        "Further, our model is significantly faster and "
        "leaves a much smaller memory footprint than our "
        "best-performing baseline. We have made the source "
        "code for our algorithm publicly accessible."
    )

    output = asyncio.run(
        run_ideanoveltychecker(idea)
    )

    os.makedirs(
        f"results/{idea_name}",
        exist_ok=True,
    )

    with open(
        f"results/{idea_name}/result.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False,
            default=str,
        )
