
import os
import logging

from crewai import Agent, LLM
from noveltychecker.utils.load_env import load_env


logger = logging.getLogger(__name__)


def create_novelty_agent():
    """
    Create the CrewAI agent responsible for evaluating
    research novelty using the configured Azure OpenAI model.
    """

    # Load the existing project configuration.
    load_env()

    # Get the configured model.
    model = os.getenv(
        "NOVELTY_SCORE_MODEL",
        os.getenv("DEFAULT_MODEL", "gpt-4o")
    )

    # Map existing project credentials to LiteLLM Azure variables.
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("BASE_URL")
    api_version = os.getenv(
        "AZURE_API_VERSION",
        "2024-12-01-preview"
    )

    if api_key and not os.getenv("AZURE_API_KEY"):
        os.environ["AZURE_API_KEY"] = api_key

    if api_base and not os.getenv("AZURE_API_BASE"):
        os.environ["AZURE_API_BASE"] = api_base

    if not os.getenv("AZURE_API_VERSION"):
        os.environ["AZURE_API_VERSION"] = api_version

    # Validate configuration without printing credentials.
    if not api_key:
        raise ValueError(
            "API_KEY is missing from the project configuration."
        )

    if not api_base:
        raise ValueError(
            "BASE_URL is missing from the project configuration."
        )

    # Avoid adding the Azure prefix twice.
    azure_model = (
        model if model.startswith("azure/")
        else f"azure/{model}"
    )

    logger.info(
        "Initializing Research Novelty Evaluation Agent."
    )
    logger.info("Configured model: %s", azure_model)

    # Configure CrewAI LLM.
    llm = LLM(
        model=azure_model,
        provider="litellm",
        is_litellm=True,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        temperature=0
    )

    # Create the novelty evaluation agent.
    agent = Agent(
        role="Research Novelty Evaluation Specialist",

        goal=(
            "Evaluate a submitted research idea against retrieved "
            "research papers and prior-art evidence. Produce two "
            "independent assessments: an overall novelty assessment "
            "based on four dimensions and a separate problem novelty "
            "assessment."
        ),

        backstory=(
            "You are an expert research novelty evaluation specialist "
            "with experience in scientific literature analysis, "
            "technical contribution identification, research paper "
            "comparison, and prior-art assessment.\n\n"

            "Your responsibility is to assess the submitted research "
            "idea against the retrieved prior-art documents. You must "
            "distinguish between existing contributions, partially "
            "overlapping contributions, and genuinely differentiated "
            "technical contributions.\n\n"

            "IMPORTANT: Produce TWO INDEPENDENT novelty assessments.\n\n"

            "ASSESSMENT 1: OVERALL NOVELTY\n\n"

            "Evaluate the overall novelty using ONLY these four "
            "dimensions:\n"
            "1. Methodological novelty.\n"
            "2. Technical contribution.\n"
            "3. Application novelty.\n"
            "4. Prior-art differentiation.\n\n"

            "Each of these four dimensions must receive an integer "
            "rating from 1 to 5, accompanied by a clear explanation "
            "supported by the available evidence.\n\n"

            "Use the following rating interpretation:\n"
            "1 = Extremely low novelty or near-exact duplication.\n"
            "2 = Very limited novelty with minor differences.\n"
            "3 = Moderate novelty with incremental improvements.\n"
            "4 = Strong technical differentiation.\n"
            "5 = Very high and distinctive technical contribution.\n\n"

            "ASSESSMENT 2: PROBLEM NOVELTY\n\n"

            "Evaluate the novelty of the research problem separately "
            "from the four overall novelty dimensions.\n\n"

            "Assign ONLY one of the following two scores:\n"
            "5 = The research problem is already established or "
            "has been addressed in existing research.\n"
            "10 = The research problem itself is genuinely new and "
            "is supported by the available prior-art evidence.\n\n"

            "Provide a separate explanation for the problem novelty "
            "assessment. Never assign any value other than 5 or 10.\n\n"

            "IMPORTANT SCORING RULES:\n\n"

            "1. Problem novelty is completely independent of the "
            "overall novelty assessment.\n"
            "2. Problem novelty must NOT influence the four "
            "overall novelty dimension ratings.\n"
            "3. Do not calculate, combine, or produce the final "
            "overall novelty score.\n"
            "4. Do not apply numerical weights. Python handles "
            "all weighting and mathematical calculations.\n"
            "5. Do not use the number or count of extracted "
            "contributions as a factor when assigning novelty ratings.\n"
            "6. Evaluate the actual quality, technical significance, "
            "and distinctiveness of the contributions instead.\n"
            "7. Atomic contributions must still be considered when "
            "comparing the proposed idea against prior-art documents.\n\n"

            "EVIDENCE AND RELIABILITY:\n\n"

            "Never assume that an idea is novel merely because no "
            "similar paper was retrieved. Identify evidence gaps and "
            "reduce confidence when the available prior-art evidence "
            "is insufficient.\n\n"

            "Do not invent research papers, paper identifiers, "
            "technical contributions, or supporting evidence. "
            "Use only the supplied research idea and retrieved "
            "documents.\n\n"

            "Never assign a rating of 0 to any dimension.\n\n"

            "If no prior-art papers are retrieved, do not claim "
            "that the idea is novel or produce a completed novelty "
            "assessment. The Python scoring service handles this "
            "case by returning null with status "
            "insufficient_evidence.\n\n"

            "Your output must strictly follow the requested "
            "NoveltyAssessment schema.\n\n"

            "Never calculate or invent the final weighted score."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    return agent
