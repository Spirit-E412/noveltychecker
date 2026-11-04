
import logging

from crewai import Crew, Process

from noveltychecker.scoring.novelty_agent import (
    create_novelty_agent
)

from noveltychecker.scoring.novelty_task import (
    create_novelty_task
)


logger = logging.getLogger(__name__)


def run_scoring_crew(idea, pipeline_results):

    # Step 1: Create the novelty evaluation agent
    agent = create_novelty_agent()

    # Step 2: Create the evaluation task
    task = create_novelty_task(
        agent=agent,
        idea=idea,
        pipeline_results=pipeline_results
    )

    # Step 3: Create CrewAI workflow
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    logger.info("Starting CrewAI novelty assessment...")

    # Step 4: Execute the Crew
    result = crew.kickoff()

    # Step 5: Extract structured output
    assessment = result.pydantic

    if assessment is None:
        logger.error(
            "CrewAI failed to generate structured output."
        )
        raise ValueError(
            "CrewAI returned no structured novelty assessment."
        )

    # Step 6: Convert Pydantic model into dictionary
    assessment_dict = assessment.model_dump()

    logger.info(
        "CrewAI novelty assessment completed successfully."
    )

    return assessment_dict
