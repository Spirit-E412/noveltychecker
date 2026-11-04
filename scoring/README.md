#  Novelty Checker 

## 1. Target Function

**Function Name:** `run_ideanoveltychecker()`

**Module:**

```text
noveltychecker/models/idea_novelty_checker/pipeline.py
```

**Purpose:**

This function initiates the research novelty assessment workflow. It accepts a research idea, retrieves relevant prior-art papers, performs structured novelty assessment using CrewAI and calculates the final weighted novelty score using Python.

---

## 2. Function Signature

```python
run_ideanoveltychecker(
    idea,
    use_retrieval=True,
    input_papers_ids=None,
    input_papers=None,
    ablation=False
)
```

---

## 3. Input Specification

### 3.1 Primary Input: Research Idea

| Property | Specification |
|---|---|
| Parameter | `idea` |
| Data type | String (`str`) |
| Required | Yes |
| Format | Plain-text research idea |
| Description | Research idea, abstract, proposal description or scientific concept to be evaluated. |

The input should contain sufficient technical information to understand the proposed research problem, methodology, mechanisms and expected contributions.

**Example input:**

```python
idea = """
We propose a multimodal cybersecurity threat detection
framework that integrates temporal graph neural networks,
transformer-based attention mechanisms and contrastive
self-supervised learning to identify evolving cyberattacks
in dynamic network environments.
"""
```

The research idea is passed as a plain-text string.

### 3.2 Optional Inputs

| Parameter | Data Type | Default | Description |
|---|---|---|---|
| `use_retrieval` | Boolean | `True` | Enables or disables automatic prior-art retrieval. |
| `input_papers_ids` | List | `None` | Accepts predefined research paper IDs. |
| `input_papers` | DataFrame / list | `None` | Accepts preloaded research paper data. |
| `ablation` | Boolean | `False` | Enables experimental retrieval configurations. |

**Note:** In normal execution, only the research idea is mandatory. The other parameters are optional pipeline-level controls.

---

## 4. Input Format Through Command Line

The application can be triggered through `main.py`.

### Command Format

```bash
python main.py --idea "<RESEARCH_IDEA>" --save_path "<OUTPUT_PATH>"
```

### Example 1: Research Idea as Input

```bash
python main.py --idea "A graph neural network-based framework for detecting evolving cyberattacks using temporal network relationships and attention mechanisms." --save_path "results/novelty_test"
```

### Example 2: Research Idea with Additional Paper IDs

```bash
python main.py --idea "A graph neural network-based framework for detecting evolving cyberattacks using temporal network relationships and attention mechanisms." --papers "220046811,267211735" --save_path "results/novelty_test"
```

### Command-Line Parameters

| Parameter | Description |
|---|---|
| `--idea` | Research idea provided as plain text. |
| `--papers` | Optional comma-separated Semantic Scholar paper IDs. |
| `--save_path` | Directory or path used to save the execution results. |

---

## 5. Python Function Invocation

The pipeline can also be called directly from another Python module.

```python
from noveltychecker.models.idea_novelty_checker.pipeline import (
    run_ideanoveltychecker
)

research_idea = """
A temporal graph neural network framework for detecting
evolving cyberattacks in dynamic network environments.
"""

result = run_ideanoveltychecker(
    idea=research_idea
)

print(result)
```

This is the programmatic interface for integrating the novelty assessment pipeline into another application.

---

## 6. Processing Workflow

After receiving the research idea, the function executes the following workflow:

1. Accepts the research idea.
2. Retrieves relevant prior-art papers.
3. Performs embedding-based ranking.
4. Performs LLM-based re-ranking.
5. Selects candidate evaluation papers.
6. Prepares the evaluation context.
7. Executes the CrewAI-based structured novelty assessment.
8. Calculates the weighted Overall Novelty Score using Python.
9. Returns the structured assessment.

---

## 7. Output Specification

**Return Type:** Python Dictionary (`dict`)

The pipeline returns a dictionary containing the submitted idea, retrieval trace and assessment results.

### Expected Output Structure

```python
{
    "input": "Submitted research idea",

    "trace": {
        # Retrieval-related information
    },

    "output": {
        "default": {
            "evaluation_papers": [
                # Retrieved evaluation papers
            ],

            "novelty_assessment": {
                "novelty_score": 8,
                "problem_novelty_score": 10,

                "methodological_novelty": 4,
                "technical_contribution": 5,
                "application_novelty": 3,
                "prior_art_differentiation": 4,

                "overlapping_contributions": [],
                "novel_contributions": [],
                "supporting_paper_ids": [],
                "evidence_gaps": [],

                "confidence": "Medium",
                "overall_reasoning": "Assessment explanation"
            }
        }
    }
}
```

*Note: The above JSON is an illustrative representation. The exact field names and nesting must match the current implementation's Pydantic schema and scoring service.*

### Output Components

| Output | Description |
|---|---|
| `evaluation_papers` | Retrieved research papers selected for assessment. |
| `novelty_score` | Overall weighted novelty score (1–10). |
| `problem_novelty_score` | Separate Problem Novelty Score (5 or 10). |
| `methodological_novelty` | Methodological novelty rating (1–5). |
| `technical_contribution` | Technical contribution rating (1–5). |
| `application_novelty` | Application novelty rating (1–5). |
| `prior_art_differentiation` | Prior-art differentiation rating (1–5). |
| `overlapping_contributions` | Contributions that overlap with retrieved prior art. |
| `novel_contributions` | Contributions identified as differentiated from retrieved prior art. |
| `supporting_paper_ids` | Paper IDs supporting the assessment. |
| `evidence_gaps` | Areas where available evidence is insufficient. |
| `confidence` | Confidence in the assessment. |
| `overall_reasoning` | Explanation of the assessment. |

---

## 8. Scoring Methodology

The CrewAI agent provides ratings from 1 to 5 for four novelty dimensions.

| Dimension | Weight |
|---|---:|
| Methodological Novelty | 30 |
| Technical Contribution | 30 |
| Application Novelty | 15 |
| Prior-Art Differentiation | 10 |
| Total | 85 |

The Python scoring service calculates the weighted average and normalizes it to a score between 1 and 10.

Problem Novelty is assessed separately and does not contribute to the Overall Novelty Score.

---

## 9. Evidence Handling

- Novelty assessments are based on retrieved prior-art evidence.
- Missing retrieved evidence is not considered proof of novelty.
- Supporting paper IDs should correspond to retrieved evaluation papers.
- If no prior-art papers are available, the system reports insufficient evidence instead of claiming novelty.

---

## 10. Output Location

The execution result is saved using the `--save_path` parameter supplied through the command line.

Example:

```bash
python main.py --idea "Your research idea" --save_path "results/novelty_test"
```

The generated result is stored in the configured result location.

