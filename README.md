# Legal RAG Evaluator

This project is a Streamlit app for comparing Legal RAG pipeline outputs across multiple retrieval and generation approaches.

The app lets you:
- compare answers from different pipelines for the same legal question
- inspect LLM judge scores such as correctness, completeness, groundedness, and hallucination
- view retrieved context chunks and latency breakdowns
- analyze which questions all models did well on or struggled with
- review the best and weakest questions for each pipeline

## How The Evaluation Works

The project uses an LLM-as-a-judge workflow implemented in `llm_judge_evaluation.py`.

For each generated answer, the judge model compares:
- the user question
- the reference answer
- the generated answer
- the retrieved context, when available

The judge is prompted to return structured JSON with the following metrics:

- `correctness` from 0 to 5
  Measures whether the generated answer is factually and legally correct compared with the reference answer.
- `completeness` from 0 to 5
  Measures how fully the answer covers the important points in the reference answer.
- `groundedness` from 0 to 5, or `null`
  Measures whether the generated answer is supported by the retrieved context. If no context is available, this is set to `null`.
- `hallucination` from 0 to 5
  Measures whether the answer invents unsupported claims. Higher is better, meaning fewer hallucinations.
- `overall_score` from 0 to 10
  A holistic score where correctness is weighted most heavily.

The judge also returns:

- `verdict`: one of `excellent`, `good`, `fair`, `poor`, or `fail`
- `matched_facts`: short facts correctly captured by the answer
- `missing_facts`: important points omitted from the answer
- `unsupported_claims`: statements not supported by the reference answer or retrieved context
- `rationale`: a short explanation for the judgment

## Verdict Mapping

In the evaluation script, the final verdict is normalized from `overall_score` using these rules:

- `9-10`: `excellent`
- `7-8`: `good`
- `5-6`: `fair`
- `3-4`: `poor`
- `0-2`: `fail`

This means the displayed verdict is derived from the final overall score after sanitization.

## What The Charts And Tables Mean

- `avg_overall_score`
  Mean of the judge's `overall_score` across all judged questions for a pipeline.
- `avg_correctness`
  Mean correctness score across judged questions.
- `avg_groundedness`
  Mean groundedness score across questions where grounding could be evaluated.
- `coverage_pct`
  Percentage of pipeline outputs that already have judge results.
- `score_gap`
  Difference between the best and worst model scores for the same question. A higher gap means stronger disagreement between models.
- `strong` and `weak` question thresholds
  In the app, these are user-configurable. A question is considered strong only if every selected model scores at or above the chosen threshold, and weak only if every selected model scores at or below the chosen threshold.

## Important Interpretation Notes

- These scores come from an LLM judge, not a human evaluator.
- `overall_score` is not a simple average of the other metrics.
- A high `hallucination` score is good because it means fewer unsupported claims.
- `groundedness` can be `null` when there is no retrieved context to check against.
- The script truncates long answers and long retrieved context before sending them to the judge, so evaluation is based on a bounded prompt window.

## Project Files

- `app.py`: main Streamlit application
- `evaluation_results_cohere.json`: evaluation results for advanced pipelines
- `legal_rag_naive_results.json`: evaluation results for naive pipelines
- `llm_judge_results/`: judge outputs for advanced pipelines
- `llm_judge_results_naive/`: judge outputs for naive pipelines
- `llm_judge_evaluation.py`: script for running LLM-as-a-judge evaluation
- `requirements.txt`: Python dependencies

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

## Streamlit Cloud Deployment

When deploying on Streamlit Community Cloud:

- connect this GitHub repository
- set the main file path to `app.py`

Because the app uses:

```python
BASE_DIR = Path(__file__).resolve().parent
```

all data files are loaded from the same folder as `app.py`, which makes deployment easier.

## Features

- pipeline overview table
- question browser with search and sorting
- side-by-side answer comparison
- configurable thresholds for strong and weak model performance
- question-level performance insights across selected pipelines

## Requirements

- Python 3.10+
- Streamlit
- pandas
- plotly

## Notes

The JSON result files are included in this repository because the app reads them directly at runtime.
