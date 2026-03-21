# Legal RAG Evaluator

This project is a Streamlit app for comparing Legal RAG pipeline outputs across multiple retrieval and generation approaches.

The app lets you:
- compare answers from different pipelines for the same legal question
- inspect LLM judge scores such as correctness, completeness, groundedness, and hallucination
- view retrieved context chunks and latency breakdowns
- analyze which questions all models did well on or struggled with
- review the best and weakest questions for each pipeline

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

## Notes

The JSON result files are included in this repository because the app reads them directly at runtime.
