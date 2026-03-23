from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAW_RESULT_SOURCES = [
    BASE_DIR / "fixed_evaluation_results_all_pipelines.json",
    BASE_DIR / "evaluation_results_all_pipelines.json",
]
JUDGE_RESULT_DIRS = [
    BASE_DIR / "llm_judge_results_fixed",
    BASE_DIR / "llm_judge_results",
]
PIPELINE_LABELS = {
    "baseline": "Baseline",
    "naive_section": "Naive Dense (Section)",
    "naive_paragraph": "Naive Dense (Paragraph)",
    "advanced_dense_rerank_section": "Advanced Dense + Rerank (Section)",
    "advanced_hybrid_rerank_section": "Advanced Hybrid + Rerank (Section)",
    "advanced_hybrid_rerank_paragraph": "Advanced Hybrid + Rerank (Paragraph)",
}


def pipeline_label(name: str) -> str:
    return PIPELINE_LABELS.get(name, name)


def load_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_context_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = record.get("context_chunks")
    if isinstance(chunks, list):
        return chunks

    chunks = record.get("retrieved_context")
    if isinstance(chunks, list):
        return chunks

    return []


def format_latency(latency: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "retrieval": latency.get("retrieval"),
        "reranking": latency.get("reranking"),
        "generation": latency.get("generation"),
        "total": latency.get("total"),
    }


def build_question_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(record.get("case_citation", "")).strip(),
        str(record.get("question", "")).strip(),
    )


@st.cache_data(show_spinner=False)
def load_raw_records() -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    records_by_pipeline: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}

    for source in RAW_RESULT_SOURCES:
        if not source.exists():
            continue

        for pipeline, records in load_json(source).items():
            pipeline_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for record in records:
                normalized = dict(record)
                normalized["pipeline"] = pipeline
                normalized["context_chunks"] = get_context_chunks(record)
                normalized["latency"] = format_latency(record.get("latency", {}))
                pipeline_records[build_question_key(normalized)] = normalized
            records_by_pipeline[pipeline] = pipeline_records

    return records_by_pipeline


@st.cache_data(show_spinner=False)
def load_judge_records() -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    judges_by_pipeline: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}

    for directory in JUDGE_RESULT_DIRS:
        if not directory.exists():
            continue

        for path in sorted(directory.glob("*.json")):
            payload = load_json(path)
            for pipeline, records in payload.items():
                pipeline_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for record in records:
                    pipeline_records[build_question_key(record)] = record.get("judge", {})
                judges_by_pipeline[pipeline] = pipeline_records

    return judges_by_pipeline


@st.cache_data(show_spinner=False)
def build_app_dataset() -> Dict[str, Any]:
    raw_records = load_raw_records()
    judge_records = load_judge_records()

    pipeline_names = sorted(raw_records.keys(), key=lambda item: pipeline_label(item).lower())
    question_keys = sorted(
        {
            key
            for pipeline_records in raw_records.values()
            for key in pipeline_records.keys()
        },
        key=lambda item: (item[0], item[1]),
    )

    overview_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []

    for pipeline in pipeline_names:
        rows = raw_records[pipeline]
        judges = judge_records.get(pipeline, {})
        overall_scores = [
            judge.get("overall_score")
            for judge in judges.values()
            if isinstance(judge.get("overall_score"), (int, float))
        ]
        correctness_scores = [
            judge.get("correctness")
            for judge in judges.values()
            if isinstance(judge.get("correctness"), (int, float))
        ]
        groundedness_scores = [
            judge.get("groundedness")
            for judge in judges.values()
            if isinstance(judge.get("groundedness"), (int, float))
        ]
        total_latencies = [
            record.get("latency", {}).get("total")
            for record in rows.values()
            if isinstance(record.get("latency", {}).get("total"), (int, float))
        ]

        overview_rows.append(
            {
                "pipeline": pipeline,
                "label": pipeline_label(pipeline),
                "questions": len(rows),
                "judged": len(judges),
                "coverage_pct": round((len(judges) / len(rows)) * 100, 1) if rows else 0.0,
                "avg_overall_score": round(sum(overall_scores) / len(overall_scores), 2)
                if overall_scores
                else None,
                "avg_correctness": round(sum(correctness_scores) / len(correctness_scores), 2)
                if correctness_scores
                else None,
                "avg_groundedness": round(sum(groundedness_scores) / len(groundedness_scores), 2)
                if groundedness_scores
                else None,
                "avg_latency_total_s": round(sum(total_latencies) / len(total_latencies), 2)
                if total_latencies
                else None,
            }
        )

    for case_citation, question in question_keys:
        row: Dict[str, Any] = {
            "case_citation": case_citation,
            "question": question,
        }
        for pipeline in pipeline_names:
            raw = raw_records.get(pipeline, {}).get((case_citation, question))
            judge = judge_records.get(pipeline, {}).get((case_citation, question), {})
            row[f"{pipeline}__answer"] = raw.get("generated_answer") if raw else None
            row[f"{pipeline}__overall"] = judge.get("overall_score") if judge else None
            row[f"{pipeline}__verdict"] = judge.get("verdict") if judge else None
        question_rows.append(row)

    return {
        "pipelines": pipeline_names,
        "question_keys": question_keys,
        "raw_records": raw_records,
        "judge_records": judge_records,
        "overview_df": pd.DataFrame(overview_rows),
        "question_df": pd.DataFrame(question_rows),
    }


def build_performance_views(
    question_df: pd.DataFrame,
    selected_pipelines: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if question_df.empty or not selected_pipelines:
        empty = pd.DataFrame()
        return empty, empty

    analytics_df = question_df[["case_citation", "question"]].copy()
    score_cols = [f"{pipeline}__overall" for pipeline in selected_pipelines]
    verdict_cols = [f"{pipeline}__verdict" for pipeline in selected_pipelines]

    analytics_df["models_compared"] = question_df[score_cols].notna().sum(axis=1)
    analytics_df["avg_score"] = question_df[score_cols].mean(axis=1, skipna=True).round(2)
    analytics_df["best_score"] = question_df[score_cols].max(axis=1, skipna=True)
    analytics_df["worst_score"] = question_df[score_cols].min(axis=1, skipna=True)
    analytics_df["score_gap"] = (
        analytics_df["best_score"] - analytics_df["worst_score"]
    ).round(2)
    analytics_df["all_models_judged"] = analytics_df["models_compared"] == len(selected_pipelines)

    best_pipeline_labels: List[Optional[str]] = []
    worst_pipeline_labels: List[Optional[str]] = []
    verdict_summaries: List[str] = []

    for _, row in question_df.iterrows():
        available_scores: List[Tuple[str, float]] = []
        available_verdicts: List[str] = []

        for pipeline in selected_pipelines:
            score = row.get(f"{pipeline}__overall")
            if isinstance(score, (int, float)):
                available_scores.append((pipeline_label(pipeline), float(score)))

            verdict = row.get(f"{pipeline}__verdict")
            if verdict:
                available_verdicts.append(f"{pipeline_label(pipeline)}: {verdict}")

        if available_scores:
            best_pipeline_labels.append(max(available_scores, key=lambda item: item[1])[0])
            worst_pipeline_labels.append(min(available_scores, key=lambda item: item[1])[0])
        else:
            best_pipeline_labels.append(None)
            worst_pipeline_labels.append(None)

        verdict_summaries.append(" | ".join(available_verdicts))

    analytics_df["best_model"] = best_pipeline_labels
    analytics_df["worst_model"] = worst_pipeline_labels
    analytics_df["verdict_summary"] = verdict_summaries

    per_pipeline_rows: List[Dict[str, Any]] = []
    for pipeline in selected_pipelines:
        score_col = f"{pipeline}__overall"
        verdict_col = f"{pipeline}__verdict"
        pipeline_rows = question_df[["case_citation", "question", score_col, verdict_col]].copy()
        pipeline_rows = pipeline_rows.rename(
            columns={
                score_col: "overall_score",
                verdict_col: "verdict",
            }
        )
        pipeline_rows["algorithm"] = pipeline_label(pipeline)
        pipeline_rows["pipeline"] = pipeline
        per_pipeline_rows.extend(pipeline_rows.to_dict("records"))

    return analytics_df, pd.DataFrame(per_pipeline_rows)


def render_context_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        st.caption("No retrieved context available for this pipeline/question.")
        return

    for idx, chunk in enumerate(chunks, start=1):
        title_bits = [f"Chunk {chunk.get('chunk_id', idx)}"]
        if chunk.get("case_citation"):
            title_bits.append(str(chunk["case_citation"]))
        if chunk.get("para_num"):
            title_bits.append(f"para {chunk['para_num']}")
        if chunk.get("score") is not None:
            try:
                title_bits.append(f"score {float(chunk['score']):.3f}")
            except (TypeError, ValueError):
                pass

        with st.expander(" | ".join(title_bits)):
            st.write(chunk.get("text", ""))


def render_judge_block(judge: Dict[str, Any]) -> None:
    if not judge:
        st.caption("No LLM judge result available yet.")
        return

    metric_columns = st.columns(5)
    metric_columns[0].metric("Overall", judge.get("overall_score"))
    metric_columns[1].metric("Correctness", judge.get("correctness"))
    metric_columns[2].metric("Completeness", judge.get("completeness"))
    metric_columns[3].metric("Groundedness", judge.get("groundedness"))
    metric_columns[4].metric("Hallucination", judge.get("hallucination"))
    st.write(f"**Verdict:** {judge.get('verdict', 'N/A')}")
    st.write(f"**Rationale:** {judge.get('rationale', '')}")

    detail_columns = st.columns(3)
    with detail_columns[0]:
        st.write("**Matched facts**")
        facts = judge.get("matched_facts") or []
        if facts:
            for fact in facts:
                st.write(f"- {fact}")
        else:
            st.caption("None")
    with detail_columns[1]:
        st.write("**Missing facts**")
        facts = judge.get("missing_facts") or []
        if facts:
            for fact in facts:
                st.write(f"- {fact}")
        else:
            st.caption("None")
    with detail_columns[2]:
        st.write("**Unsupported claims**")
        facts = judge.get("unsupported_claims") or []
        if facts:
            for fact in facts:
                st.write(f"- {fact}")
        else:
            st.caption("None")


def render_pipeline_card(
    pipeline: str,
    raw_record: Optional[Dict[str, Any]],
    judge_record: Optional[Dict[str, Any]],
) -> None:
    st.subheader(pipeline_label(pipeline))

    if not raw_record:
        st.warning("No raw result found for this question in this pipeline.")
        return

    total_latency = raw_record.get("latency", {}).get("total")
    context_count = len(raw_record.get("context_chunks", []))
    top_columns = st.columns(3)
    top_columns[0].metric("Context chunks", context_count)
    top_columns[1].metric("Total latency (s)", f"{total_latency:.2f}" if isinstance(total_latency, (int, float)) else "N/A")
    top_columns[2].metric("Judge available", "Yes" if judge_record else "No")

    st.write("**Generated answer**")
    st.write(raw_record.get("generated_answer", ""))

    with st.expander("Reference answer", expanded=False):
        st.write(raw_record.get("reference_answer", ""))

    with st.expander("Latency breakdown", expanded=False):
        st.json(raw_record.get("latency", {}))

    with st.expander("Retrieved context", expanded=False):
        render_context_chunks(raw_record.get("context_chunks", []))

    with st.expander("LLM judge result", expanded=True):
        render_judge_block(judge_record or {})


def render_overview_visualizations(overview_df: pd.DataFrame) -> None:
    if overview_df.empty:
        return

    st.write("**Visual Summary**")
    chart_cols = st.columns(2)

    metric_config = {
        "avg_overall_score": {"label": "Overall", "max_score": 10},
        "avg_correctness": {"label": "Correctness", "max_score": 5},
    }
    metric_rows: List[Dict[str, Any]] = []
    for _, row in overview_df.iterrows():
        for column_name, config in metric_config.items():
            raw_score = row.get(column_name)
            if not isinstance(raw_score, (int, float)):
                continue
            metric_rows.append(
                {
                    "label": row["label"],
                    "metric": config["label"],
                    "raw_score": float(raw_score),
                    "max_score": config["max_score"],
                    "normalized_pct": round((float(raw_score) / config["max_score"]) * 100, 1),
                }
            )
    metric_df = pd.DataFrame(metric_rows)

    with chart_cols[0]:
        if not metric_df.empty:
            score_chart = px.bar(
                metric_df,
                x="label",
                y="normalized_pct",
                color="metric",
                barmode="group",
                title="Normalized Average Judge Scores by Algorithm",
                custom_data=["raw_score", "max_score"],
            )
            score_chart.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Metric=%{fullData.name}<br>"
                    "Normalized=%{y:.1f}%<br>"
                    "Raw=%{customdata[0]:.2f}/%{customdata[1]}<extra></extra>"
                )
            )
            score_chart.update_layout(
                xaxis_title="Algorithm",
                yaxis_title="Normalized score (%)",
                legend_title="Metric",
            )
            st.plotly_chart(score_chart, use_container_width=True)
            st.caption(
                "This chart uses normalized percentages so `overall_score` (0-10) is comparable with `correctness` (0-5)."
            )

    with chart_cols[1]:
        latency_df = overview_df.dropna(subset=["avg_latency_total_s"]).copy()
        if not latency_df.empty:
            latency_chart = px.scatter(
                latency_df,
                x="avg_latency_total_s",
                y="avg_overall_score",
                size="questions",
                color="label",
                hover_data=["coverage_pct", "judged"],
                title="Accuracy vs Latency",
            )
            latency_chart.update_layout(
                xaxis_title="Average total latency (s)",
                yaxis_title="Average overall score",
                showlegend=False,
            )
            st.plotly_chart(latency_chart, use_container_width=True)


def render_performance_visualizations(
    analytics_df: pd.DataFrame,
    per_pipeline_df: pd.DataFrame,
) -> None:
    if analytics_df.empty or per_pipeline_df.empty:
        return

    st.write("**Visual Insights**")
    chart_cols = st.columns(2)

    with chart_cols[0]:
        score_distribution_df = per_pipeline_df.dropna(subset=["overall_score"]).copy()
        if not score_distribution_df.empty:
            distribution_chart = px.box(
                score_distribution_df,
                x="algorithm",
                y="overall_score",
                color="algorithm",
                points="outliers",
                title="Score Distribution by Algorithm",
            )
            distribution_chart.update_layout(
                xaxis_title="Algorithm",
                yaxis_title="Overall score",
                showlegend=False,
            )
            st.plotly_chart(distribution_chart, use_container_width=True)

    with chart_cols[1]:
        disagreement_plot_df = analytics_df.dropna(subset=["score_gap"]).copy()
        disagreement_plot_df = disagreement_plot_df.sort_values(
            ["score_gap", "avg_score"], ascending=[False, False]
        ).head(10)
        if not disagreement_plot_df.empty:
            disagreement_plot_df["question_label"] = disagreement_plot_df.apply(
                lambda row: f"{row['case_citation']} | {str(row['question'])[:65]}",
                axis=1,
            )
            disagreement_chart = px.bar(
                disagreement_plot_df,
                x="score_gap",
                y="question_label",
                color="avg_score",
                orientation="h",
                title="Top Question Disagreements",
                color_continuous_scale="Blues",
            )
            disagreement_chart.update_layout(
                xaxis_title="Score gap between best and worst model",
                yaxis_title="Question",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(disagreement_chart, use_container_width=True)


def render_selected_question_visualizations(comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty:
        return

    st.write("**Visual Comparison**")
    chart_cols = st.columns(2)

    with chart_cols[0]:
        score_df = comparison_df.dropna(subset=["judge_overall"]).copy()
        if not score_df.empty:
            question_score_chart = px.bar(
                score_df,
                x="algorithm",
                y="judge_overall",
                color="verdict",
                title="Judge Score for Selected Question",
            )
            question_score_chart.update_layout(
                xaxis_title="Algorithm",
                yaxis_title="Overall score",
            )
            st.plotly_chart(question_score_chart, use_container_width=True)

    with chart_cols[1]:
        latency_df = comparison_df.dropna(subset=["latency_total_s"]).copy()
        if not latency_df.empty:
            latency_chart = go.Figure(
                go.Bar(
                    x=latency_df["algorithm"],
                    y=latency_df["latency_total_s"],
                    marker_color="#4C78A8",
                    text=latency_df["latency_total_s"].round(2),
                    textposition="auto",
                )
            )
            latency_chart.update_layout(
                title="Latency for Selected Question",
                xaxis_title="Algorithm",
                yaxis_title="Latency (s)",
            )
            st.plotly_chart(latency_chart, use_container_width=True)


def render_question_performance_section(
    question_df: pd.DataFrame,
    selected_pipelines: List[str],
    strong_threshold: float,
    weak_threshold: float,
) -> None:
    st.markdown("### Question Performance Insights")

    if not selected_pipelines:
        st.info("Select at least one algorithm to view question-level performance insights.")
        return

    analytics_df, per_pipeline_df = build_performance_views(question_df, selected_pipelines)
    if analytics_df.empty or per_pipeline_df.empty:
        st.info("No judged question data is available for the selected algorithms.")
        return

    full_coverage_df = analytics_df[analytics_df["all_models_judged"]].copy()
    excellent_df = full_coverage_df[full_coverage_df["worst_score"] >= strong_threshold].copy()
    weak_df = full_coverage_df[full_coverage_df["best_score"] <= weak_threshold].copy()
    disagreement_df = full_coverage_df.sort_values(
        ["score_gap", "avg_score"], ascending=[False, False]
    ).copy()

    top_metrics = st.columns(4)
    top_metrics[0].metric("Questions compared", len(analytics_df))
    top_metrics[1].metric("All models judged", len(full_coverage_df))
    top_metrics[2].metric(f"All models strong (>={strong_threshold:.1f})", len(excellent_df))
    top_metrics[3].metric(f"All models weak (<={weak_threshold:.1f})", len(weak_df))

    render_performance_visualizations(analytics_df, per_pipeline_df)

    insight_tab_1, insight_tab_2 = st.tabs(
        ["Across All Selected Models", "Per-Model Breakdown"]
    )

    with insight_tab_1:
        st.caption(
            "Use this to spot questions where every selected model did well, struggled, or strongly disagreed."
        )
        st.caption(
            f"Current thresholds: strong if every selected model scores >= {strong_threshold:.1f}, weak if every selected model scores <= {weak_threshold:.1f}."
        )

        st.write("**Questions all selected models handled well**")
        if excellent_df.empty:
            st.caption("No questions met the current strong-performance rule across all selected models.")
        else:
            st.dataframe(
                excellent_df[
                    ["case_citation", "question", "avg_score", "best_score", "worst_score", "score_gap"]
                ].sort_values(["avg_score", "worst_score"], ascending=[False, False]),
                use_container_width=True,
                hide_index=True,
            )

        st.write("**Questions all selected models struggled with**")
        if weak_df.empty:
            st.caption("No questions met the current weak-performance rule across all selected models.")
        else:
            st.dataframe(
                weak_df[
                    ["case_citation", "question", "avg_score", "best_score", "worst_score", "score_gap"]
                ].sort_values(["avg_score", "best_score"], ascending=[True, True]),
                use_container_width=True,
                hide_index=True,
            )

        st.write("**Questions with the biggest disagreement between models**")
        st.dataframe(
            disagreement_df[
                [
                    "case_citation",
                    "question",
                    "avg_score",
                    "best_score",
                    "worst_score",
                    "score_gap",
                    "best_model",
                    "worst_model",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    with insight_tab_2:
        selected_pipeline_for_analysis = st.selectbox(
            "Pick an algorithm to inspect",
            options=selected_pipelines,
            format_func=pipeline_label,
            key="performance_pipeline_select",
        )
        selected_pipeline_rows = per_pipeline_df[
            per_pipeline_df["pipeline"] == selected_pipeline_for_analysis
        ].copy()
        selected_pipeline_rows = selected_pipeline_rows.dropna(subset=["overall_score"])

        if selected_pipeline_rows.empty:
            st.info("No judged questions are available for this algorithm.")
            return

        metric_cols = st.columns(3)
        metric_cols[0].metric("Judged questions", len(selected_pipeline_rows))
        metric_cols[1].metric(
            "Average score",
            f"{selected_pipeline_rows['overall_score'].mean():.2f}",
        )
        metric_cols[2].metric(
            "Best score",
            f"{selected_pipeline_rows['overall_score'].max():.2f}",
        )

        st.write("**Best questions for this algorithm**")
        st.dataframe(
            selected_pipeline_rows[
                ["case_citation", "question", "overall_score", "verdict"]
            ].sort_values(["overall_score", "case_citation", "question"], ascending=[False, True, True]).head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.write("**Weakest questions for this algorithm**")
        st.dataframe(
            selected_pipeline_rows[
                ["case_citation", "question", "overall_score", "verdict"]
            ].sort_values(["overall_score", "case_citation", "question"], ascending=[True, True, True]).head(10),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Legal RAG Comparator",
        page_icon="scales",
        layout="wide",
    )

    data = build_app_dataset()
    pipelines: List[str] = data["pipelines"]
    raw_records = data["raw_records"]
    judge_records = data["judge_records"]
    overview_df: pd.DataFrame = data["overview_df"]
    question_df: pd.DataFrame = data["question_df"]
    question_keys: List[Tuple[str, str]] = data["question_keys"]

    st.title("Legal RAG Output Comparator")
    st.caption(
        "Compare answers, retrieved context, latency, and LLM-judge scores across all algorithms for the same question."
    )

    with st.sidebar:
        st.header("Controls")
        case_options = sorted({case for case, _ in question_keys})
        selected_case = st.selectbox("Case citation", options=["All"] + case_options)
        search_text = st.text_input("Search question text")
        selected_pipelines = st.multiselect(
            "Algorithms",
            options=pipelines,
            default=pipelines,
            format_func=pipeline_label,
        )
        sort_mode = st.selectbox(
            "Question sort",
            options=[
                "Case citation + question",
                "Best judged score",
                "Worst judged score",
            ],
        )
        strong_threshold = st.slider(
            "Strong score threshold",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="A question counts as strong only if every selected algorithm scores at or above this value.",
        )
        weak_threshold = st.slider(
            "Weak score threshold",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="A question counts as weak only if every selected algorithm scores at or below this value.",
        )

    filtered_df = question_df.copy()
    if selected_case != "All":
        filtered_df = filtered_df[filtered_df["case_citation"] == selected_case]
    if search_text.strip():
        search_query = search_text.strip()
        filtered_df = filtered_df[
            filtered_df["question"].str.contains(search_query, case=False, na=False, regex=False)
        ]

    if selected_pipelines:
        score_cols = [f"{pipeline}__overall" for pipeline in selected_pipelines]
        filtered_df["best_score"] = filtered_df[score_cols].max(axis=1, skipna=True)
        filtered_df["worst_score"] = filtered_df[score_cols].min(axis=1, skipna=True)
    else:
        filtered_df["best_score"] = None
        filtered_df["worst_score"] = None

    if sort_mode == "Best judged score":
        filtered_df = filtered_df.sort_values(["best_score", "case_citation", "question"], ascending=[False, True, True])
    elif sort_mode == "Worst judged score":
        filtered_df = filtered_df.sort_values(["worst_score", "case_citation", "question"], ascending=[True, True, True])
    else:
        filtered_df = filtered_df.sort_values(["case_citation", "question"])

    st.markdown("### Pipeline Overview")
    if not overview_df.empty:
        display_df = overview_df.copy()
        display_df["label"] = display_df["pipeline"].map(pipeline_label)
        st.dataframe(
            display_df[
                [
                    "label",
                    "questions",
                    "judged",
                    "coverage_pct",
                    "avg_overall_score",
                    "avg_correctness",
                ]
            ].rename(columns={"label": "algorithm"}),
            use_container_width=True,
            hide_index=True,
        )
        render_overview_visualizations(display_df)
    else:
        st.info("No pipeline overview data found.")

    render_question_performance_section(
        filtered_df,
        selected_pipelines,
        strong_threshold,
        weak_threshold,
    )

    st.markdown("### Question Browser")
    st.caption(f"{len(filtered_df)} questions matched the current filters.")

    if filtered_df.empty:
        st.warning("No questions matched your current filters.")
        return

    question_options = filtered_df.apply(
        lambda row: f"{row['case_citation']} | {row['question']}",
        axis=1,
    ).tolist()
    selected_question_label = st.selectbox("Pick a question", options=question_options)
    selected_row = filtered_df[
        filtered_df.apply(
            lambda row: f"{row['case_citation']} | {row['question']}" == selected_question_label,
            axis=1,
        )
    ].iloc[0]

    case_citation = selected_row["case_citation"]
    question = selected_row["question"]

    st.markdown("### Selected Question")
    st.write(f"**Case citation:** {case_citation}")
    st.write(f"**Question:** {question}")

    comparison_rows: List[Dict[str, Any]] = []
    for pipeline in selected_pipelines:
        raw_record = raw_records.get(pipeline, {}).get((case_citation, question))
        judge_record = judge_records.get(pipeline, {}).get((case_citation, question), {})
        comparison_rows.append(
            {
                "algorithm": pipeline_label(pipeline),
                "judge_overall": judge_record.get("overall_score"),
                "verdict": judge_record.get("verdict"),
                "latency_total_s": (raw_record or {}).get("latency", {}).get("total"),
                "context_chunks": len((raw_record or {}).get("context_chunks", [])),
            }
        )

    st.markdown("### Snapshot")
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    render_selected_question_visualizations(comparison_df)

    st.markdown("### Side-by-Side Comparison")
    if not selected_pipelines:
        st.info("Select at least one algorithm in the sidebar.")
        return

    tabs = st.tabs([pipeline_label(pipeline) for pipeline in selected_pipelines])
    for tab, pipeline in zip(tabs, selected_pipelines):
        with tab:
            raw_record = raw_records.get(pipeline, {}).get((case_citation, question))
            judge_record = judge_records.get(pipeline, {}).get((case_citation, question))
            render_pipeline_card(pipeline, raw_record, judge_record)


if __name__ == "__main__":
    main()
