import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are a careful legal QA evaluator.

Your job is to judge a generated answer against:
1. the question,
2. the reference answer,
3. the retrieved context, if any.

Use the rubric below:
- correctness: 0 to 5
  5 = fully correct, 4 = mostly correct with minor omissions, 3 = partially correct,
  2 = mostly incorrect, 1 = clearly incorrect, 0 = nonsense or no answer.
- completeness: 0 to 5
  5 = covers all key facts in the reference answer, 3 = covers some key facts,
  1 = major omissions, 0 = does not answer the question.
- groundedness: 0 to 5, or null if no retrieved context is available
  5 = fully supported by retrieved context, 3 = partly supported, 1 = weakly supported, 0 = unsupported.
- hallucination: 0 to 5
  5 = no material hallucination, 3 = minor unsupported detail, 1 = major unsupported claims, 0 = mostly fabricated.
- overall_score: 0 to 10
  Judge the answer holistically, weighting correctness most heavily.

Also return:
- verdict: one of ["excellent", "good", "fair", "poor", "fail"]
- matched_facts: short list of facts correctly captured
- missing_facts: short list of important facts missing from the answer
- unsupported_claims: short list of claims not supported by the reference answer or context
- rationale: brief explanation in 1 to 3 sentences

Important rules:
- Prefer semantic equivalence, not exact wording.
- Penalize legal inaccuracies and invented facts heavily.
- If the answer says "I do not have enough information" despite the reference containing the answer, score it low.
- If there is no retrieved context, set groundedness to null and do not guess.
- Output valid JSON only. No markdown fences, no extra text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation on notebook output JSON."
    )
    parser.add_argument(
        "--input",
        default=str(BASE_DIR / "evaluation_results_cohere.json"),
        help="Path to the evaluation results JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "llm_judge_results"),
        help="Directory to save per-pipeline detailed judge results.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "phi3:latest"),
        help="Local Ollama judge model. Defaults to env OLLAMA_MODEL or phi3:latest.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples per pipeline.",
    )
    parser.add_argument(
        "--pipelines",
        default=None,
        help="Optional comma-separated pipeline names to evaluate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based start index within each selected pipeline.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based exclusive end index within each selected pipeline.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size per pipeline.",
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        default=None,
        help="Optional 1-based batch number to run when --batch-size is set.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1800,
        help="Maximum retrieved context characters sent to the judge model.",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=2200,
        help="Maximum characters kept from reference and generated answers.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between judge calls.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each Ollama judge request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries for transient Ollama request failures.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file if present.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text: Any, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def get_context_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = record.get("context_chunks")
    if isinstance(chunks, list):
        return chunks

    chunks = record.get("retrieved_context")
    if isinstance(chunks, list):
        return chunks

    return []


def extract_context_text(record: Dict[str, Any], max_chars: int = 1800) -> str:
    chunks = get_context_chunks(record)
    if not chunks:
        return ""

    context_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunk_id = chunk.get("chunk_id", idx)
        context_parts.append(f"[Chunk {chunk_id}] {text}")

    joined = "\n\n".join(context_parts)
    return joined[:max_chars]


def build_user_prompt(
    record: Dict[str, Any],
    max_context_chars: int,
    max_answer_chars: int,
    compact: bool = False,
) -> str:
    context_text = extract_context_text(record, max_chars=max_context_chars)
    payload = {
        "question": record.get("question", ""),
        "reference_answer": truncate_text(
            record.get("reference_answer", ""),
            max_answer_chars,
        ),
        "generated_answer": truncate_text(
            record.get("generated_answer", ""),
            max_answer_chars,
        ),
        "retrieved_context": context_text if context_text else None,
    }
    if compact:
        payload["extra_instruction"] = (
            "Keep arrays short with at most 2 items each and keep rationale to 1 sentence."
        )
    else:
        payload["required_json_schema"] = {
            "correctness": "integer 0-5",
            "completeness": "integer 0-5",
            "groundedness": "integer 0-5 or null",
            "hallucination": "integer 0-5",
            "overall_score": "integer 0-10",
            "verdict": 'one of ["excellent", "good", "fair", "poor", "fail"]',
            "matched_facts": "list of short strings",
            "missing_facts": "list of short strings",
            "unsupported_claims": "list of short strings",
            "rationale": "short string",
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_judge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer", "minimum": 0, "maximum": 5},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 5},
            "groundedness": {
                "anyOf": [
                    {"type": "integer", "minimum": 0, "maximum": 5},
                    {"type": "null"},
                ]
            },
            "hallucination": {"type": "integer", "minimum": 0, "maximum": 5},
            "overall_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "verdict": {
                "type": "string",
                "enum": ["excellent", "good", "fair", "poor", "fail"],
            },
            "matched_facts": {"type": "array", "items": {"type": "string"}},
            "missing_facts": {"type": "array", "items": {"type": "string"}},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "correctness",
            "completeness",
            "groundedness",
            "hallucination",
            "overall_score",
            "verdict",
            "matched_facts",
            "missing_facts",
            "unsupported_claims",
            "rationale",
        ],
    }


def post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 600,
    max_retries: int = 3,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc

        if attempt < max_retries:
            wait_seconds = min(10, attempt * 2)
            print(
                f"Ollama request failed on attempt {attempt}/{max_retries}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    if isinstance(last_error, TimeoutError):
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s at {url}. "
            "Try a larger --request-timeout, a smaller --max-examples, or resume the run."
        ) from last_error

    raise RuntimeError(
        f"Could not reach Ollama at {url}. Make sure the Ollama app or server is running."
    ) from last_error


def call_judge(
    ollama_url: str,
    model: str,
    record: Dict[str, Any],
    request_timeout: int,
    max_retries: int,
    max_context_chars: int,
    max_answer_chars: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    current_context_chars = max_context_chars

    for parse_attempt in range(1, max_retries + 1):
        compact = parse_attempt > 1
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        record,
                        max_context_chars=current_context_chars,
                        max_answer_chars=max_answer_chars,
                        compact=compact,
                    ),
                },
            ],
            "stream": False,
            "format": get_judge_schema(),
            "options": {
                "temperature": 0,
            },
        }
        response = post_json(
            f"{ollama_url.rstrip('/')}/api/chat",
            payload,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        text = response.get("message", {}).get("content", "").strip()
        if not text:
            last_error = ValueError(f"Empty response from Ollama model {model}.")
        else:
            try:
                return json.loads(text)
            except JSONDecodeError as exc:
                last_error = exc

        if parse_attempt < max_retries:
            current_context_chars = max(400, current_context_chars // 2)
            print(
                f"Judge returned malformed JSON on parse attempt {parse_attempt}/{max_retries}. "
                f"Retrying with smaller prompt ({current_context_chars} context chars)..."
            )

    raise RuntimeError(
        f"Failed to parse judge output from model {model} after {max_retries} attempts."
    ) from last_error


def make_record_id(record: Dict[str, Any], index: int) -> str:
    pipeline = record.get("pipeline", "unknown")
    question = record.get("question", "")
    return f"{pipeline}::{index}::{question}"


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def verdict_from_score(overall_score: int) -> str:
    if overall_score >= 9:
        return "excellent"
    if overall_score >= 7:
        return "good"
    if overall_score >= 5:
        return "fair"
    if overall_score >= 3:
        return "poor"
    return "fail"


def sanitize_judgment(judgment: Dict[str, Any], has_context: bool) -> Dict[str, Any]:
    cleaned = dict(judgment)
    cleaned["correctness"] = clamp_int(cleaned.get("correctness", 0), 0, 5, 0)
    cleaned["completeness"] = clamp_int(cleaned.get("completeness", 0), 0, 5, 0)
    cleaned["hallucination"] = clamp_int(cleaned.get("hallucination", 0), 0, 5, 0)
    cleaned["overall_score"] = clamp_int(cleaned.get("overall_score", 0), 0, 10, 0)

    groundedness = cleaned.get("groundedness")
    if groundedness is None or not has_context:
        cleaned["groundedness"] = None
    else:
        cleaned["groundedness"] = clamp_int(groundedness, 0, 5, 0)

    for key in ["matched_facts", "missing_facts", "unsupported_claims"]:
        value = cleaned.get(key, [])
        cleaned[key] = value if isinstance(value, list) else [str(value)]

    cleaned["verdict"] = verdict_from_score(cleaned["overall_score"])
    cleaned["rationale"] = str(cleaned.get("rationale", "")).strip()
    return cleaned


def select_pipelines(
    data: Dict[str, List[Dict[str, Any]]],
    pipelines_arg: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    if not pipelines_arg:
        return data

    wanted = [name.strip() for name in pipelines_arg.split(",") if name.strip()]
    selected = {name: data[name] for name in wanted if name in data}
    missing = [name for name in wanted if name not in data]
    if missing:
        print(f"Skipping unknown pipelines: {', '.join(missing)}")
    return selected


def slice_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    start_index = max(0, args.start_index)
    end_index = args.end_index

    if args.batch_size is not None:
        if args.batch_number is None or args.batch_number < 1:
            raise ValueError("--batch-number must be >= 1 when --batch-size is used.")
        batch_start = (args.batch_number - 1) * args.batch_size
        batch_end = batch_start + args.batch_size
        start_index = max(start_index, batch_start)
        end_index = batch_end if end_index is None else min(end_index, batch_end)

    sliced = records[start_index:end_index]
    if args.max_examples is not None:
        sliced = sliced[: args.max_examples]
    return sliced


def average(values: List[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return round(statistics.mean(filtered), 4)


def build_summary(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for pipeline, records in results.items():
        correctness = [r["judge"]["correctness"] for r in records]
        completeness = [r["judge"]["completeness"] for r in records]
        groundedness = [r["judge"]["groundedness"] for r in records]
        hallucination = [r["judge"]["hallucination"] for r in records]
        overall = [r["judge"]["overall_score"] for r in records]

        verdict_counts: Dict[str, int] = {}
        for record in records:
            verdict = record["judge"]["verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        summary[pipeline] = {
            "num_examples": len(records),
            "avg_correctness_0_5": average(correctness),
            "avg_completeness_0_5": average(completeness),
            "avg_groundedness_0_5": average(groundedness),
            "avg_hallucination_0_5": average(hallucination),
            "avg_overall_score_0_10": average(overall),
            "verdict_counts": verdict_counts,
        }
    return summary


def load_resume_state(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pipeline_output_path(output_dir: Path, pipeline: str) -> Path:
    return output_dir / f"{pipeline}.json"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    data = load_json(input_path)
    data = select_pipelines(data, args.pipelines)

    for pipeline, records in data.items():
        output_path = pipeline_output_path(output_dir, pipeline)
        existing_records = load_resume_state(output_path).get(pipeline, []) if args.resume else []
        completed_ids = {record["record_id"] for record in existing_records}
        judged_results: Dict[str, List[Dict[str, Any]]] = {
            pipeline: list(existing_records)
        }
        original_records = records
        records = slice_records(records, args)
        start_offset = original_records.index(records[0]) if records else 0

        for local_index, record in enumerate(records):
            index = start_offset + local_index
            record = dict(record)
            record.setdefault("pipeline", pipeline)
            record_id = make_record_id(record, index)
            if record_id in completed_ids:
                continue

            has_context = bool(get_context_chunks(record))
            try:
                judgment = call_judge(
                    args.ollama_url,
                    args.model,
                    record,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    max_context_chars=args.max_context_chars,
                    max_answer_chars=args.max_answer_chars,
                )
            except Exception as exc:
                print(
                    f"Skipping {pipeline} [{index + 1}/{len(original_records)}] "
                    f"due to judge error: {exc}"
                )
                continue
            judgment = sanitize_judgment(judgment, has_context=has_context)

            enriched = dict(record)
            enriched["record_id"] = record_id
            enriched["judge_model"] = args.model
            enriched["judge"] = judgment
            judged_results[pipeline].append(enriched)

            save_json(output_path, judged_results)

            print(
                f"Judged {pipeline} [{index + 1}/{len(original_records)}] "
                f"=> overall={judgment['overall_score']}, verdict={judgment['verdict']}"
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        save_json(output_path, judged_results)
        print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
