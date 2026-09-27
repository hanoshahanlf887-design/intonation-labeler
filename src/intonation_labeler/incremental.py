from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .export import write_evaluation_results
from .io import FeatureFormatError, load_feature_json
from .prompts import build_prompt
from .schema import EvaluationResult, ValidationReason, ValidationStatus
from .validation import ModelOutputParseError, ValidationConfig, evaluate_model_output, parse_model_output


class ErrorKind(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    FATAL = "FATAL"


@dataclass(frozen=True)
class PilotTask:
    sample_id: str
    wav_path: Path
    feature_json_path: Path


@dataclass(frozen=True)
class ErrorClassification:
    kind: ErrorKind
    status_code: int | None = None
    message: str = ""


@dataclass(frozen=True)
class IncrementalRunSummary:
    total_samples: int
    skipped_success: int
    succeeded: int
    failed: int
    paused: bool
    fatal_error: str | None = None


Annotator = Callable[[str, Path, str], str]
SleepFn = Callable[[float], None]


def load_manifest_tasks(manifest_path: str | Path) -> list[PilotTask]:
    """Load unique sample tasks from the pilot manifest without reading human labels."""
    tasks: dict[str, PilotTask] = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            if not sample_id or sample_id in tasks:
                continue
            tasks[sample_id] = PilotTask(
                sample_id=sample_id,
                wav_path=Path(row["wav_path"]),
                feature_json_path=Path(row["feature_json_path"]),
            )
    return list(tasks.values())


def run_incremental_batch(
    tasks: list[PilotTask],
    output_dir: str | Path,
    model_id: str,
    annotator: Annotator,
    validation_config: ValidationConfig | None = None,
    max_retries: int = 2,
    retry_base_delay: float = 5.0,
    pause_after_consecutive_retryable: int = 3,
    sleep_fn: SleepFn = time.sleep,
) -> IncrementalRunSummary:
    output_path = Path(output_dir)
    success_dir = output_path / "individual"
    failure_dir = output_path / "failures"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    validation_config = validation_config or ValidationConfig()
    skipped_success = 0
    succeeded = 0
    failed = 0
    paused = False
    fatal_error: str | None = None
    consecutive_retryable_failures = 0

    for task in tasks:
        if load_success_result(output_path, task.sample_id) is not None:
            skipped_success += 1
            continue

        result, classification, attempts = _run_one_task(
            task=task,
            output_dir=output_path,
            model_id=model_id,
            annotator=annotator,
            validation_config=validation_config,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            sleep_fn=sleep_fn,
        )

        if result is not None:
            persist_success(output_path, result, attempts=attempts, model_id=model_id)
            _remove_failure(output_path, task.sample_id)
            succeeded += 1
            consecutive_retryable_failures = 0
            continue

        failed += 1
        if classification and classification.kind == ErrorKind.FATAL:
            paused = True
            fatal_error = classification.message
            break
        if classification and classification.kind == ErrorKind.RETRYABLE:
            consecutive_retryable_failures += 1
            if consecutive_retryable_failures >= pause_after_consecutive_retryable:
                paused = True
                break
        else:
            consecutive_retryable_failures = 0

    _write_current_outputs(output_path)
    summary = IncrementalRunSummary(
        total_samples=len(tasks),
        skipped_success=skipped_success,
        succeeded=succeeded,
        failed=failed,
        paused=paused,
        fatal_error=fatal_error,
    )
    _write_run_state(output_path, summary)
    return summary


def _run_one_task(
    task: PilotTask,
    output_dir: Path,
    model_id: str,
    annotator: Annotator,
    validation_config: ValidationConfig,
    max_retries: int,
    retry_base_delay: float,
    sleep_fn: SleepFn,
) -> tuple[EvaluationResult | None, ErrorClassification | None, int]:
    try:
        feature_data = load_feature_json(task.feature_json_path, task.sample_id)
    except FeatureFormatError as exc:
        classification = ErrorClassification(ErrorKind.NON_RETRYABLE, message=str(exc))
        persist_failure(output_dir, task.sample_id, classification, attempts=1)
        return None, classification, 1

    total_attempts = max_retries + 1
    last_classification: ErrorClassification | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            prompt = build_prompt(feature_data.transcript, feature_data.anchors)
            raw_response = annotator(model_id, task.wav_path, prompt)
            model_output = parse_model_output(raw_response)
            result = evaluate_model_output(task.sample_id, feature_data, model_output, validation_config)
            return result, None, attempt
        except ModelOutputParseError as exc:
            last_classification = ErrorClassification(ErrorKind.NON_RETRYABLE, message=str(exc))
            persist_failure(output_dir, task.sample_id, last_classification, attempts=attempt)
            return None, last_classification, attempt
        except Exception as exc:
            last_classification = classify_inference_error(exc)
            persist_failure(output_dir, task.sample_id, last_classification, attempts=attempt)
            if last_classification.kind != ErrorKind.RETRYABLE:
                return None, last_classification, attempt
            if attempt < total_attempts:
                sleep_fn(retry_base_delay * (2 ** (attempt - 1)))

    return None, last_classification, total_attempts


def classify_inference_error(error: Exception) -> ErrorClassification:
    message = str(error)
    status_code = _extract_status_code(message)
    if status_code in (401, 403):
        return ErrorClassification(ErrorKind.FATAL, status_code=status_code, message=message)
    if status_code == 400:
        return ErrorClassification(ErrorKind.NON_RETRYABLE, status_code=status_code, message=message)
    if status_code == 429 or status_code in (500, 502, 503, 504):
        return ErrorClassification(ErrorKind.RETRYABLE, status_code=status_code, message=message)

    class_name = error.__class__.__name__
    retryable_names = ("ConnectTimeout", "ReadTimeout", "TimeoutException", "ConnectError", "NetworkError")
    if any(name in class_name for name in retryable_names):
        return ErrorClassification(ErrorKind.RETRYABLE, message=message)
    if "timed out" in message.lower() or "temporary" in message.lower():
        return ErrorClassification(ErrorKind.RETRYABLE, message=message)
    return ErrorClassification(ErrorKind.NON_RETRYABLE, status_code=status_code, message=message)


def persist_success(output_dir: str | Path, result: EvaluationResult, attempts: int, model_id: str) -> Path:
    path = _success_path(Path(output_dir), result.sample_id)
    if path.exists():
        return path
    payload = {
        "sample_id": result.sample_id,
        "attempts": attempts,
        "model_id": model_id,
        "completed": True,
        "result": result.to_dict(),
    }
    _write_json_atomic(path, payload)
    return path


def persist_failure(
    output_dir: str | Path,
    sample_id: str,
    classification: ErrorClassification,
    attempts: int,
) -> Path:
    path = _failure_path(Path(output_dir), sample_id)
    payload = {
        "sample_id": sample_id,
        "attempts": attempts,
        "completed": False,
        "error_kind": classification.kind.value,
        "status_code": classification.status_code,
        "message": classification.message,
    }
    _write_json_atomic(path, payload)
    return path


def load_success_result(output_dir: str | Path, sample_id: str) -> EvaluationResult | None:
    path = _success_path(Path(output_dir), sample_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not data.get("completed") or not isinstance(data.get("result"), dict):
        return None
    return _evaluation_result_from_dict(data["result"])


def _write_current_outputs(output_dir: Path) -> None:
    results = _load_all_success_results(output_dir)
    if not results:
        return
    write_evaluation_results(results, output_dir)


def _load_all_success_results(output_dir: Path) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for path in sorted((output_dir / "individual").glob("*.json")):
        item = load_success_result(output_dir, path.stem)
        if item is not None:
            results.append(item)
    return results


def _evaluation_result_from_dict(data: dict[str, Any]) -> EvaluationResult:
    from .schema import AnchorPrediction, ModelOutput

    model_output = None
    model_data = data.get("model_output")
    if isinstance(model_data, dict):
        model_output = ModelOutput(
            transcript=model_data.get("transcript", ""),
            annotated_text=model_data.get("annotated_text", ""),
            raw_response=model_data.get("raw_response", ""),
            annotations=[
                AnchorPrediction(
                    anchor_id=item.get("anchor_id", ""),
                    label=item.get("label", ""),
                    confidence=item.get("confidence"),
                    reason=item.get("reason", ""),
                )
                for item in model_data.get("annotations", [])
                if isinstance(item, dict)
            ],
        )
    return EvaluationResult(
        sample_id=data.get("sample_id", ""),
        transcript=data.get("transcript", ""),
        acoustic_features=data.get("acoustic_features", {}),
        model_output=model_output,
        validation_status=ValidationStatus(data.get("validation_status", ValidationStatus.REVIEW.value)),
        validation_reasons=list(data.get("validation_reasons", [])),
        needs_human_review=bool(data.get("needs_human_review", True)),
        human_review_label=data.get("human_review_label"),
        human_review_note=data.get("human_review_note", ""),
        final_label=data.get("final_label"),
        error=data.get("error"),
    )


def _write_run_state(output_dir: Path, summary: IncrementalRunSummary) -> None:
    _write_json_atomic(output_dir / "run_state.json", summary.__dict__)


def _remove_failure(output_dir: Path, sample_id: str) -> None:
    _failure_path(output_dir, sample_id).unlink(missing_ok=True)


def _success_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "individual" / f"{sample_id}.json"


def _failure_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "failures" / f"{sample_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _extract_status_code(message: str) -> int | None:
    match = re.search(r"\bHTTP\s+(\d{3})\b", message)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(400|401|403|429|500|502|503|504)\b", message)
    if match:
        return int(match.group(1))
    return None
