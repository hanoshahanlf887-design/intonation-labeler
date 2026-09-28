from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..demo import build_demo_model_response
from ..export import write_evaluation_results
from ..incremental import (
    ErrorKind as IncrementalErrorKind,
    IncrementalRunSummary,
    PilotTask,
    load_manifest_tasks,
    load_success_result,
    run_incremental_batch,
)
from ..io import FeatureFormatError, load_feature_json, pair_directory
from ..report import build_summary
from ..schema import EvaluationResult, ValidationStatus
from ..validation import ValidationConfig, evaluate_model_output, parse_model_output
from .state import AgentError, AgentState, AgentStatus


@dataclass
class ToolResult:
    action: str
    observation: dict[str, Any]
    state: AgentState
    summary: str = ""


@dataclass
class DatasetInspection:
    valid: bool
    total_samples: int
    errors: list[str] = field(default_factory=list)


class EvaluationAgentTools:
    """Thin wrappers around existing deterministic workflow modules."""

    def inspect_dataset(self, state: AgentState) -> ToolResult:
        errors: list[str] = []
        total = 0
        if state.manifest_path:
            try:
                tasks = load_manifest_tasks(state.manifest_path)
                total = len(tasks)
                for task in tasks:
                    if not task.wav_path.exists():
                        errors.append(f"missing wav for {task.sample_id}")
                    if not task.feature_json_path.exists():
                        errors.append(f"missing feature json for {task.sample_id}")
                    if task.feature_json_path.exists():
                        try:
                            load_feature_json(task.feature_json_path, task.sample_id)
                        except FeatureFormatError as exc:
                            errors.append(f"{task.sample_id}: {exc}")
            except Exception as exc:
                errors.append(str(exc))
        else:
            pairs, missing_wav, missing_json = pair_directory(state.input_location)
            total = len(pairs)
            errors.extend(f"json without wav: {sample_id}" for sample_id in missing_wav)
            errors.extend(f"wav without json: {sample_id}" for sample_id in missing_json)
            for pair in pairs:
                try:
                    load_feature_json(pair.json_path, pair.sample_id)
                except FeatureFormatError as exc:
                    errors.append(f"{pair.sample_id}: {exc}")

        valid = total > 0 and not errors
        new_state = state.copy(
            status=AgentStatus.RUNNING.value if valid else AgentStatus.FAILED.value,
            total_samples=total,
            pending_count=total,
            failed_count=0 if valid else len(errors),
            last_error=AgentError(None if valid else "NON_RETRYABLE", None if valid else "; ".join(errors)).to_dict(),
            next_allowed_actions=["run_evaluation_batch"] if valid else [],
        )
        observation = {"valid": valid, "total_samples": total, "error_count": len(errors), "errors": errors[:5]}
        return ToolResult("inspect_dataset", observation, new_state, f"valid={valid}, total={total}, errors={len(errors)}")

    def run_evaluation_batch(self, state: AgentState) -> ToolResult:
        if state.manifest_path:
            result = self._run_manifest_batch(state)
        else:
            result = self._run_directory_batch(state)
        return result

    def resume_evaluation_batch(self, state: AgentState) -> ToolResult:
        if not state.manifest_path:
            return ToolResult(
                "resume_evaluation_batch",
                {"resumable": False, "reason": "resume requires manifest/incremental output"},
                state.copy(status=AgentStatus.BLOCKED.value, next_allowed_actions=[]),
                "resume unavailable without manifest",
            )
        return self._run_manifest_batch(state, action="resume_evaluation_batch")

    def get_batch_status(self, state: AgentState) -> ToolResult:
        output_dir = Path(state.output_dir)
        results = _load_success_results(output_dir)
        failures = list((output_dir / "failures").glob("*.json")) if (output_dir / "failures").exists() else []
        completed = len(results)
        failed = len(failures)
        total = state.total_samples or completed + failed
        pending = max(total - completed - failed, 0)
        review_count = sum(1 for item in results if item.needs_human_review)
        status = _status_from_counts(total, completed, failed, pending, review_count, state.last_error.kind)
        next_actions = _next_actions_for_counts(status, pending, review_count)
        new_state = state.copy(
            status=status.value,
            total_samples=total,
            completed_count=completed,
            failed_count=failed,
            pending_count=pending,
            review_count=review_count,
            pending_human_review=review_count > 0,
            next_allowed_actions=next_actions,
        )
        observation = {"completed": completed, "failed": failed, "pending": pending, "review_count": review_count}
        return ToolResult("get_batch_status", observation, new_state, f"completed={completed}, failed={failed}, pending={pending}")

    def get_validation_summary(self, state: AgentState) -> ToolResult:
        results = _load_success_results(Path(state.output_dir))
        summary = build_summary(results)
        review_count = sum(1 for item in results if item.needs_human_review)
        new_state = state.copy(
            review_count=review_count,
            pending_human_review=review_count > 0,
            next_allowed_actions=["get_review_queue"] if review_count else ["generate_report"],
        )
        observation = {
            "status_counts": summary["status_counts"],
            "validation_reason_distribution": summary["validation_reason_distribution"],
            "review_count": review_count,
        }
        return ToolResult("get_validation_summary", observation, new_state, f"review_count={review_count}")

    def get_review_queue(self, state: AgentState) -> ToolResult:
        results = _load_success_results(Path(state.output_dir))
        queue = [
            {
                "sample_id": item.sample_id,
                "validation_status": item.validation_status.value,
                "validation_reasons": item.validation_reasons,
            }
            for item in results
            if item.needs_human_review
        ]
        pending = len(queue) > 0
        new_state = state.copy(
            status=AgentStatus.WAITING_FOR_REVIEW.value if pending else state.status.value,
            review_count=len(queue),
            pending_human_review=pending,
            next_allowed_actions=[] if pending else ["generate_report"],
        )
        return ToolResult("get_review_queue", {"review_queue_count": len(queue), "cases": queue[:10]}, new_state, f"review_queue={len(queue)}")

    def get_case_detail(self, state: AgentState, sample_id: str) -> ToolResult:
        result = load_success_result(state.output_dir, sample_id)
        if result is None:
            observation = {"found": False, "sample_id": sample_id}
            return ToolResult("get_case_detail", observation, state, "case not found")
        observation = {
            "found": True,
            "sample_id": result.sample_id,
            "validation_status": result.validation_status.value,
            "validation_reasons": result.validation_reasons,
            "needs_human_review": result.needs_human_review,
            "model_label": result.model_label,
            "final_label_present": result.final_label is not None,
        }
        return ToolResult("get_case_detail", observation, state, f"case={sample_id}")

    def generate_report(self, state: AgentState) -> ToolResult:
        results = _load_success_results(Path(state.output_dir))
        if not results:
            new_state = state.copy(status=AgentStatus.FAILED.value, last_error=AgentError("NON_RETRYABLE", "No successful results available for reporting.").to_dict())
            return ToolResult("generate_report", {"report_ready": False, "reason": "no results"}, new_state, "no results")
        paths = write_evaluation_results(results, state.output_dir)
        new_state = state.copy(status=AgentStatus.COMPLETED.value, report_ready=True, next_allowed_actions=[])
        observation = {"report_ready": True, "paths": {key: str(path) for key, path in paths.items()}}
        return ToolResult("generate_report", observation, new_state, "report_ready=True")

    def _run_manifest_batch(self, state: AgentState, action: str = "run_evaluation_batch") -> ToolResult:
        tasks = load_manifest_tasks(state.manifest_path or "")
        annotator = _make_dry_run_annotator(tasks)
        summary = run_incremental_batch(
            tasks=tasks,
            output_dir=state.output_dir,
            model_id=state.model,
            annotator=annotator,
            validation_config=ValidationConfig(),
            retry_base_delay=0,
            sleep_fn=lambda seconds: None,
        )
        return self._state_from_incremental_summary(state, action, summary)

    def _run_directory_batch(self, state: AgentState) -> ToolResult:
        pairs, _, _ = pair_directory(state.input_location)
        results: list[EvaluationResult] = []
        for pair in pairs:
            feature_data = load_feature_json(pair.json_path, pair.sample_id)
            raw_response = build_demo_model_response(feature_data)
            model_output = parse_model_output(raw_response)
            results.append(evaluate_model_output(pair.sample_id, feature_data, model_output, ValidationConfig()))
        paths = write_evaluation_results(results, state.output_dir)
        review_count = sum(1 for item in results if item.needs_human_review)
        new_state = state.copy(
            status=AgentStatus.WAITING_FOR_REVIEW.value if review_count else AgentStatus.RUNNING.value,
            total_samples=len(results),
            completed_count=len(results),
            failed_count=0,
            pending_count=0,
            review_count=review_count,
            pending_human_review=review_count > 0,
            next_allowed_actions=["get_review_queue"] if review_count else ["generate_report"],
        )
        observation = {"completed": len(results), "review_count": review_count, "paths": {key: str(path) for key, path in paths.items()}}
        return ToolResult("run_evaluation_batch", observation, new_state, f"completed={len(results)}, review_count={review_count}")

    def _state_from_incremental_summary(self, state: AgentState, action: str, summary: IncrementalRunSummary) -> ToolResult:
        results = _load_success_results(Path(state.output_dir))
        failures = _load_failure_payloads(Path(state.output_dir))
        completed = len(results)
        failed = len(failures)
        total = summary.total_samples
        pending = max(total - completed - failed, 0)
        review_count = sum(1 for item in results if item.needs_human_review)
        last_error = _last_error_from_failures(failures, summary)
        status = _status_from_counts(total, completed, failed, pending, review_count, last_error.kind)
        if summary.paused and last_error.kind == "FATAL":
            status = AgentStatus.BLOCKED
        next_actions = _next_actions_for_counts(status, pending, review_count)
        new_state = state.copy(
            status=status.value,
            total_samples=total,
            completed_count=completed,
            failed_count=failed,
            pending_count=pending,
            review_count=review_count,
            pending_human_review=review_count > 0,
            last_error=last_error.to_dict(),
            next_allowed_actions=next_actions,
        )
        observation = {
            "total": total,
            "skipped_success": summary.skipped_success,
            "succeeded_this_run": summary.succeeded,
            "failed_this_run": summary.failed,
            "paused": summary.paused,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "review_count": review_count,
            "last_error_kind": last_error.kind,
        }
        return ToolResult(action, observation, new_state, f"completed={completed}/{total}, failed={failed}, pending={pending}, review={review_count}")


def _make_dry_run_annotator(tasks: list[PilotTask]):
    feature_by_wav = {str(task.wav_path): task.feature_json_path for task in tasks}

    def annotator(model_id: str, wav_path: Path, prompt: str) -> str:
        feature_path = feature_by_wav.get(str(wav_path))
        if feature_path:
            feature_data = load_feature_json(feature_path, wav_path.stem)
            return build_demo_model_response(feature_data)
        return _dry_run_annotator(model_id, wav_path, prompt)

    return annotator

def _dry_run_annotator(model_id: str, wav_path: Path, prompt: str) -> str:
    import re

    quoted = [
        item
        for item in re.findall(r'"([^"]*)"', prompt)
        if item and not item.startswith("tag_") and not item.startswith("anchor_")
    ]
    transcript = quoted[0] if quoted else ""
    anchor_ids = re.findall(r'"(tag_[^"]+|anchor_[^"]+)"\s*:', prompt) or ["tag_1"]
    annotations = [
        {
            "anchor_id": anchor_id,
            "label": "RISING",
            "confidence": 0.82,
            "reason": "Dry-run mock annotation.",
        }
        for anchor_id in sorted(set(anchor_ids))
    ]
    return json.dumps({"transcript": transcript, "annotated_text": transcript, "annotations": annotations}, ensure_ascii=False)

def _load_success_results(output_dir: Path) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    individual_dir = output_dir / "individual"
    if individual_dir.exists():
        for path in sorted(individual_dir.glob("*.json")):
            result = load_success_result(output_dir, path.stem)
            if result is not None:
                results.append(result)
    return results


def _load_failure_payloads(output_dir: Path) -> list[dict[str, Any]]:
    failure_dir = output_dir / "failures"
    if not failure_dir.exists():
        return []
    payloads = []
    for path in sorted(failure_dir.glob("*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            payloads.append({"message": f"Unreadable failure file: {path.name}", "error_kind": "NON_RETRYABLE"})
    return payloads


def _last_error_from_failures(failures: list[dict[str, Any]], summary: IncrementalRunSummary) -> AgentError:
    if summary.fatal_error:
        return AgentError("FATAL", summary.fatal_error)
    if not failures:
        return AgentError()
    last = failures[-1]
    return AgentError(last.get("error_kind"), last.get("message"))


def _status_from_counts(total: int, completed: int, failed: int, pending: int, review_count: int, error_kind: str | None) -> AgentStatus:
    if error_kind == IncrementalErrorKind.FATAL.value or error_kind == "FATAL":
        return AgentStatus.BLOCKED
    if completed >= total and total > 0:
        return AgentStatus.WAITING_FOR_REVIEW if review_count else AgentStatus.RUNNING
    if completed > 0 or pending > 0:
        return AgentStatus.PARTIAL
    if failed > 0:
        return AgentStatus.FAILED
    return AgentStatus.RUNNING


def _next_actions_for_counts(status: AgentStatus, pending: int, review_count: int) -> list[str]:
    if status == AgentStatus.PARTIAL and pending > 0:
        return ["resume_evaluation_batch"]
    if status == AgentStatus.WAITING_FOR_REVIEW and review_count > 0:
        return ["get_review_queue"]
    if status == AgentStatus.RUNNING and pending == 0:
        return ["generate_report"]
    return []






