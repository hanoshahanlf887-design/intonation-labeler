from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intonation_labeler.agent import AgentState, EvaluationAgent
from intonation_labeler.agent.state import AgentStatus
from intonation_labeler.incremental import ErrorClassification, ErrorKind, persist_failure
from intonation_labeler.review import apply_human_review
from intonation_labeler.incremental import load_success_result, persist_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thin Evaluation Agent in dry-run/demo mode.")
    parser.add_argument("--scenario", choices=["A", "B", "C"], default="A", help="Demo scenario to run.")
    parser.add_argument("--work-dir", help="Optional directory for synthetic demo files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="intonation-agent-demo-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.scenario == "A":
            state, trace = scenario_a(work_dir)
        elif args.scenario == "B":
            state, trace = scenario_b(work_dir)
        else:
            state, trace = scenario_c(work_dir)
        print(json.dumps({"final_state": state.to_dict(), "trace": trace}, ensure_ascii=False, indent=2))
        return 0
    finally:
        if not args.work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def scenario_a(work_dir: Path):
    manifest = _write_demo_dataset(work_dir / "input", review=False, sample_ids=["demo_pass"])
    state = AgentState(
        task_id="scenario-a",
        input_location=str(work_dir / "input"),
        output_dir=str(work_dir / "out_a"),
        provider="dry-run",
        model="demo-model",
        manifest_path=str(manifest),
    )
    agent = EvaluationAgent()
    final_state = agent.run(state)
    return final_state, agent.trace_dicts()


def scenario_b(work_dir: Path):
    manifest = _write_demo_dataset(work_dir / "input", review=True, sample_ids=["demo_review"])
    state = AgentState(
        task_id="scenario-b",
        input_location=str(work_dir / "input"),
        output_dir=str(work_dir / "out_b"),
        provider="dry-run",
        model="demo-model",
        manifest_path=str(manifest),
    )
    agent = EvaluationAgent()
    waiting_state = agent.run(state)

    result = load_success_result(waiting_state.output_dir, "demo_review")
    if result is not None:
        apply_human_review(result, accept_model=True, human_review_note="Synthetic demo review completed.")
        persist_success(waiting_state.output_dir, result, attempts=1, model_id=waiting_state.model)
        path = Path(waiting_state.output_dir) / "individual" / "demo_review.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result"] = result.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    resume_state = waiting_state.copy(status=AgentStatus.WAITING_FOR_REVIEW.value, pending_human_review=False, review_count=0)
    final_state = agent.run(resume_state)
    return final_state, agent.trace_dicts()


def scenario_c(work_dir: Path):
    manifest = _write_demo_dataset(work_dir / "input", review=False, sample_ids=["demo_done", "demo_retry"])
    output_dir = work_dir / "out_c"
    output_dir.mkdir(parents=True, exist_ok=True)
    persist_failure(
        output_dir,
        "demo_retry",
        ErrorClassification(ErrorKind.RETRYABLE, status_code=429, message="Synthetic retryable timeout."),
        attempts=1,
    )
    state = AgentState(
        task_id="scenario-c",
        input_location=str(work_dir / "input"),
        output_dir=str(output_dir),
        provider="dry-run",
        model="demo-model",
        status=AgentStatus.PARTIAL,
        total_samples=2,
        completed_count=1,
        failed_count=1,
        pending_count=1,
        last_error={"kind": "RETRYABLE", "message": "Synthetic retryable timeout."},
        manifest_path=str(manifest),
    )
    agent = EvaluationAgent()
    final_state = agent.run(state)
    return final_state, agent.trace_dicts()


def _write_demo_dataset(input_dir: Path, review: bool, sample_ids: list[str]) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = input_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "wav_path", "feature_json_path", "anchor_id"])
        writer.writeheader()
        for sample_id in sample_ids:
            wav_path = input_dir / f"{sample_id}.wav"
            json_path = input_dir / f"{sample_id}.json"
            wav_path.write_bytes(b"RIFF")
            _write_feature_json(json_path, review=review and sample_id == sample_ids[0])
            writer.writerow({"sample_id": sample_id, "wav_path": str(wav_path), "feature_json_path": str(json_path), "anchor_id": "tag_1"})
    return manifest


def _write_feature_json(path: Path, review: bool) -> None:
    feature = {} if review else {"t_maxf0": "0.78", "z_f0": "0.64", "rangeF0": "92", "slope": "18.40", "t_min": "0.12"}
    path.write_text(
        json.dumps(
            {
                "transcript": "Could we meet after lunch?",
                "anchors": {"tag_1": {"word": "lunch?", "start": 1.0, "end": 1.4, "feature": feature}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
