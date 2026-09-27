from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intonation_labeler.demo import build_demo_model_response
from intonation_labeler.export import write_evaluation_results
from intonation_labeler.gemini_client import annotate_audio, create_client
from intonation_labeler.incremental import load_manifest_tasks, run_incremental_batch
from intonation_labeler.io import FeatureFormatError, load_feature_json, pair_directory
from intonation_labeler.prompts import build_prompt
from intonation_labeler.relay_client import DEFAULT_RELAY_MODEL, annotate_audio_via_relay
from intonation_labeler.schema import EvaluationResult, ValidationReason, ValidationStatus
from intonation_labeler.validation import ModelOutputParseError, ValidationConfig, evaluate_model_output, parse_model_output


DEFAULT_OFFICIAL_MODEL = "gemini-2.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch annotate intonation from paired WAV and JSON files.")
    parser.add_argument("input_path", nargs="?", help="Directory containing <sample_id>.wav and <sample_id>.json files.")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data"), help="Directory containing <sample_id>.wav and <sample_id>.json files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"), help="Directory for result files.")
    parser.add_argument("--provider", choices=["official", "relay"], default="official", help="Model provider to use for live annotation.")
    parser.add_argument("--model", default=None, help="Gemini model id. Defaults to gemini-2.5-flash for official and gemini-3.8-flash for relay.")
    parser.add_argument("--dry-run", action="store_true", help="Use mock annotations without calling Gemini.")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Self-reported model confidence below this value is routed to review.")
    parser.add_argument("--manifest", help="CSV manifest with sample_id, wav_path, and feature_json_path for incremental/resumable inference.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry retryable API/network errors this many times per sample.")
    parser.add_argument("--retry-base-delay", type=float, default=5.0, help="Base seconds for exponential backoff between retryable failures.")
    parser.add_argument("--pause-after-consecutive-retryable", type=int, default=3, help="Pause the batch after this many consecutive retryable sample failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    model_id = args.model or (DEFAULT_RELAY_MODEL if args.provider == "relay" else DEFAULT_OFFICIAL_MODEL)
    if args.manifest:
        if args.dry_run:
            annotator = _dry_run_annotator_from_manifest
        elif args.provider == "relay":
            annotator = annotate_audio_via_relay
        else:
            client = create_client()

            def annotator(model_id: str, wav_path: Path, prompt: str) -> str:
                return annotate_audio(client, model_id, wav_path, prompt)

        tasks = load_manifest_tasks(args.manifest)
        summary = run_incremental_batch(
            tasks=tasks,
            output_dir=args.output_dir,
            model_id=model_id,
            annotator=annotator,
            validation_config=ValidationConfig(confidence_threshold=args.confidence_threshold),
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
            pause_after_consecutive_retryable=args.pause_after_consecutive_retryable,
        )
        print(
            "Incremental batch complete: "
            f"total={summary.total_samples}, skipped={summary.skipped_success}, "
            f"succeeded={summary.succeeded}, failed={summary.failed}, paused={summary.paused}"
        )
        if summary.fatal_error:
            print(f"Fatal error: {summary.fatal_error}")
        return 1 if summary.paused and summary.succeeded == 0 else 0

    input_dir = args.input_path or args.input_dir
    pairs, missing_wav, missing_json = pair_directory(input_dir)
    if missing_wav:
        print(f"Skipping {len(missing_wav)} JSON file(s) without matching WAV: {', '.join(missing_wav)}")
    if missing_json:
        print(f"Skipping {len(missing_json)} WAV file(s) without matching JSON: {', '.join(missing_json)}")
    if not pairs:
        print("No matching <sample_id>.wav + <sample_id>.json pairs found.")
        return 1

    client = None if args.dry_run or args.provider == "relay" else create_client()
    validation_config = ValidationConfig(confidence_threshold=args.confidence_threshold)
    results: list[EvaluationResult] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Processing {pair.sample_id}...")
        try:
            feature_data = load_feature_json(pair.json_path, pair.sample_id)
            if args.dry_run:
                raw_response = build_demo_model_response(feature_data)
            else:
                prompt = build_prompt(feature_data.transcript, feature_data.anchors)
                if args.provider == "relay":
                    raw_response = annotate_audio_via_relay(model_id, pair.wav_path, prompt)
                else:
                    raw_response = annotate_audio(client, model_id, pair.wav_path, prompt)
            model_output = parse_model_output(raw_response)
            results.append(evaluate_model_output(pair.sample_id, feature_data, model_output, validation_config))
        except FeatureFormatError as exc:
            results.append(_invalid_input_result(pair.sample_id, str(exc)))
            print(f"  Feature JSON error: {exc}")
        except ModelOutputParseError as exc:
            results.append(_invalid_model_result(pair.sample_id, str(exc)))
            print(f"  Model output error: {exc}")
        except Exception as exc:
            results.append(_api_failed_result(pair.sample_id, str(exc)))
            print(f"  Annotation error: {exc}")

    paths = write_evaluation_results(results, args.output_dir)
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    return 0


def _invalid_input_result(sample_id: str, error: str) -> EvaluationResult:
    return EvaluationResult(
        sample_id=sample_id,
        validation_status=ValidationStatus.INVALID_INPUT,
        validation_reasons=[ValidationReason.MISSING_REQUIRED_FIELD.value],
        needs_human_review=True,
        error=error,
    )


def _invalid_model_result(sample_id: str, error: str) -> EvaluationResult:
    return EvaluationResult(
        sample_id=sample_id,
        validation_status=ValidationStatus.REVIEW,
        validation_reasons=[ValidationReason.INVALID_MODEL_OUTPUT.value],
        needs_human_review=True,
        error=error,
    )


def _api_failed_result(sample_id: str, error: str) -> EvaluationResult:
    return EvaluationResult(
        sample_id=sample_id,
        validation_status=ValidationStatus.REVIEW,
        validation_reasons=[ValidationReason.API_CALL_FAILED.value],
        needs_human_review=True,
        error=error,
    )


def _dry_run_annotator_from_manifest(model_id: str, wav_path: Path, prompt: str) -> str:
    import json
    import re

    transcript_match = re.search(r'"([^"]+)"', prompt)
    transcript = transcript_match.group(1) if transcript_match else ""
    anchor_ids = re.findall(r'"([^"]+)"\s*:\s*\{', prompt) or ["tag_1"]
    return json.dumps(
        {
            "transcript": transcript,
            "annotated_text": transcript,
            "annotations": [
                {
                    "anchor_id": anchor_id,
                    "label": "RISING",
                    "confidence": 0.82,
                    "reason": "Dry-run mock annotation.",
                }
                for anchor_id in anchor_ids
            ],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
