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

from intonation_labeler.demo import build_demo_result
from intonation_labeler.export import AnnotationResult, write_batch_results
from intonation_labeler.gemini_client import annotate_audio, create_client
from intonation_labeler.io import FeatureFormatError, load_feature_json, pair_directory
from intonation_labeler.prompts import build_prompt


DEFAULT_MODEL = "gemini-2.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch annotate intonation from paired WAV and JSON files.")
    parser.add_argument("input_path", nargs="?", help="Directory containing <sample_id>.wav and <sample_id>.json files.")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data"), help="Directory containing <sample_id>.wav and <sample_id>.json files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"), help="Directory for result files.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id.")
    parser.add_argument("--dry-run", action="store_true", help="Use mock annotations without calling Gemini.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    input_dir = args.input_path or args.input_dir
    pairs, missing_wav, missing_json = pair_directory(input_dir)
    if missing_wav:
        print(f"Skipping {len(missing_wav)} JSON file(s) without matching WAV: {', '.join(missing_wav)}")
    if missing_json:
        print(f"Skipping {len(missing_json)} WAV file(s) without matching JSON: {', '.join(missing_json)}")
    if not pairs:
        print("No matching <sample_id>.wav + <sample_id>.json pairs found.")
        return 1

    client = None if args.dry_run else create_client()
    results: list[AnnotationResult] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Processing {pair.sample_id}...")
        try:
            feature_data = load_feature_json(pair.json_path, pair.sample_id)
            if args.dry_run:
                result = build_demo_result(feature_data)
            else:
                prompt = build_prompt(feature_data.transcript, feature_data.anchors)
                result = annotate_audio(client, args.model, pair.wav_path, prompt)
            results.append(AnnotationResult(sample_id=pair.sample_id, result=result))
        except FeatureFormatError as exc:
            results.append(AnnotationResult(sample_id=pair.sample_id, error=str(exc)))
            print(f"  Feature JSON error: {exc}")
        except Exception as exc:
            results.append(AnnotationResult(sample_id=pair.sample_id, error=str(exc)))
            print(f"  Annotation error: {exc}")

    txt_path, json_path = write_batch_results(results, args.output_dir)
    print(f"Saved merged text result: {txt_path}")
    print(f"Saved JSON result: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
