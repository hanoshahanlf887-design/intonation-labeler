from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


class FeatureFormatError(ValueError):
    """Raised when an acoustic feature JSON file is missing required fields."""


@dataclass(frozen=True)
class FeatureData:
    transcript: str
    anchors: dict[str, Any]


@dataclass(frozen=True)
class FilePair:
    sample_id: str
    wav_path: Path
    json_path: Path


def pair_uploaded_files(wav_files: Iterable[Any], json_files: Iterable[Any]) -> tuple[list[tuple[str, Any, Any]], list[str], list[str]]:
    wav_map = {Path(file.name).stem: file for file in wav_files}
    json_map = {Path(file.name).stem: file for file in json_files}
    matched_ids = sorted(set(wav_map) & set(json_map))
    pairs = [(sample_id, wav_map[sample_id], json_map[sample_id]) for sample_id in matched_ids]
    missing_wav = sorted(set(json_map) - set(wav_map))
    missing_json = sorted(set(wav_map) - set(json_map))
    return pairs, missing_wav, missing_json


def pair_directory(input_dir: str | Path) -> tuple[list[FilePair], list[str], list[str]]:
    base = Path(input_dir)
    wav_map = {path.stem: path for path in base.glob("*.wav")}
    json_map = {path.stem: path for path in base.glob("*.json")}
    matched_ids = sorted(set(wav_map) & set(json_map))
    pairs = [FilePair(sample_id, wav_map[sample_id], json_map[sample_id]) for sample_id in matched_ids]
    missing_wav = sorted(set(json_map) - set(wav_map))
    missing_json = sorted(set(wav_map) - set(json_map))
    return pairs, missing_wav, missing_json


def load_feature_json(source: str | Path | BinaryIO, sample_id: str | None = None) -> FeatureData:
    if hasattr(source, "seek"):
        source.seek(0)
        data = json.load(source)
    else:
        with Path(source).open("r", encoding="utf-8") as file:
            data = json.load(file)

    content = _extract_feature_content(data, sample_id)
    transcript = content.get("transcript")
    anchors = content.get("anchors")

    if not isinstance(transcript, str) or not transcript.strip():
        raise FeatureFormatError("Feature JSON must contain a non-empty string field: transcript.")
    if not isinstance(anchors, dict) or not anchors:
        raise FeatureFormatError("Feature JSON must contain a non-empty object field: anchors.")

    return FeatureData(transcript=transcript, anchors=anchors)


def _extract_feature_content(data: Any, sample_id: str | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise FeatureFormatError("Feature JSON must be a JSON object.")

    if "transcript" in data or "anchors" in data:
        return data

    legacy_keys = []
    if sample_id:
        legacy_keys.extend([sample_id, f"{sample_id}_feature"])
    legacy_keys.extend(key for key in data.keys() if key not in legacy_keys)

    for key in legacy_keys:
        value = data.get(key)
        if isinstance(value, dict) and ("transcript" in value or "anchors" in value):
            return value

    raise FeatureFormatError("Feature JSON must contain transcript and anchors fields.")

