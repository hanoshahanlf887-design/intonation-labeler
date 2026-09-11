from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnnotationResult:
    sample_id: str
    result: str | None = None
    error: str | None = None


def render_merged_text(results: list[AnnotationResult]) -> str:
    lines: list[str] = []
    for item in results:
        lines.append(f"ID: {item.sample_id}")
        lines.append(f"Result: {item.result if item.result else f'ERROR: {item.error}'}")
        lines.append("-" * 60)
    return "\n".join(lines)


def render_results_json(results: list[AnnotationResult]) -> str:
    data = [
        {"id": item.sample_id, "result": item.result, "error": item.error}
        for item in results
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_results_zip(results: list[AnnotationResult]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in results:
            content = item.result if item.result else f"ERROR: {item.error}"
            zf.writestr(f"individual/{item.sample_id}.txt", content)
        zf.writestr("m_results.txt", render_merged_text(results))
        zf.writestr("m_results.json", render_results_json(results))
    buf.seek(0)
    return buf.read()


def write_batch_results(results: list[AnnotationResult], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    txt_path = _next_available_path(output_path / "m_results.txt")
    json_path = txt_path.with_suffix(".json")
    txt_path.write_text(render_merged_text(results), encoding="utf-8")
    json_path.write_text(render_results_json(results), encoding="utf-8")
    return txt_path, json_path


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available output path for {path}.")

