from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .report import build_bad_cases, build_summary, render_summary_markdown
from .schema import EvaluationResult


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


def render_evaluation_json(results: list[EvaluationResult]) -> str:
    return json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2)


def build_evaluation_zip(results: list[EvaluationResult]) -> bytes:
    buf = io.BytesIO()
    summary = build_summary(results)
    bad_cases = build_bad_cases(results)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evaluation_results.json", render_evaluation_json(results))
        zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        zf.writestr("summary.md", render_summary_markdown(summary))
        zf.writestr("bad_cases.json", json.dumps(bad_cases, ensure_ascii=False, indent=2))
        zf.writestr("m_results.txt", render_evaluation_text(results))
        for item in results:
            zf.writestr(f"individual/{item.sample_id}.json", json.dumps(item.to_dict(), ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.read()


def render_evaluation_text(results: list[EvaluationResult]) -> str:
    lines: list[str] = []
    for item in results:
        lines.append(f"ID: {item.sample_id}")
        lines.append(f"Validation: {item.validation_status.value}")
        if item.validation_reasons:
            lines.append(f"Reasons: {', '.join(item.validation_reasons)}")
        if item.model_output and item.model_output.annotated_text:
            lines.append(f"Result: {item.model_output.annotated_text}")
        elif item.error:
            lines.append(f"ERROR: {item.error}")
        else:
            lines.append(f"Labels: {json.dumps(item.model_label, ensure_ascii=False)}")
        lines.append(f"Final label: {json.dumps(item.final_label, ensure_ascii=False)}")
        lines.append("-" * 60)
    return "\n".join(lines)


def write_batch_results(results: list[AnnotationResult], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    txt_path = _next_available_path(output_path / "m_results.txt")
    json_path = txt_path.with_suffix(".json")
    txt_path.write_text(render_merged_text(results), encoding="utf-8")
    json_path.write_text(render_results_json(results), encoding="utf-8")
    return txt_path, json_path


def write_evaluation_results(results: list[EvaluationResult], output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = _next_available_path(output_path / "evaluation_results.json")
    base = result_path.with_suffix("")
    summary_path = base.with_name(f"{base.name}_summary.json")
    summary_md_path = base.with_name(f"{base.name}_summary.md")
    bad_cases_path = base.with_name(f"{base.name}_bad_cases.json")
    text_path = base.with_name(f"{base.name}_m_results.txt")

    summary = build_summary(results)
    result_path.write_text(render_evaluation_json(results), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    bad_cases_path.write_text(json.dumps(build_bad_cases(results), ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(render_evaluation_text(results), encoding="utf-8")
    return {
        "evaluation_results": result_path,
        "summary_json": summary_path,
        "summary_markdown": summary_md_path,
        "bad_cases": bad_cases_path,
        "text_results": text_path,
    }


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

