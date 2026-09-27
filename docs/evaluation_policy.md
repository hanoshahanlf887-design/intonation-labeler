# Evaluation Policy

This project is an AI-assisted intonation evaluation and annotation workflow.
It is not an autonomous agent and does not replace expert human review.

## Model Confidence

`model_confidence` is self-reported by the model. It is useful as one routing
signal for review, but it is not a calibrated probability and must not be used
as the only basis for `final_label`.

## Validation

The validation layer is deterministic and does not call an LLM. It checks:

- parseable structured JSON output
- required fields
- complete anchor coverage
- allowed label values
- transcript preservation
- acoustic feature presence
- obviously abnormal acoustic feature values
- low model confidence

## Transcript Preservation

Transcript validation uses a deliberately small normalization step before
comparison:

- trim surrounding whitespace
- remove one matched pair of surrounding single or double quotes

This was added after real pilot runs showed formatting-only false positives
where the model returned the full transcript wrapped in quotes. The validator
still treats real additions, deletions, punctuation changes, case changes,
contraction rewrites, and other textual changes as `TRANSCRIPT_CHANGED`.

## Acoustic Conflict Rules

The first public version deliberately does not infer intonation directly from
simple F0 thresholds such as `slope > 0 = RISING` or `slope < 0 = FALLING`.
The current synthetic schema documents feature names, but it does not provide
project-specific thresholds that are reliable enough to override or conflict
with multimodal model output.

The code keeps a reserved `detect_acoustic_conflicts` extension point. Conflict
rules should only be enabled after a small labeled validation set and explicit
feature interpretation policy exist.

## Bad Cases

Bad cases are defined as:

- all non-PASS samples
- PASS samples later changed by human override

Bad cases are exported for prompt iteration, rule refinement, and review
operations. They are not benchmark errors unless ground-truth labels are added.

## Metrics

Without ground-truth labels, reports must not include accuracy, precision,
recall, or F1. Summary reports only describe workflow status, routing, labels,
validation reasons, and human review activity.
