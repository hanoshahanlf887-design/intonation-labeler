# Intonation Labeler

AI-assisted Audio Intonation Evaluation Workflow for English utterance-level and
anchor-level intonation review.

This project is not an autonomous Agent, an ASR system, or a simple Gemini API
demo. It is a workflow that combines multimodal model first-pass annotation with
deterministic Python validation, risk-based human review, final-label handling,
summary reporting, and bad-case export.

```text
WAV + acoustic features
  -> multimodal model first-pass annotation
  -> structured output
  -> deterministic validation
  -> risk-based human review
  -> final label
  -> summary / bad-case export
```

The Claude Code Skill in this repository orchestrates the workflow. Deterministic
business rules live in testable Python modules.

## Project Overview

`intonation-labeler` evaluates pre-selected transcript anchors in audio clips.
Each sample combines:

- a `.wav` file
- a feature JSON file containing transcript, anchor ids, anchor words, timing,
  and acoustic features
- a model response parsed into a stable sample-level and anchor-level schema

The workflow preserves the model prediction, validation result, human decision,
and final label as separate fields so later review does not overwrite the raw
model output.

## Why This Project

The original problem was English question intonation annotation. The upgraded
version turns that task into a small AI evaluation workflow:

- prompt-based first-pass labeling
- structured model output
- deterministic validation
- bad-case routing
- human-in-the-loop correction
- reviewable final labels
- reliability features for unstable provider calls

The goal is operational evaluation quality: reliable data flow, transparent
review routing, and auditable decisions.

## Workflow

```text
Input directory or manifest
  -> pair WAV and feature JSON
  -> build prompt from transcript, anchors, and acoustic features
  -> call provider or dry-run mock
  -> parse structured JSON
  -> validate output and inputs
  -> route PASS / REVIEW / CONFLICT / INVALID_INPUT
  -> optional human review
  -> final labels
  -> summary and bad-case files
```

`PASS` samples default to `final_label = model_label`. `REVIEW`, `CONFLICT`,
and `INVALID_INPUT` samples keep `final_label = null` until human review.

## Key Features

- Official Gemini provider and Gemini-compatible relay provider
- Audio input with pre-extracted acoustic features
- Structured model output parser with code-fence robustness
- Sample-level and anchor-level labels
- Deterministic validation without LLM calls
- Confidence-based review routing
- Human review fields and final-label assignment
- Streamlit Review Queue
- JSON / Markdown / TXT / ZIP exports
- Bad-case export
- Dry-run mode for demos and tests
- Incremental persistence, resume, retry/backoff, and error classification
- Claude Code Skill for workflow orchestration

## Architecture / Project Structure

```text
intonation-labeler/
|-- app/
|   `-- streamlit_app.py          # Review Queue and demo UI
|-- scripts/
|   `-- batch_annotate.py         # Batch CLI
|-- src/intonation_labeler/
|   |-- gemini_client.py          # Official Gemini provider
|   |-- relay_client.py           # Gemini-compatible relay provider
|   |-- incremental.py            # Resume/retry/incremental persistence
|   |-- io.py                     # Input pairing and feature JSON loading
|   |-- prompts.py                # Prompt construction
|   |-- schema.py                 # Structured result objects
|   |-- validation.py             # Deterministic validation
|   |-- review.py                 # Human review helpers
|   |-- report.py                 # Summary and bad-case report data
|   |-- export.py                 # File exports
|   `-- demo.py                   # Synthetic dry-run output
|-- examples/
|   `-- synthetic_001.json
|-- docs/
|   |-- evaluation_policy.md
|   `-- workflow.md
|-- tests/
|-- .claudecode/skills/
|   `-- audio_annotation.md
|-- .env.example
|-- requirements.txt
`-- pyproject.toml
```

## Validation Design

Validation is deterministic and does not call another model. It checks:

- model output can be parsed as structured JSON
- required fields exist
- every expected anchor has a result
- no unknown anchor ids are returned
- labels are in `RISING`, `FALLING`, or `LEVEL`
- transcript is preserved after minimal transcript normalization
- acoustic features are present and parseable
- model self-reported confidence is not below the configured threshold

`model_confidence` is self-reported by the model. It is not calibrated
probability and is not safe as the only automatic release signal.

### V2 Transcript Normalization

V1 used strict string comparison for transcript preservation. Real pilot runs
found five `TRANSCRIPT_CHANGED` reviews where the only difference was a matched
pair of outer quotes around the full transcript.

V2 adds a minimal deterministic normalization:

- trim surrounding whitespace
- remove one matched pair of surrounding single or double quotes

It does not lowercase text, remove punctuation, remove sentence-internal quotes,
expand contractions, or use semantic similarity. Synthetic tests still require
`TRANSCRIPT_CHANGED` for genuine add/delete/rewrite cases.

## Human-in-the-loop Review

The Review Queue lets a reviewer inspect:

- transcript
- model label and confidence
- model reason
- acoustic features
- validation status and reasons
- final-label controls
- review note

Human review records are separate from model predictions. The workflow preserves:

- raw model output
- validation result
- human review decision
- final label

## Small-scale Evaluation

A random Pilot was run on:

- 25 WAV files
- 28 anchors
- single-person blind manual reference

After one manual reference correction, 26 of 28 anchor predictions matched the
manual reference: 92.86% on this small Pilot.

This number is an observation for this Pilot only. It is not a formal benchmark,
not a claim of overall model accuracy, and not a gold-standard evaluation. The
manual reference was single-person blind labeling, and the label distribution was
highly imbalanced:

- human `FALLING`: 25
- human `RISING`: 3
- human `LEVEL`: 0

Model self-reported confidence was concentrated between 0.85 and 0.95. The
current `confidence < 0.6` routing threshold did not capture the two model
errors. Both errors were high-confidence `RISING -> FALLING` cases, reinforcing
that self-reported confidence is not calibrated and should not be the sole
automatic pass criterion.

## Bad Case Findings

The random Pilot found:

- 2 `RISING -> FALLING` errors
- both were `PASS` under V1 validation
- both had high self-reported confidence

A separate RISING-focused Challenge Set was then built:

- 3 new human-labeled `RISING` anchors
- 5 randomly sampled `FALLING` controls

Results:

- 2 of 3 new `RISING` anchors were predicted as `RISING`
- 1 of 3 new `RISING` anchors was predicted as `FALLING`
- all 5 `FALLING` controls were predicted correctly

This is failure-pattern evidence, not a benchmark. With only three new RISING
examples, it cannot support a general RISING recall claim.

## Acoustic Bad Case Analysis

A follow-up analysis compared:

- correct RISING cases
- `RISING -> FALLING` cases
- randomly sampled correct FALLING controls

Observations:

- earlier `t_maxf0` in error RISING cases is a candidate risk signal
- slope was not stable enough to use as a rule
- acoustic feature overlap between error RISING and correct FALLING was obvious

The project deliberately does not enable `ACOUSTIC_CONFLICT` rules yet. This is
an engineering choice: exploratory correlations should not be hard-coded into
deterministic validation until there is enough labeled evidence and a clear
feature interpretation policy.

## V1 -> V2 Rule Iteration

V1:

- strict transcript string comparison
- five formatting-only `TRANSCRIPT_CHANGED` reviews in real runs

Bad-case review:

- all five were caused by the model wrapping the whole transcript in quotes
- no actual words were added, deleted, or rewritten

V2:

- minimal transcript normalization for surrounding whitespace and one matched
  pair of outer quotes
- frozen-result revalidation removed all five formatting-only reviews
- synthetic add/delete/rewrite tests still trigger `TRANSCRIPT_CHANGED`

V2 does not solve prediction errors. The known `RISING -> FALLING` errors remain
`PASS` without acoustic conflict rules.

## Reliability / Resume

Real batch runs can be interrupted by provider issues such as rate limits,
timeouts, and transient disconnects. The batch runner includes:

- incremental per-sample persistence
- resume that skips completed samples
- retry/backoff for retryable failures
- error classification
- batch pause after repeated retryable failures
- protection against overwriting successful results with later failures

Public configuration uses generic provider placeholders. Do not publish private
provider domains or API keys.

## Quick Start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Copy the environment template:

```text
.env.example -> .env
```

Use placeholders such as:

```text
GOOGLE_API_KEY=YOUR_API_KEY
GEMINI_RELAY_BASE_URL=YOUR_RELAY_BASE_URL
GEMINI_RELAY_API_KEY=YOUR_API_KEY
```

`.env` is ignored by Git.

## Dry-run

Dry-run mode uses synthetic model output and does not call an external API.

```bash
python scripts/batch_annotate.py path/to/input --dry-run --output-dir results
```

Live official provider:

```bash
python scripts/batch_annotate.py --input-dir path/to/input --output-dir results --provider official
```

Live Gemini-compatible relay provider:

```bash
python scripts/batch_annotate.py --manifest path/to/manifest.csv --output-dir results --provider relay --model gemini-3.8-flash
```

The manifest used for inference must not contain human reference labels.

## Tests

```bash
python -m unittest discover -s tests
python -m compileall app scripts src tests
```

Current suite: 43 tests.

Tests cover:

- schema and validation behavior
- parser robustness
- transcript normalization
- review and final-label handling
- reporting and bad-case export
- incremental persistence and resume
- retry/error classification

Tests use synthetic data and do not call Gemini.

## Limitations

- The Pilot is small and label-imbalanced.
- Manual reference labels are single-person blind labels, not a formal gold
  standard.
- Model confidence is self-reported and not calibrated.
- The confidence threshold has not been optimized by a formal benchmark.
- Acoustic conflict rules are reserved but not enabled.
- The RISING Challenge Set is very small and cannot establish model recall.
- Provider availability depends on external APIs.
- This project is not an autonomous Agent.

## Privacy / Data

- Real audio does not belong in Git.
- Public examples are synthetic.
- API keys are configured through environment variables or `.env`.
- `.env`, `data/`, `results/`, raw audio, and private evaluation outputs are
  ignored.
- Evaluation files containing private data should remain outside the repository.
