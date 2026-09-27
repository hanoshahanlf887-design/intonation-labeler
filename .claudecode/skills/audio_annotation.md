# Audio Intonation Evaluation Workflow

## Purpose

Use this skill to run the reusable AI-assisted intonation evaluation workflow.
This is a fixed workflow orchestration skill, not an autonomous Agent.

## Trigger Examples

- "run intonation evaluation"
- "process the demo input"
- "start audio annotation review"
- "generate validation and bad cases"

## Workflow

Run each step in order and stop if a required step fails.

### 1. Check Inputs

Confirm the selected input directory contains matching `.wav` and `.json` files
with the same stem.

For public demos, use only synthetic or user-provided non-sensitive files. Do not
copy private `data/` or `results/` directories into this repository.

### 2. Check Environment

Check that Python and project dependencies are available.

```powershell
python -m unittest discover -s tests
```

If dependencies are missing, ask the user to install them with:

```powershell
pip install -r requirements.txt
```

### 3. Choose Mode

Prefer dry-run/demo mode unless the user explicitly asks for live Gemini
inference and confirms that API usage is acceptable.

Dry run:

```powershell
python scripts/batch_annotate.py path\to\input --dry-run
```

Live Gemini:

```powershell
python scripts/batch_annotate.py --input-dir path\to\input --output-dir results
```

### 4. Run Batch Annotation

The Python workflow handles:

- input pairing
- feature JSON parsing
- prompt construction
- structured model output parsing
- deterministic validation
- review routing
- summary generation
- bad-case export

The skill should not duplicate validation rules in prose.

### 5. Review Routing

Explain the routing result:

- `PASS`: final labels default to model labels
- `REVIEW`: user should inspect the case
- `CONFLICT`: reserved for enabled acoustic conflict rules
- `INVALID_INPUT`: input or output could not be used safely

### 6. Human Review

For cases requiring review, ask the user to inspect the Review Queue in the
Streamlit app or edit the exported structured result with explicit human labels
and notes.

Human review must preserve the original model output, validation result, human
decision, and final label.

### 7. Report Outputs

Point the user to:

- `evaluation_results.json`
- `summary.json`
- `summary.md`
- `bad_cases.json`
- `m_results.txt`

Do not report accuracy, precision, recall, or F1 unless a real ground-truth
benchmark has been added.

## Safety Rules

- Do not print API keys or secrets.
- Do not upload private audio or annotation data.
- Do not commit `.env`, `data/`, `results/`, or raw `.wav` files.
- Do not call live Gemini APIs in dry-run/demo mode.
- Do not describe this workflow as an autonomous Agent.
