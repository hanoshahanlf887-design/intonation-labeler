# Intonation Labeler

Multimodal LLM-assisted English intonation labeling with audio and pre-extracted acoustic features.

## Overview

Intonation Labeler is a small annotation-assistance tool for English intonation labeling. It accepts WAV audio and a matching acoustic-feature JSON file, then combines transcript text, acoustic cues, and Gemini multimodal audio understanding to produce initial intonation labels.

The project is intended for annotation assistance and first-pass screening. It is not a replacement for expert human review.

## Workflow

```text
WAV audio
+
acoustic-feature JSON
        |
        v
input validation & pairing
        |
        v
prompt construction
        |
        v
Gemini multimodal inference
        |
        v
annotation result
        |
        v
TXT / JSON / ZIP export
```

Demo mode runs the same local parsing, pairing, and export workflow without making Gemini API calls.

## Project Structure

```text
MTI_Project/
├── app/                      # Streamlit UI
├── scripts/                  # Batch entry point
├── src/intonation_labeler/    # Shared parsing, prompt, Gemini, export, demo logic
├── examples/                 # Synthetic schema example
├── tests/                    # Standard-library unittest suite
├── .env.example
├── requirements.txt
└── 启动网页版标注工具.bat
```

## Input Format

Files are paired by stem:

```text
sample_001.wav
sample_001.json
```

Canonical JSON schema:

```json
{
  "transcript": "Could we meet after lunch?",
  "anchors": {
    "tag_1": {
      "word": "lunch?",
      "feature": {
        "t_maxf0": "0.78",
        "z_f0": "0.64",
        "rangeF0": "92",
        "slope": "18.40",
        "t_min": "0.12"
      },
      "start": 1.05,
      "end": 1.72
    }
  }
}
```

## Installation

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
```

For editable local imports:

```bash
pip install -e .
```

## Configuration

Copy the example environment file and add your Gemini API key:

```text
.env.example -> .env
```

```text
GOOGLE_API_KEY=your_api_key_here
```

`.env` is ignored by Git. Do not commit credentials.

## Run The Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Windows users can also run:

```text
启动网页版标注工具.bat
```

## Demo Mode

Demo / Dry Run mode:

- does not require an API key
- does not call Gemini
- does not upload audio
- returns a clearly marked mock annotation
- is useful for checking parsing, pairing, and export behavior

## Batch Usage

Dry run requires a local input directory with matching WAV and JSON stems. The JSON can follow `examples/synthetic_001.json`, but the WAV should be your own file:

```text
demo_input/
├── sample_001.wav
└── sample_001.json
```

The stems must match exactly: `sample_001.wav` pairs with `sample_001.json`.

Run dry mode without calling Gemini:

```bash
python scripts/batch_annotate.py demo_input --dry-run
```

Live Gemini inference:

```bash
python scripts/batch_annotate.py --input-dir path/to/input --output-dir results
```

For live inference, each JSON file must also have a matching WAV file with the same stem.

## Example

See `examples/synthetic_001.json` for a fully synthetic schema example. It does not include original internship audio, private transcripts, or historical sample IDs.

To run live Gemini inference with this example name, provide your own `examples/synthetic_001.wav`.

## Limitations

- Acoustic features must currently be pre-extracted.
- This repository does not currently perform Praat or TextGrid feature extraction.
- Live inference requires Gemini API access.
- LLM output should be manually reviewed.
- Demo mode output is synthetic/mock, not model inference.
- Model behavior may vary by model version.

## Privacy / Data

This repository contains no original internship audio, no private transcripts, and no API credentials. Included examples are synthetic.

## Evaluation

This repository focuses on the engineering workflow and reproducible demo interface. No public benchmark is claimed because the original evaluation data are not distributed with the project.

The tool originated from an experimental workflow for reducing manual screening effort in English intonation annotation.
