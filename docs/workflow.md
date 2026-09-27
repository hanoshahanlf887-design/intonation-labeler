# Workflow

```text
WAV audio + acoustic-feature JSON
        |
        v
input pairing and schema validation
        |
        v
prompt construction
        |
        v
Demo mock output or Gemini multimodal first-pass annotation
        |
        v
structured model output parsing
        |
        v
deterministic validation
        |
        v
risk-based review routing
        |
        v
human review for REVIEW / CONFLICT / INVALID_INPUT cases
        |
        v
final labels + summary + bad-case export
```

The workflow preserves:

- original input
- raw model response
- parsed model prediction
- validation status and reasons
- human decision
- final label

The Claude Code Skill in `.claudecode/skills/audio_annotation.md` only
orchestrates this workflow. Deterministic business logic lives in Python modules
under `src/intonation_labeler/`.
