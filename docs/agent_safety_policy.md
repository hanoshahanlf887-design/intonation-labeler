# Agent Safety Policy

The Evaluation Agent is intentionally thin. It orchestrates existing evaluation
workflow tools but does not change evaluation logic.

## Allowed Actions

The Agent may call:

- inspect dataset
- run dry-run or configured evaluation batch
- resume a partial batch
- inspect batch status
- summarize validation outcomes
- prepare a review queue
- inspect a case detail
- generate reports

## Forbidden Actions

The action whitelist rejects:

- `modify_prompt`
- `modify_validation_rule`
- `modify_threshold`
- `write_human_label`
- `enable_acoustic_conflict`
- `git_commit`
- `git_push`

## Human Authority

The Agent must not complete human review. It may route cases to review and stop.
Only a human reviewer can accept or override model labels.

## No Autonomous Rule Changes

If repeated validation or provider patterns suggest a possible improvement, the
Agent may produce a proposal in a future extension, but it must not apply the
change. Prompt, validation, threshold, and acoustic conflict settings require
explicit human approval and code changes outside the Agent loop.

## Privacy

Action traces should record counts, statuses, action names, and compact failure
classes. They should not record API keys, credentials, raw private audio, or full
private transcripts.

## Metrics Boundary

Without human reference labels or human overrides, review cases are not confirmed
model errors. The Agent must not report accuracy, precision, recall, or F1 unless
an explicit reference dataset is supplied by a separate evaluation stage.
