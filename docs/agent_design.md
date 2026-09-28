# Thin Evaluation Agent Design

This document describes the optional thin Agent layer built above the existing
intonation evaluation workflow.

The existing workflow remains deterministic. The Agent does not judge intonation,
replace validation rules, fill human labels, tune thresholds, or modify prompts.

## Why The Existing Workflow Is Not An Agent

The current workflow executes a fixed sequence:

```text
input pairing
-> prompt construction
-> model/dry-run inference
-> structured parsing
-> deterministic validation
-> review routing
-> report export
```

That is a reliable evaluation workflow, but it does not observe task state and
choose among alternative next actions.

## Agent Responsibility

The Evaluation Agent adds state-driven orchestration:

```text
AgentState
-> policy decision
-> whitelisted tool call
-> observation
-> updated AgentState
-> next decision
```

The Agent decides whether to inspect, run, resume, prepare review, stop for human
review, or generate reports. It does not implement the evaluation business rules.

## Components

```text
src/intonation_labeler/agent/
|-- state.py    # serializable AgentState and statuses
|-- policy.py   # deterministic decision policy and action whitelist
|-- tools.py    # thin wrappers around existing workflow modules
`-- runner.py   # agent loop and action trace
```

CLI demo:

```text
scripts/evaluation_agent.py
```

## State

The state records task identity, input/output locations, provider/model, current
status, counts, review status, report readiness, last error, and allowed next
actions.

Statuses:

- `NEW`
- `RUNNING`
- `PARTIAL`
- `WAITING_FOR_REVIEW`
- `COMPLETED`
- `FAILED`
- `BLOCKED`

## Tools

The Agent can call only these tools:

- `inspect_dataset`
- `run_evaluation_batch`
- `resume_evaluation_batch`
- `get_batch_status`
- `get_validation_summary`
- `get_review_queue`
- `get_case_detail`
- `generate_report`

The tools call existing Python modules such as `io.py`, `incremental.py`,
`validation.py`, `report.py`, and `export.py`.

## Decision Policy

Examples:

```text
NEW -> inspect_dataset
valid dataset -> run_evaluation_batch
PARTIAL + retryable error -> resume_evaluation_batch
fatal auth error -> BLOCKED / stop
review cases -> get_review_queue -> WAITING_FOR_REVIEW
WAITING_FOR_REVIEW + pending human review -> stop
review complete -> generate_report
complete with no review -> generate_report -> COMPLETED
```

## Human Review Boundary

A risk case is not a confirmed model error. `REVIEW`, `LOW_MODEL_CONFIDENCE`, and
`TRANSCRIPT_CHANGED` mean the case needs attention. A confirmed model error
requires either a human override or an independent reference comparison.

The Agent must stop in `WAITING_FOR_REVIEW` and must not write human labels.

## Action Trace

Each step records:

- `state_before`
- `selected_action`
- `observation`
- `tool_result_summary`
- `state_after`

The trace explains why the Agent chose the next action without recording API keys
or private transcript content.

## Skill Relationship

The Claude Code Skill remains a reusable domain procedure and workflow guide.
The Agent is a state-driven decision layer. Python modules remain the source of
deterministic execution.
