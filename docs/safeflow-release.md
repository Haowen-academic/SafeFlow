# SafeFlow Research Release

This repository contains the SafeFlow implementation used to study information-flow control for multi-agent systems. SafeFlow combines rule-based taint annotation, subtask propagation, context reconstruction, local tool checks, and global validation.

## What Is Included

- `safeagents/core/src/safety/safeflow_research.py`: the maintained SafeFlow implementation.
- `safeagents/core/src/safety/recent_baselines.py`: lightweight, benchmark-adapted baseline defenses used in the comparison runner.
- `safeagents/datasets/asb/master100_release/`: a 100-sample ASB-derived release artifact, including schema, JSONL/CSV data, and aggregate results.
- `docs/examples/06_safeflow_demo.py`: a simulated SafeFlow walkthrough.
- `docs/examples/07_safeflow_experiments.py`: a configurable experiment runner.
- `safeagents/core/src/evaluation/unified_5way_comparison.py`: the five-method comparison runner.

## Installation

Use Python 3.10 or later. A virtual environment is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Copy `.env.example` to `.env` only when running model-backed agents or semantic validation. The deterministic safety rules and release smoke tests do not require credentials.

## Verify The Release

```bash
python -m pytest -q
python docs/examples/06_safeflow_demo.py --example 1
```

## Run The Released Master100 Artifact

The command below invokes the five-method runner. It requires a configured model provider because the agent-framework executions are model-backed.

```bash
python -m safeagents.core.src.evaluation.unified_5way_comparison \
  --dataset safeagents/datasets/asb/master100_release/master100.jsonl \
  --framework autogen \
  --method all \
  --output-dir logs/master100
```

The published Master100 files record the selected samples and aggregate outcomes. They are not a substitute for the upstream ASB benchmark. The full external benchmark collections and copied third-party baseline repositories are intentionally excluded from this repository; obtain and use them under their respective licenses.
