# SafeFlow

Research release of SafeAgents for building and evaluating safe multi-agent systems. SafeFlow combines rule-based taint annotation, subtask propagation, context reconstruction, local tool checks, and global validation.

Repository: [Haowen-academic/SafeFlow](https://github.com/Haowen-academic/SafeFlow)

## What Is Included

- `safeagents/core/src/safety/safeflow_research.py`: SafeFlow implementation.
- `safeagents/core/src/safety/recent_baselines.py`: benchmark-adapted baseline defenses.
- `safeagents/core/src/evaluation/unified_5way_comparison.py`: five-method comparison runner.
- `safeagents/datasets/asb/master100_release/`: released 100-sample ASB-derived artifact and aggregate results.
- `docs/examples/06_safeflow_demo.py`: deterministic SafeFlow walkthrough.
- `docs/examples/07_safeflow_experiments.py`: configurable experiment runner.

## Installation

Python 3.10 or later is required. A virtual environment is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Copy `.env.example` to `.env` and fill in a model provider configuration only when running model-backed agents or semantic validation. Deterministic safety rules and the release smoke tests do not require credentials.

## Verify the Release

```bash
python -m pytest -q
python docs/examples/06_safeflow_demo.py --example 1
```

## Run SafeFlow Examples

```bash
python docs/examples/06_safeflow_demo.py --example 1
python docs/examples/07_safeflow_experiments.py --help
```

The demo uses deterministic inputs. Experiment runs that invoke agent frameworks require a configured model provider.

## Reproduce the Master100 Experiment

The following command runs the five-method comparison on the released artifact:

```bash
python -m safeagents.core.src.evaluation.unified_5way_comparison \
  --dataset safeagents/datasets/asb/master100_release/master100.jsonl \
  --framework autogen \
  --method all \
  --output-dir logs/master100
```

The release artifact contains the selected samples, schema, and aggregate results. It is not a replacement for the upstream ASB benchmark. External benchmark collections and third-party baseline repositories are intentionally not included.

## Documentation

- [SafeFlow release notes](docs/safeflow-release.md)
- [Installation guide](docs/getting-started/installation.md)
- [Getting started](docs/getting-started/first-agent.md)
- [Examples](docs/examples/README.md)
- [Assessment guide](docs/guides/assessment.md)
- [Attack detection guide](docs/guides/attack-detection.md)

## License

This project is released under the [MIT License](LICENSE).

