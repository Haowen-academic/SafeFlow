# SafeFlow Paper Experiment Protocol

This document describes the executable protocol for the experiments reported
in the SafeFlow manuscript. The runner records one raw JSONL record for every
method and benchmark instance, then derives JSON and CSV summaries from those
records. It never reads values from a paper table as experiment inputs.

## Prerequisites

Use Python 3.10 or later and install the repository dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Live runs require an enabled model provider. Configure either the Azure
settings or the OpenAI-compatible settings in `.env`; the latter requires
`OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_BASE_URL`, and
`OPENAI_COMPAT_MODEL`. Do not commit `.env` or provider keys.

The commands below are run from the repository root. Output directories should
be treated as run artifacts: retain their JSONL files with the final reported
tables so results can be audited.

The public release includes only the compact ASB-100 sample under
`safeagents/datasets/asb/master100_release/`. The default command therefore
runs ASB in a fresh clone. To run the four-benchmark protocol, obtain the
normalized benchmark inputs separately and pass their `sampled_110_random`
directory with `--dataset` (and explicitly select all four benchmarks).

## Main comparison

The main-comparison harness supports SafeAgents (undefended), GuardAgent,
AutoDefense, AegisLLM, and SafeFlow on ASB, AgentHarm, RedCode, and SafeArena.
The default public-release input contains only ASB; the remaining benchmark
inputs and their native runtime environments are separate prerequisites.

```powershell
python -m safeagents.core.src.evaluation.paper_experiments main `
  --include-benign `
  --output-dir logs/paper_main
```

Run a one-example provider smoke test first:

```powershell
python -m safeagents.core.src.evaluation.paper_experiments main `
  --limit 1 `
  --output-dir logs/paper_smoke
```

With a separately obtained four-benchmark sample, use for example:

```powershell
python -m safeagents.core.src.evaluation.paper_experiments main `
  --dataset D:/benchmarks/sampled_110_random `
  --benchmarks ASB,AgentHarm,RedCode,SafeArena `
  --include-benign `
  --output-dir logs/paper_main
```

The three external-defense categories currently run the repository's clearly
identified lightweight adapters (`GuardAgentLite`, `AutoDefenseASB`, and
`AegisLLMLite`), not the authors' original evaluation stacks. Each JSONL row
and run manifest records the exact implementation. Do not label adapter output
as a result from the corresponding published baseline.

`--include-benign` loads only available public companion tasks. ASB and
AgentHarm have explicit pair identifiers. RedCode and SafeArena currently have
no verified harmful--benign mapping in this release, so their `paired` result
is deliberately `null`; it must not be inferred from row order, task wording,
or a modulo match. TCR is computed only when the runtime provides a
benchmark-native completion signal. The optional `--trace-completion-proxy` is
for adapter debugging only and its output is explicitly marked as a proxy.

## Robustness and sensitivity experiments

```powershell
python -m safeagents.core.src.evaluation.paper_experiments jailbreak --output-dir logs/paper_jailbreak
python -m safeagents.core.src.evaluation.paper_experiments model_sensitivity --output-dir logs/paper_models
python -m safeagents.core.src.evaluation.paper_experiments ablation --output-dir logs/paper_ablation
python -m safeagents.core.src.evaluation.paper_experiments policy --output-dir logs/paper_policy
python -m safeagents.core.src.evaluation.paper_experiments context_scope --output-dir logs/paper_context
python -m safeagents.core.src.evaluation.paper_experiments adaptive --output-dir logs/paper_adaptive
python -m safeagents.core.src.evaluation.paper_experiments defense_injection --output-dir logs/paper_injection
```

These commands cover the four jailbreak wrappers, defense-model sensitivity,
five pipeline ablations, label-schema and partial-propagation sensitivity,
context-reconstruction scope, adaptive workflow attacks, and defense-side
indirect injections. The main command also writes the Prompt-Local / Cross-Agent
diagnostic stratification. This split is derived only from emitted message and
tool-call traces, never from the benchmark prompt, final defense verdict, or
attack outcome. Incomplete traces are recorded as `ambiguous` and excluded from
the two diagnostic slices rather than being guessed into a category.

## Output and statistics

Every command writes:

- `<experiment>_records.jsonl`: one unaggregated execution record per method and task;
- `<experiment>_summary.json` and `.csv`: ASR, TCR, FPR, paired success, runtime, model-call, and reconstruction summaries when available;
- `run_manifest.json`: dataset, benchmark, framework, and execution metadata.

The summaries calculate Wilson 95% confidence intervals and one-sided pooled
two-proportion tests from completed harmful runs. They are reproducibility
checks, not substitutes for matching the manuscript's exact benchmark slice,
provider versions, seeds, and benchmark-native success callbacks.

## Offline checks

Use offline mode only to validate parser, perturbation, metric, and output
plumbing; it intentionally produces `offline_not_executed` records and no
benchmark score:

```powershell
python -m safeagents.core.src.evaluation.paper_experiments main `
  --limit 1 --offline --output-dir build_check/paper_smoke
python -m pytest tests/test_paper_experiments.py tests/test_safeflow_release.py -q
```

An offline record, an execution error, a proxy completion result, or a failed
provider authentication must never be copied into a paper table as a measured
experimental result.

## Reproduction coverage and limitations

The runner implements every experimental *condition* described in the paper,
but a paper-level reproduction additionally requires the upstream method and
benchmark environments. See [experiment coverage](paper-experiment-coverage.md)
for the exact boundary between runnable local plumbing and official evaluation.
