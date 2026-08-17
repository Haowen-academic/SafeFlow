# SafeFlow Paper Experiment Coverage

This is an implementation audit for the experiments described in
`Association_for_Computational_Linguistics__ACL__conference/arxiv/SafeFlow_arxiv_combined.pdf`.
It distinguishes a runnable local experiment harness from a verified
paper-level reproduction. No result is inferred from the manuscript tables.

| Paper experiment | Local code | Current scope | Required before publication-level reproduction |
| --- | --- | --- | --- |
| Main comparison: ASB, AgentHarm, RedCode, SafeArena | `safeagents/core/src/evaluation/paper_experiments.py` | All four normalized 110-example input slices and five method categories run through one trace-recording interface when the input slices are supplied. This release bundles only the compact ASB-100 input. | Use each benchmark's native environment and success callback. The current RedCode and SafeArena adapters use generic tool-call handling; they do not execute RedCode's official isolated environment or SafeArena's authenticated site environment. |
| Baselines: SafeAgents, GuardAgent, AutoDefense, AegisLLM, SafeFlow | `safeagents/core/src/safety/recent_baselines.py`, `safeagents/core/src/safety/safeflow_research.py` | SafeAgents is undefended. GuardAgent, AutoDefense, and AegisLLM entries are named, lightweight in-repository adapters and are marked as such in run records. | Obtain, configure, and cite the original baseline repositories and their paper configurations; rerun all methods under the same frozen protocol. |
| Benign utility: TCR, FPR, Paired | `paper_experiments.py` | ASB and AgentHarm public companion data can be loaded. FPR is a block-rate field; TCR requires a runtime completion signal. SafeArena and RedCode paired values remain unavailable without a verified mapping. | Connect official task-completion graders and an explicit benign-harmful mapping for every benchmark. A tool-call proxy is debugging only. |
| Jailbreak robustness | `paper_experiments.py` | Runs four labeled prompt-wrapper stress conditions. | Integrate and freeze the original ReNeLLM, GPTFuzz, JailBroken, and MultiLingual attack generators, their seed/configuration, and generated prompts. The local wrappers are not source-method replications. |
| Defense-model sensitivity | `paper_experiments.py`, `safeflow_research.py` | `SAFEFLOW_SEMANTIC_MODEL` selects the defense-side semantic model name and records it. | Configure valid endpoint/credentials for every listed model and archive resolved provider IDs, dates, temperature, retries, and raw records. |
| Pipeline ablation | `paper_experiments.py`, `safeflow_research.py` | Full, no intent annotation, no planning/propagation, no context reconstruction, and no global validation are executable. The planning and propagation switches are regression-tested. | Run on the paper's fixed slice with a working model provider and native benchmark evaluator. |
| Policy and context sensitivity | `paper_experiments.py`, `safeflow_research.py` | Four label schemas, partial propagation, and local/one-hop/full-upstream context are executable. | Freeze label definitions, model/provider, seeds, and official success signals for the reported slice. |
| Prompt-Local vs Cross-Agent analysis | `paper_experiments.py` | Classifies only emitted messages and tool calls; insufficient evidence is `ambiguous`, not guessed. | Ensure every framework logs complete delegation/provenance edges, then archive the judged trace labels and audit protocol. |
| Adaptive attacks and defense-side injection | `paper_experiments.py`, `safeflow_research.py` | Three adaptive wrappers and three injection variants plus control are executable; untrusted context is expressly treated as data by the reconstructor. | Freeze generated attack artifacts and run against native environments with the specified threat model. |
| Confidence intervals, proportion tests, and overhead | `paper_experiments.py` | Computes Wilson intervals, pooled tests, wall-clock duration, semantic-call counts, and reconstructed-node counts from completed raw records. | Use only complete, native-evaluated runs. Statistical output from offline, failed, proxy, or adapter-only runs is not publication evidence. |

## What has been verified locally

```powershell
python -m pytest tests/test_paper_experiments.py tests/test_safeflow_release.py -q
python -m safeagents.core.src.evaluation.paper_experiments all --limit 1 --offline --output-dir build_check/paper_experiments_all_offline
```

The offline run validates all condition branches and deliberately writes
`offline_not_executed` records with null metrics. It is not a benchmark run.

## Current blocker for live runs

The active OpenAI-compatible provider returned HTTP 401 during a one-example
live smoke test. Supply a valid provider credential before running the live
commands. A successful authentication alone is not sufficient for a
paper-level reproduction: the native evaluator and original baseline
requirements above still apply.
