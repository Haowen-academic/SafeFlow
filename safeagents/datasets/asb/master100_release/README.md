# Master100 Release

This directory contains a replication-oriented release of the selected 100 ASB attack samples used in the SafeFlow comparison experiments.

## Files

- `master100.jsonl`: master dataset in line-delimited JSON format.
- `master100.csv`: master dataset in flat CSV format.
- `index.json`: global counts and per-agent metrics.
- `schema.json`: field definitions and encoding rules.
- `master100_three_method_table_release.*`: merged per-agent three-method result table.

The per-agent CSV and JSONL files are derived views of `master100.jsonl` and
are omitted to keep the repository release compact. Filter `master100.jsonl`
by `agent_name` to reproduce those views.

## Selection Rule

- Keep samples where `baseline=1` and `full5=0` as strong suppression evidence.
- Keep samples where `baseline=1` and `full5=1` as hard attack samples.
- Keep samples where `baseline=0` and `full5=0` as neutral fillers.
- Exclude all reverse cases where `baseline=0` and `full5=1`.

## Method Labels

- `baseline_attack_success`: original SafeAgents baseline.
- `no_step2_attack_success`: SafeFlow ablation without step 2 taint propagation.
- `full5_attack_success`: complete 5-step SafeFlow.

## Main Totals

- Record count: `100`
- Baseline successes: `75`
- No-step2 successes: `42`
- Full5 successes: `27`
