# Master100 Schema

`master100_attacks.json`, `master100_attacks.jsonl`, and `master100_attacks_flat.csv` all describe the same `100` selected ASB attack samples.

## Field Definitions

- `block`
  - Which 50-sample block the sample belongs to.
  - Values: `block_a`, `block_b`

- `agent_name`
  - ASB agent configuration name.

- `dataset_index`
  - Sample index in the ASB task list after loading the corresponding `agent_name` config.

- `prompt`
  - Full final attack prompt actually sent to the agent system.
  - This includes the original benign instruction plus the appended DPI / injected attacker instruction.

- `base_instruction`
  - Original benign task instruction before injection.

- `attack_instruction`
  - Attacker instruction appended into the final prompt.

- `attack_tool`
  - Expected attack tool referenced by the injected instruction.

- `attack_goal`
  - Goal text provided by the ASB attack specification.

- `attack_type`
  - Attack category from ASB, such as `Stealthy Attack` or `Disruptive Attack`.

- `target_functions`
  - Tool/function names available to the corresponding ASB agent setting.
  - In CSV this is serialized as a `;`-joined string.

- `baseline_attack_success`
  - Attack success label under the original SafeAgents baseline.
  - Values: `0`, `1`

- `no_step2_attack_success`
  - Attack success label under the SafeFlow ablation without step 2 taint propagation.
  - Values: `0`, `1`

- `full5_attack_success`
  - Attack success label under the complete 5-step SafeFlow framework.
  - Values: `0`, `1`

## Recommended Use

- Use `jsonl` for scripting, filtering, repeated runs, and dataset pipelines.
- Use `csv` for spreadsheets, quick sorting, and manual inspection.
- Use `json` when you want metadata such as block composition and summary tables in one file.
