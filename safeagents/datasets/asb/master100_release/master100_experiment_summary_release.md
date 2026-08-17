# SafeFlow Master100 Experiment Summary

## Dataset

- Exported subset: `safeagents/datasets/asb/master100_attacks.json`
- Release directory: `safeagents/datasets/asb/master100_release`
- Source dataset: ASB
- Total attack samples: `100`

## Methods

- `baseline`: original SafeAgents multi-agent workflow without SafeFlow.
- `no_step2`: SafeFlow ablation without step 2 taint propagation / subtask-level taint tracking.
- `full5`: full SafeFlow with all five stages enabled.

## Main Result

| Agent | Count | Baseline ASR | No Step2 ASR | Full5 ASR |
|---|---:|---:|---:|---:|
| academic_search_agent | 43 | 0.6977 | 0.4186 | 0.2093 |
| aerospace_engineer_agent | 4 | 1.0000 | 0.5000 | 0.5000 |
| ecommerce_manager_agent | 37 | 0.6757 | 0.4324 | 0.2973 |
| financial_analyst_agent | 7 | 1.0000 | 0.5714 | 0.4286 |
| legal_consultant_agent | 6 | 1.0000 | 0.3333 | 0.3333 |
| medical_advisor_agent | 3 | 1.0000 | 0.0000 | 0.0000 |
| TOTAL | 100 | 0.7500 | 0.4200 | 0.2700 |

## Key Takeaway

- `baseline = 0.75`
- `no_step2 = 0.42`
- `full5 = 0.27`
- `baseline - full5 = 0.48`
- `no_step2 - full5 = 0.15`
