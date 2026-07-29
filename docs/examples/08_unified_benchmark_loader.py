"""Unified benchmark loader example.

Loads the local JSONL-normalized benchmark interface used by all framework comparisons.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeagents import Dataset


def preview(benchmark: str, limit: int = 2):
    dataset = Dataset(
        name="unified_benchmarks",
        config=benchmark,
        split="all",
        indices=range(limit),
        framework="autogen",
        architecture="centralized",
    ).load()

    print(f"Benchmark={benchmark} | count={len(dataset.dataset)}")
    for idx, row in enumerate(dataset):
        print(f"[{idx}] sample_id={row.get('sample_id')} attack_type={row.get('attack_type')}")
        print(f"     prompt={row.get('prompt', '')[:180]}")
        print(f"     base_instruction={row.get('base_instruction', '')[:120]}")
        print(f"     attack_instruction={row.get('attack_instruction', '')[:120]}")
        print(f"     target_functions={row.get('target_functions')}\n")


if __name__ == "__main__":
    for benchmark_name in ["asb", "agentharm", "safearena", "redcode"]:
        preview(benchmark_name)
