from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeagents.core.src.evaluation.unified_eval_loader import load_eval_rows
from safeagents.core.src.datasets import DatasetRegistry
from safeagents.core.src.frameworks.base import Team
import safeagents.datasets  # noqa: F401


def ensure_framework_registered(framework: str) -> None:
    module_map = {
        "autogen": "safeagents.core.src.frameworks.autogen.autogen",
        "langgraph": "safeagents.core.src.frameworks.langgraph.langgraph",
        "openai-agents": "safeagents.core.src.frameworks.openai_agents.openai_agents",
    }
    module_name = module_map.get(framework)
    if not module_name:
        raise ValueError(f"Unsupported framework: {framework}")
    __import__(module_name)


def resolve_dataset_handler(task_data: Dict[str, Any]):
    benchmark = str(task_data.get("benchmark") or "").lower()
    if benchmark:
        return DatasetRegistry.get_handler("unified_benchmarks")
    return DatasetRegistry.get_handler("asb")


def create_team_for_task(task_data: Dict[str, Any], framework: str, architecture: str):
    ensure_framework_registered(framework)
    dataset_handler = resolve_dataset_handler(task_data)
    if dataset_handler is None:
        raise ValueError("No dataset handler is registered for the current task.")
    agents = dataset_handler.create_agents(task_data, framework)
    return Team.create(agents=agents, framework=framework, architecture=architecture)


def create_attack_detector(task_data: Dict[str, Any]):
    dataset_handler = resolve_dataset_handler(task_data)
    if dataset_handler is None or not hasattr(dataset_handler, "create_attack_detector"):
        return None
    return dataset_handler.create_attack_detector(task_data)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_execution_result(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"messages": value, "logs": "", "execution_trace": {}, "raw_execution_result": value}
    if value is None:
        return {}
    return {"messages": [], "logs": str(value), "execution_trace": {}, "raw_execution_result": value}


def _is_execution_failure(stage: Any, reason: Any, execution_result: Any) -> bool:
    stage_text = str(stage or "").strip().lower()
    reason_text = str(reason or "").strip().lower()
    if stage_text == "error":
        return True
    if reason_text.startswith("execution failed:"):
        return True
    if stage_text == "execution" and not execution_result:
        return True
    return False


async def run_one(row: Dict[str, Any], framework: str, architecture: str) -> Dict[str, Any]:
    team = create_team_for_task(row, framework=framework, architecture=architecture)
    attack_detector = create_attack_detector(row)
    result = await team.run(task=row["prompt"], verbose=False, attack_detector=attack_detector)
    result_dict = _safe_dict(result)
    execution_result = _normalize_execution_result(result_dict)
    attack_success = int(bool(_safe_dict(execution_result).get("attack_detected")))
    stage = result_dict.get("stage", "completed")
    reason = result_dict.get("reason", "Undefended baseline completed task execution.")
    valid_execution = not _is_execution_failure(stage, reason, execution_result)
    return {
        "sample_id": str(row.get("sample_id") or ""),
        "benchmark": row.get("benchmark"),
        "agent_name": row.get("agent_name"),
        "attack_tool": row.get("attack_tool"),
        "method": "baseline_vanilla",
        "blocked": False,
        "stage": stage,
        "reason": reason,
        "attack_success": attack_success,
        "valid_execution": valid_execution,
        "execution_result": execution_result,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [x for x in rows if bool(x.get("valid_execution", True))]
    asr = round(sum(int(bool(x.get("attack_success", 0))) for x in valid_rows) / len(valid_rows), 4) if valid_rows else None
    return {
        "count": len(rows),
        "valid_count": len(valid_rows),
        "execution_failure_count": len(rows) - len(valid_rows),
        "asr": asr,
    }


async def main_async(args):
    rows = load_eval_rows(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / 'baseline_results.jsonl'
    json_path = args.output_dir / 'baseline_results.json'
    csv_path = args.output_dir / 'baseline_summary.csv'

    outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            coro = run_one(row, args.framework, args.architecture)
            record = await asyncio.wait_for(coro, timeout=args.task_timeout) if args.task_timeout > 0 else await coro
        except Exception as exc:
            record = {
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "agent_name": row.get("agent_name"),
                "attack_tool": row.get("attack_tool"),
                "method": "baseline_vanilla",
                "blocked": False,
                "stage": "error",
                "reason": f"Execution failed: {exc}",
                "attack_success": 0,
                "valid_execution": False,
                "execution_result": {"error": str(exc)},
            }
            errors.append({
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "agent_name": row.get("agent_name"),
                "error": str(exc),
            })
        outputs.append(record)
        print(f"[{idx}/{len(rows)}] {row.get('sample_id')} -> attack_success={record['attack_success']} valid={record['valid_execution']}")

    with jsonl_path.open('w', encoding='utf-8') as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\\n')
    summary = summarize(outputs)
    json_path.write_text(json.dumps({"results": outputs, "errors": errors, "summary": summary}, ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["count", "valid_count", "execution_failure_count", "asr"])
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description='Run 22-sample undefended baseline on a unified benchmark file.')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--framework', default='autogen', choices=['autogen', 'langgraph', 'openai-agents'])
    p.add_argument('--architecture', default='centralized', choices=['centralized', 'decentralized'])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--task-timeout', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()

