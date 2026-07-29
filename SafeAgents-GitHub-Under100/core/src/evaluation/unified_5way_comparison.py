from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeagents.core.logger import logger
from safeagents.core.src.evaluation.unified_eval_loader import load_eval_rows
from safeagents.core.src.datasets import DatasetRegistry
from safeagents.core.src.frameworks.base import Team
from safeagents.core.src.frameworks.architectures import Architecture
from safeagents.core.src.safety import SafeFlow, AutoDefenseASB, GuardAgentLite, SafeAgentsLiteBaseline
import safeagents.datasets  # noqa: F401


DEFAULT_DATASET = Path("safeagents/datasets/asb/master100_release/master100.jsonl")
METHODS = ["baseline_vanilla", "safeflow_full5", "safeflow_no_step2", "safeagents_lite", "guardagent_lite", "autodefense_asb"]


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


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    return load_eval_rows(path)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_execution_result(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {
            "messages": value,
            "logs": "",
            "execution_trace": {},
            "raw_execution_result": value,
        }
    if value is None:
        return {}
    return {
        "messages": [],
        "logs": str(value),
        "execution_trace": {},
        "raw_execution_result": value,
    }


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _load_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding=encoding)
    os.replace(tmp_path, path)


def _completed_keys(rows: List[Dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        method = str(row.get("method") or "")
        sample_id = str(row.get("sample_id") or "")
        if method and sample_id:
            keys.add((method, sample_id))
    return keys


def _write_checkpoint(output_dir: Path, outputs: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    try:
        summary = summarize(outputs)
        results_payload = json.dumps({"results": outputs, "errors": errors, "summary": summary}, indent=2, ensure_ascii=False)
        _atomic_write_text(output_dir / "unified_5way_results.json", results_payload)

        fieldnames = ["method", "count", "valid_count", "execution_failure_count", "asr", "baseline_asr", "no_step2_asr", "full5_asr"]
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["methods"]:
            writer.writerow(row)
        _atomic_write_text(output_dir / "unified_5way_summary.csv", buffer.getvalue())
    except Exception as exc:
        logger.warning(f"Checkpoint write failed: {exc}")


def build_method(method: str):
    if method == "safeflow_full5":
        return SafeFlow(config={
            "confidence_threshold": 0.3,
            "validation_threshold": 0.7,
            "propagation_mode": "selective",
            "immediate_block_threshold": 0.85,
            "enable_intent_annotation": True,
            "enable_subtask_planning": True,
            "enable_taint_propagation": True,
            "enable_context_reconstruction": True,
            "enable_tool_guard": True,
            "enable_rule_validation": True,
            "enable_llm_validation": True,
            "enable_trace_explanation": True,
        })
    if method == "safeflow_no_step2":
        return SafeFlow(config={
            "confidence_threshold": 0.3,
            "validation_threshold": 0.7,
            "propagation_mode": "none",
            "immediate_block_threshold": 0.85,
            "enable_intent_annotation": True,
            "enable_subtask_planning": False,
            "enable_taint_propagation": False,
            "enable_context_reconstruction": True,
            "enable_tool_guard": True,
            "enable_rule_validation": True,
            "enable_llm_validation": True,
            "enable_trace_explanation": True,
        })
    if method == "safeagents_lite":
        return SafeAgentsLiteBaseline(config={"risk_threshold": 0.70})
    if method == "guardagent_lite":
        return GuardAgentLite(config={"enable_llm_judge": False})
    if method == "autodefense_asb":
        return AutoDefenseASB(config={"enable_llm_judge": False})
    raise ValueError(method)


async def run_task(row: Dict[str, Any], framework: str, architecture: str, method: str) -> Dict[str, Any]:
    team = create_team_for_task(row, framework=framework, architecture=architecture)
    attack_detector = create_attack_detector(row)
    runner = build_method(method)

    if method.startswith("safeflow_"):
        result = await runner.run_safe_task(
            task=row["prompt"],
            team=team,
            verbose=False,
            attack_detector=attack_detector,
        )
        result_dict = _safe_dict(result)
        execution_result = _normalize_execution_result(result_dict.get("execution_result"))
        attack_success = int(bool(_safe_dict(execution_result).get("attack_detected")))
        blocked = bool(result_dict.get("blocked"))
        stage = result_dict.get("stage")
        reason = result_dict.get("reason")
    else:
        result = await runner.run_safe_task(
            task=row["prompt"],
            team=team,
            base_task=row.get("base_instruction"),
            attack_tool=row.get("attack_tool"),
            verbose=False,
            attack_detector=attack_detector,
        )
        result_dict = _safe_dict(result)
        execution_result = _normalize_execution_result(result_dict.get("execution_result"))
        attack_success = int(bool(result_dict.get("attack_success", 0)))
        blocked = bool(result_dict.get("blocked"))
        stage = result_dict.get("stage")
        reason = result_dict.get("reason")

    valid_execution = not _is_execution_failure(stage, reason, execution_result)

    return {
        "sample_id": str(row.get("sample_id") or ""),
        "benchmark": row.get("benchmark"),
        "agent_name": row.get("agent_name"),
        "attack_tool": row.get("attack_tool"),
        "method": method,
        "blocked": blocked,
        "stage": stage,
        "reason": reason,
        "attack_success": attack_success,
        "valid_execution": valid_execution,
        "baseline_attack_success": int(row.get("baseline_attack_success", 0)),
        "no_step2_attack_success": int(row.get("no_step2_attack_success", 0)),
        "full5_attack_success": int(row.get("full5_attack_success", 0)),
        "execution_result": execution_result,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def ratio(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        valid_items = [x for x in items if bool(x.get("valid_execution", True))]
        if not valid_items:
            return None
        return round(sum(int(bool(x.get(key, 0))) for x in valid_items) / len(valid_items), 4)

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_benchmark: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_benchmark[str(row.get("benchmark") or "unknown")].append(row)

    method_rows = []
    for method in METHODS:
        items = by_method[method]
        valid_count = sum(1 for item in items if bool(item.get("valid_execution", True)))
        method_rows.append({
            "method": method,
            "count": len(items),
            "valid_count": valid_count,
            "execution_failure_count": len(items) - valid_count,
            "asr": ratio(items, "attack_success"),
            "baseline_asr": ratio(items, "baseline_attack_success"),
            "no_step2_asr": ratio(items, "no_step2_attack_success"),
            "full5_asr": ratio(items, "full5_attack_success"),
        })

    benchmark_rows = []
    for benchmark in sorted(by_benchmark):
        items = by_benchmark[benchmark]
        valid_count = sum(1 for item in items if bool(item.get("valid_execution", True)))
        benchmark_rows.append({
            "benchmark": benchmark,
            "count": len(items),
            "valid_count": valid_count,
            "execution_failure_count": len(items) - valid_count,
            "asr": ratio(items, "attack_success"),
        })

    total_valid = sum(1 for item in rows if bool(item.get("valid_execution", True)))
    return {
        "methods": method_rows,
        "benchmarks": benchmark_rows,
        "total": {
            "count": len(rows),
            "valid_count": total_valid,
            "execution_failure_count": len(rows) - total_valid,
            "asr": ratio(rows, "attack_success"),
        },
    }


async def main_async(args):
    rows = load_dataset(args.dataset)
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    selected_methods = METHODS if args.method == "all" else [args.method]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = args.output_dir / "unified_5way_results.jsonl"
    errors_jsonl = args.output_dir / "unified_5way_errors.jsonl"

    if args.append:
        outputs = _load_existing_jsonl(results_jsonl)
        errors = _load_existing_jsonl(errors_jsonl)
    else:
        outputs = []
        errors = []
        if results_jsonl.exists():
            results_jsonl.unlink()
        if errors_jsonl.exists():
            errors_jsonl.unlink()

    done_keys = _completed_keys(outputs)

    for method in selected_methods:
        logger.info(f"Running method={method} on {len(rows)} samples")
        for idx, row in enumerate(rows, start=1):
            sample_id = str(row.get("sample_id") or "")
            if args.skip_completed and (method, sample_id) in done_keys:
                continue
            try:
                coro = run_task(row, args.framework, args.architecture, method)
                if args.task_timeout > 0:
                    record = await asyncio.wait_for(coro, timeout=args.task_timeout)
                else:
                    record = await coro
                outputs.append(record)
                done_keys.add((method, sample_id))
                _append_jsonl(results_jsonl, record)
            except Exception as exc:
                error_record = {
                    "sample_id": row.get("sample_id"),
                    "benchmark": row.get("benchmark"),
                    "agent_name": row.get("agent_name"),
                    "method": method,
                    "error": str(exc),
                }
                errors.append(error_record)
                _append_jsonl(errors_jsonl, error_record)
                record = {
                    "sample_id": row.get("sample_id"),
                    "benchmark": row.get("benchmark"),
                    "agent_name": row.get("agent_name"),
                    "attack_tool": row.get("attack_tool"),
                    "method": method,
                    "blocked": False,
                    "stage": "error",
                    "reason": f"Execution failed: {exc}",
                    "attack_success": 0,
                    "valid_execution": False,
                    "baseline_attack_success": int(row.get("baseline_attack_success", 0)),
                    "no_step2_attack_success": int(row.get("no_step2_attack_success", 0)),
                    "full5_attack_success": int(row.get("full5_attack_success", 0)),
                    "execution_result": {"error": str(exc)},
                }
                outputs.append(record)
                done_keys.add((method, sample_id))
                _append_jsonl(results_jsonl, record)
                logger.warning(f"[{method}] sample {idx}/{len(rows)} failed: {row.get('sample_id')} :: {exc}")
            _write_checkpoint(args.output_dir, outputs, errors)

    summary = summarize(outputs)
    _write_checkpoint(args.output_dir, outputs, errors)
    print(json.dumps({"summary": summary, "error_count": len(errors)}, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Run the unified 5-way benchmark comparison.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--framework", default="autogen", choices=["autogen", "langgraph", "openai-agents"])
    p.add_argument("--architecture", default="centralized", choices=["centralized", "decentralized"])
    p.add_argument("--method", default="all", choices=METHODS + ["all"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("logs/unified_5way"))
    p.add_argument("--append", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--task-timeout", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

