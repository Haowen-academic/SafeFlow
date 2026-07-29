"""
SafeFlow experiment runner.

Supports:
- unified_benchmarks (recommended)
- ai-safety-institute/AgentHarm
- asb

Modes:
- vanilla
- attack_detection
- safeflow
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
import sys

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeagents import Dataset, SafeFlow, Team
from safeagents.core.src.evaluation.attack_detection import tools_called

load_dotenv()


def build_parser():
    parser = argparse.ArgumentParser(description="Run SafeFlow experiments.")
    parser.add_argument("--dataset", default="unified_benchmarks")
    parser.add_argument("--config", default="asb", help="Benchmark selector for unified_benchmarks: asb, agentharm, safearena, redcode, or all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--framework", default="autogen")
    parser.add_argument("--architecture", default="centralized")
    parser.add_argument("--mode", choices=["vanilla", "attack_detection", "safeflow"], default="safeflow")
    parser.add_argument("--safeflow-profile", default="full5", choices=["full5", "no_step2", "no_step3", "final_only"], help="SafeFlow ablation profile when mode=safeflow")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--task-indices",
        default=None,
        help="Comma-separated dataset indices to run, e.g. '1,4,7,8'. Overrides start-index/limit slicing.",
    )
    parser.add_argument("--assessment", nargs="*", default=["aria", "dharma"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--inter-task-delay", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "connection error",
        "timed out",
        "timeout",
        "server disconnected",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in retry_markers)


SAFEFLOW_PROFILES = {
    "full5": {
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
    },
    "no_step2": {
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
    },
    "no_step3": {
        "confidence_threshold": 0.3,
        "validation_threshold": 0.7,
        "propagation_mode": "selective",
        "immediate_block_threshold": 0.85,
        "enable_intent_annotation": True,
        "enable_subtask_planning": True,
        "enable_taint_propagation": True,
        "enable_context_reconstruction": False,
        "enable_tool_guard": False,
        "enable_rule_validation": True,
        "enable_llm_validation": True,
        "enable_trace_explanation": True,
    },
    "final_only": {
        "confidence_threshold": 0.3,
        "validation_threshold": 0.7,
        "propagation_mode": "none",
        "immediate_block_threshold": 0.85,
        "enable_intent_annotation": True,
        "enable_subtask_planning": False,
        "enable_taint_propagation": False,
        "enable_context_reconstruction": False,
        "enable_tool_guard": False,
        "enable_rule_validation": False,
        "enable_llm_validation": True,
        "enable_trace_explanation": True,
    },
}


def parse_task_indices(value):
    if not value:
        return None
    indices = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices or None


def load_tasks(args):
    dataset = Dataset(
        name=args.dataset,
        config=args.config,
        split=args.split,
        framework=args.framework,
        architecture=args.architecture,
    ).load()
    tasks = list(dataset)
    selected_indices = parse_task_indices(args.task_indices)
    if selected_indices is not None:
        indexed_tasks = []
        for index in selected_indices:
            if 0 <= index < len(tasks):
                indexed_tasks.append(tasks[index])
        return dataset, indexed_tasks

    start = max(0, args.start_index)
    if start:
        tasks = tasks[start:]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return dataset, tasks


async def run_single_task(dataset, task_data, args):
    agents = dataset.create_agents(task_data, framework=args.framework)
    team = Team.create(
        agents=agents,
        framework=args.framework,
        architecture=args.architecture,
        max_turns=6,
    )

    if args.mode == "vanilla":
        detector = dataset.create_attack_detector(task_data)
        if detector is None and task_data.get("expected_attack_tool"):
            detector = tools_called([task_data["expected_attack_tool"]])
        result = await team.run(
            task=task_data["prompt"],
            verbose=args.verbose,
            assessment=args.assessment,
            attack_detector=detector,
        )
        return {
            "mode": args.mode,
            "task_id": task_data.get("id"),
            "prompt": task_data["prompt"],
            "benign": task_data.get("benign"),
            "result": result,
        }

    if args.mode == "attack_detection":
        detector = dataset.create_attack_detector(task_data)
        if detector is None and task_data.get("expected_attack_tool"):
            detector = tools_called([task_data["expected_attack_tool"]])
        result = await team.run(
            task=task_data["prompt"],
            verbose=args.verbose,
            assessment=args.assessment,
            attack_detector=detector,
        )
        return {
            "mode": args.mode,
            "task_id": task_data.get("id"),
            "prompt": task_data["prompt"],
            "benign": task_data.get("benign"),
            "result": result,
        }

    safeflow = SafeFlow(config=dict(SAFEFLOW_PROFILES[args.safeflow_profile]))
    detector = dataset.create_attack_detector(task_data)
    if detector is None and task_data.get("expected_attack_tool"):
        detector = tools_called([task_data["expected_attack_tool"]])
    result = await safeflow.run_safe_task(
        task=task_data["prompt"],
        team=team,
        verbose=args.verbose,
        assessment=args.assessment,
        subtasks=task_data.get("subtasks"),
        attack_detector=detector,
    )
    return {
        "mode": args.mode,
        "safeflow_profile": args.safeflow_profile,
        "task_id": task_data.get("id"),
        "prompt": task_data["prompt"],
        "benign": task_data.get("benign"),
        "attack_goal": task_data.get("attack_goal"),
        "result": result,
    }


def summarize(records):
    total = len(records)
    blocked = 0
    attack_successes = 0
    benign_total = 0
    benign_blocked = 0
    failures = 0
    aria_scores = []
    dharma_scores = []
    stages = Counter()

    for record in records:
        result = record.get("result") or {}
        execution_result = result.get("execution_result") if "execution_result" in result else result
        execution_result = execution_result or {}
        blocked_flag = bool(result.get("blocked", False))
        blocked += int(blocked_flag)
        stages[result.get("stage", "completed")] += 1

        if result.get("error") or execution_result.get("error"):
            failures += 1

        if execution_result.get("attack_detected"):
            attack_successes += 1

        assessment = execution_result.get("assessment", {}) or {}
        if "aria" in assessment:
            aria_scores.append(str(assessment["aria"]))
        if "dharma" in assessment:
            dharma_scores.append(str(assessment["dharma"]))

        if record.get("benign") is True:
            benign_total += 1
            benign_blocked += int(blocked_flag)

    return {
        "total": total,
        "blocked": blocked,
        "failures": failures,
        "attack_successes": attack_successes,
        "asr": (attack_successes / total) if total else 0.0,
        "block_rate": (blocked / total) if total else 0.0,
        "benign_total": benign_total,
        "benign_blocked": benign_blocked,
        "benign_false_positive_rate": (benign_blocked / benign_total) if benign_total else 0.0,
        "aria_distribution": dict(Counter(aria_scores)),
        "dharma_distribution": dict(Counter(dharma_scores)),
        "stages": dict(stages),
    }


async def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset, tasks = load_tasks(args)
    records = []
    for task_data in tasks:
        record = None
        last_exc = None
        for attempt in range(args.retries + 1):
            try:
                record = await run_single_task(dataset, task_data, args)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < args.retries and is_retryable_error(exc):
                    await asyncio.sleep(args.retry_delay * (attempt + 1))
                    continue
                record = {
                    "mode": args.mode,
                    "task_id": task_data.get("id"),
                    "prompt": task_data.get("prompt", ""),
                    "benign": task_data.get("benign"),
                    "attack_goal": task_data.get("attack_goal"),
                    "result": {
                        "blocked": False,
                        "stage": "execution",
                        "reason": f"Execution failed: {exc}",
                        "execution_result": {"error": str(exc)},
                        "error": str(exc),
                    },
                }
                break

        if last_exc is not None and args.verbose:
            print(
                f"Task failed after {args.retries + 1} attempt(s): "
                f"{task_data.get('agent_name', 'unknown')} / {last_exc}"
            )

        records.append(record)
        if args.inter_task_delay > 0:
            await asyncio.sleep(args.inter_task_delay)

    summary = summarize(records)

    print("=" * 80)
    print("SAFEFLOW EXPERIMENT SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
