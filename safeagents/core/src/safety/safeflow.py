"""
SafeFlow - Information Flow Control for Multi-Agent Security

Based on the SafeFlow architecture described in the research paper, this module
implements taint analysis for detecting and blocking malicious intent propagation
across multi-agent systems.

Core components:
1. IntentTaintAnnotator - Labels user input with potential malicious intent tags
2. TaintPropagationTracker - Tracks how intent labels propagate through subtasks
3. ContextReconstructor - Reconstructs global intent for sub-agents
4. GlobalConsistencyValidator - Validates overall behavior consistency

Usage:
    from safeagents.safeflow import SafeFlow
    
    safeflow = SafeFlow()
    result = await safeflow.run_safe_task(
        task="Delete all files",
        agents=[agent1, agent2],
        framework="openai-agents"
    )
    
    if result['blocked']:
        print(f"Blocked due to: {result['reason']}")
        print(f"Trace: {result['trace_report']}")
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
from pathlib import Path

from ...logger import logger


class TaintCategory(Enum):
    """Categories of potential malicious intent."""
    FILE_DELETION = "file_deletion"
    FILE_MODIFICATION = "file_modification"
    FILE_READ = "file_read"
    NETWORK_ACCESS = "network_access"
    EMAIL_SENDING = "email_sending"
    CODE_EXECUTION = "code_execution"
    SYSTEM_COMMAND = "system_command"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SOCIAL_ENGINEERING = "social_engineering"
    PHISHING = "phishing"
    MALWARE = "malware"
    DENIAL_OF_SERVICE = "denial_of_service"
    UNKNOWN = "unknown"


@dataclass
class TaintLabel:
    """
    Represents a taint label attached to user input or subtask.
    
    Attributes:
        category: The type of malicious intent
        confidence: Confidence score (0-1) in the detection
        keywords: List of keywords that triggered this label
        context: Additional context about where this label applies
    """
    category: TaintCategory
    confidence: float
    keywords: List[str] = field(default_factory=list)
    context: Optional[str] = None

    def __str__(self):
        return f"{self.category.value} (confidence: {self.confidence:.2f})"


@dataclass
class TaintedTask:
    """
    A task with associated taint labels.
    
    Attributes:
        task: The original task string
        taint_labels: List of taint labels attached to this task
        parent_task: Reference to parent task (for subtasks)
        subtasks: List of child subtasks
        agent_assigned: Name of agent assigned to this task
        executed: Whether this task was executed
        result: Execution result
    """
    task: str
    taint_labels: List[TaintLabel] = field(default_factory=list)
    parent_task: Optional['TaintedTask'] = None
    subtasks: List['TaintedTask'] = field(default_factory=list)
    agent_assigned: Optional[str] = None
    executed: bool = False
    result: Optional[Any] = None

    def has_taint(self, category: TaintCategory = None) -> bool:
        """Check if task has any taint labels (optionally of specific category)."""
        if category:
            return any(t.category == category for t in self.taint_labels)
        return len(self.taint_labels) > 0

    def get_max_confidence(self) -> float:
        """Get the highest confidence score among all taint labels."""
        if not self.taint_labels:
            return 0.0
        return max(t.confidence for t in self.taint_labels)


class IntentTaintAnnotator:
    """
    Labels user input with potential malicious intent tags using rule-based detection.
    
    Uses keyword matching and pattern recognition to identify semantic dimensions
    of potential harm without making binary "is malicious" decisions.
    """

    # Keyword patterns for each taint category
    TAINT_PATTERNS: Dict[TaintCategory, List[str]] = {
        TaintCategory.FILE_DELETION: [
            "delete", "remove", "erase", "destroy", "rm ", "rmdir", "del ", "erase",
            "wipe", "purge", "trash", "unlink"
        ],
        TaintCategory.FILE_MODIFICATION: [
            "modify", "change", "update", "edit", "write", "overwrite", "append",
            "create", "make", "generate", "save", "dump"
        ],
        TaintCategory.FILE_READ: [
            "read", "view", "cat", "get content", "show me", "display", "access",
            "retrieve", "fetch", "scan"
        ],
        TaintCategory.NETWORK_ACCESS: [
            "download", "upload", "connect", "http", "url", "website", "web page",
            "api", "request", "curl", "wget", "fetch"
        ],
        TaintCategory.EMAIL_SENDING: [
            "send email", "email", "mail to", "send to", "message", "smtp"
        ],
        TaintCategory.CODE_EXECUTION: [
            "run", "execute", "eval", "python", "bash", "script", "command",
            "compile", "run code", "execute code"
        ],
        TaintCategory.SYSTEM_COMMAND: [
            "terminal", "shell", "cmd", "powershell", "sudo", "chmod", "chown",
            "system(", "os.", "subprocess"
        ],
        TaintCategory.DATA_EXFILTRATION: [
            "send to", "transfer", "copy to", "export", "leak", "exfiltrate",
            "upload to", "share with", "send externally"
        ],
        TaintCategory.PRIVILEGE_ESCALATION: [
            "sudo", "root", "admin", "privilege", "elevate", "su ", "chmod 777"
        ],
        TaintCategory.SOCIAL_ENGINEERING: [
            "pretend", "fake", "impersonate", "trick", "deceive", "phish",
            "spoof", "mislead", "con"
        ],
        TaintCategory.PHISHING: [
            "phish", "fake login", "credentials", "password", "credit card",
            "bank info", "SSN", "social security"
        ],
        TaintCategory.MALWARE: [
            "virus", "malware", "worm", "trojan", "backdoor", "payload",
            "exploit", "zero-day"
        ],
        TaintCategory.DENIAL_OF_SERVICE: [
            "crash", "freeze", "overload", "DoS", "DDoS", "flood", "spam",
            "overwhelm", "shut down"
        ]
    }

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the annotator.
        
        Args:
            confidence_threshold: Minimum confidence score for a label to be included
        """
        self.confidence_threshold = confidence_threshold

    def annotate(self, task: str) -> List[TaintLabel]:
        """
        Annotate a task with taint labels.
        
        Args:
            task: The user input task string
            
        Returns:
            List of TaintLabel objects representing potential malicious intents
        """
        labels = []
        task_lower = task.lower()
        
        for category, keywords in self.TAINT_PATTERNS.items():
            matched_keywords = []
            for keyword in keywords:
                # Handle multi-word patterns
                if ' ' in keyword:
                    if keyword in task_lower:
                        matched_keywords.append(keyword)
                else:
                    # Check for keyword as word (not substring)
                    words = task_lower.split()
                    if keyword in words:
                        matched_keywords.append(keyword)
                    # Also check with common punctuation
                    for word in words:
                        clean_word = word.strip('.,!?;:()[]{}')
                        if clean_word == keyword:
                            matched_keywords.append(keyword)
                            break
            
            if matched_keywords:
                # Calculate confidence based on number of matched keywords
                # and their importance (longer keywords = higher confidence)
                confidence = min(1.0, len(matched_keywords) * 0.3 + 
                               sum(len(k) * 0.02 for k in matched_keywords))
                
                if confidence >= self.confidence_threshold:
                    labels.append(TaintLabel(
                        category=category,
                        confidence=confidence,
                        keywords=matched_keywords,
                        context=task[:100] if len(task) > 100 else task
                    ))
        
        # Sort by confidence (highest first)
        labels.sort(key=lambda x: x.confidence, reverse=True)
        return labels


class TaintPropagationTracker:
    """
    Tracks how taint labels propagate through task decomposition.
    
    Implements rules similar to data flow analysis for tracking intent
    propagation from parent tasks to subtasks.
    """

    def __init__(self):
        self.propagation_history: List[Dict[str, Any]] = []

    def propagate_to_subtasks(
        self,
        parent_task: TaintedTask,
        subtasks: List[str]
    ) -> List[TaintedTask]:
        """
        Propagate taint labels from parent task to its subtasks.
        
        Args:
            parent_task: The parent task with existing taint labels
            subtasks: List of subtask strings to create
            
        Returns:
            List of TaintedTask objects with propagated taint labels
        """
        tainted_subtasks = []
        
        for subtask in subtasks:
            # Create taint labels for subtask based on parent labels
            subtask_labels = self._calculate_propagated_labels(parent_task, subtask)
            
            tainted_subtask = TaintedTask(
                task=subtask,
                taint_labels=subtask_labels,
                parent_task=parent_task
            )
            tainted_subtasks.append(tainted_subtask)
            
            # Record propagation
            self.propagation_history.append({
                'parent_task': parent_task.task[:50],
                'subtask': subtask[:50],
                'propagated_labels': [str(l) for l in subtask_labels],
                'timestamp': asyncio.get_event_loop().time()
            })
            
            parent_task.subtasks.append(tainted_subtask)
        
        return tainted_subtasks

    def _calculate_propagated_labels(
        self,
        parent_task: TaintedTask,
        subtask: str
    ) -> List[TaintLabel]:
        """
        Calculate which taint labels should propagate to a subtask.
        
        Propagation rules:
        1. If subtask mentions keywords from parent's taint labels -> propagate with reduced confidence
        2. If subtask is a direct action related to parent's intent -> propagate with full confidence
        3. If subtask is completely unrelated -> no propagation
        """
        propagated = []
        subtask_lower = subtask.lower()
        
        for parent_label in parent_task.taint_labels:
            # Check if subtask contains any keywords from parent label
            keyword_match = any(k in subtask_lower for k in parent_label.keywords)
            
            # Check if subtask is semantically related to the category
            category_related = self._is_category_related(parent_label.category, subtask)
            
            if keyword_match or category_related:
                # Calculate propagated confidence
                # Higher confidence if both keyword match and semantic relation
                confidence_factor = 0.7 if keyword_match and category_related else 0.5
                propagated_confidence = parent_label.confidence * confidence_factor
                
                propagated.append(TaintLabel(
                    category=parent_label.category,
                    confidence=propagated_confidence,
                    keywords=parent_label.keywords,
                    context=f"Propagated from parent: {parent_label.category.value}"
                ))
        
        return propagated

    def _is_category_related(self, category: TaintCategory, subtask: str) -> bool:
        """Check if a subtask is semantically related to a taint category."""
        subtask_lower = subtask.lower()
        
        # Define semantic indicators for each category
        semantic_indicators = {
            TaintCategory.FILE_DELETION: ["file", "document", "folder", "directory", "path"],
            TaintCategory.FILE_MODIFICATION: ["file", "document", "write", "create", "save"],
            TaintCategory.FILE_READ: ["file", "document", "read", "view", "content"],
            TaintCategory.NETWORK_ACCESS: ["url", "link", "website", "download", "upload"],
            TaintCategory.EMAIL_SENDING: ["email", "send", "message", "mail"],
            TaintCategory.CODE_EXECUTION: ["code", "run", "execute", "script"],
            TaintCategory.SYSTEM_COMMAND: ["command", "shell", "terminal", "execute"],
            TaintCategory.DATA_EXFILTRATION: ["send", "transfer", "copy", "export"],
            TaintCategory.PRIVILEGE_ESCALATION: ["admin", "root", "sudo", "access"],
            TaintCategory.SOCIAL_ENGINEERING: ["pretend", "fake", "impersonate"],
            TaintCategory.PHISHING: ["login", "password", "credentials", "account"],
            TaintCategory.MALWARE: ["virus", "infect", "payload", "exploit"],
            TaintCategory.DENIAL_OF_SERVICE: ["crash", "shutdown", "overload", "flood"]
        }
        
        indicators = semantic_indicators.get(category, [])
        return any(indicator in subtask_lower for indicator in indicators)

    def get_propagation_report(self) -> str:
        """Generate a report of taint propagation history."""
        report = ["Taint Propagation Report:"]
        for i, entry in enumerate(self.propagation_history, 1):
            report.append(f"\n{i}. Parent: {entry['parent_task']}")
            report.append(f"   Subtask: {entry['subtask']}")
            report.append(f"   Propagated Labels: {', '.join(entry['propagated_labels'])}")
        return '\n'.join(report)


class ContextReconstructor:
    """
    Reconstructs global intent summary for sub-agents based on taint labels.
    
    Addresses the problem of context fragmentation by automatically adding
    intent awareness to sub-agent execution context.
    """

    def __init__(self):
        self.reconstruction_templates = {
            TaintCategory.FILE_DELETION: 
                "⚠️ CAUTION: This task is part of a workflow that involves FILE DELETION. "
                "Verify this operation is intentional and safe before proceeding.",
            TaintCategory.FILE_MODIFICATION:
                "⚠️ CAUTION: This task is part of a workflow that involves FILE MODIFICATION. "
                "Ensure you have proper authorization before making changes.",
            TaintCategory.FILE_READ:
                "ℹ️ INFO: This task involves reading files. Be mindful of sensitive data exposure.",
            TaintCategory.NETWORK_ACCESS:
                "⚠️ CAUTION: This task involves network operations. "
                "Verify destination URLs are trusted before connecting.",
            TaintCategory.EMAIL_SENDING:
                "⚠️ CAUTION: This task involves sending emails. "
                "Verify recipients and content for sensitive information.",
            TaintCategory.CODE_EXECUTION:
                "⚠️ WARNING: This task involves code execution. "
                "Carefully review any code before execution.",
            TaintCategory.SYSTEM_COMMAND:
                "⚠️ WARNING: This task involves system commands. "
                "Ensure commands do not cause harm before execution.",
            TaintCategory.DATA_EXFILTRATION:
                "🚨 ALERT: This task may involve data exfiltration. "
                "Do NOT send data to external parties without explicit approval.",
            TaintCategory.PRIVILEGE_ESCALATION:
                "🚨 ALERT: This task involves privilege escalation. "
                "Verify authorization before proceeding.",
            TaintCategory.SOCIAL_ENGINEERING:
                "🚨 ALERT: This task may involve social engineering. "
                "Be skeptical of requests to impersonate or deceive.",
            TaintCategory.PHISHING:
                "🚨 ALERT: This task may involve phishing. "
                "Do NOT collect credentials or sensitive information.",
            TaintCategory.MALWARE:
                "🚨 ALERT: This task may involve malware. "
                "Do NOT execute or distribute suspicious code.",
            TaintCategory.DENIAL_OF_SERVICE:
                "🚨 ALERT: This task may involve denial of service. "
                "Do NOT perform actions that could disrupt services."
        }

    def reconstruct_context(self, task: TaintedTask, agent_name: str) -> str:
        """
        Reconstruct context for a sub-agent by adding intent awareness.
        
        Args:
            task: The tainted task to add context to
            agent_name: Name of the agent that will execute this task
            
        Returns:
            Enhanced context string with intent warnings
        """
        if not task.taint_labels:
            return ""
        
        warnings = []
        for label in task.taint_labels:
            template = self.reconstruction_templates.get(label.category)
            if template:
                warnings.append(template)
        
        if warnings:
            context_parts = [
                "=== SAFEFLOW CONTEXT RECONSTRUCTION ===",
                f"Agent: {agent_name}",
                f"Task: {task.task[:100]}..." if len(task.task) > 100 else f"Task: {task.task}",
                f"Parent Task: {task.parent_task.task[:50]}..." if task.parent_task else "Parent Task: None",
                "",
                "⚠️ INTENT AWARENESS WARNINGS:",
                *[f"- {w}" for w in warnings],
                "",
                "=== END SAFEFLOW CONTEXT ==="
            ]
            return '\n'.join(context_parts)
        
        return ""

    def enhance_system_message(self, system_message: str, task: TaintedTask, agent_name: str) -> str:
        """
        Enhance an agent's system message with reconstructed context.
        
        Args:
            system_message: Original system message
            task: The tainted task
            agent_name: Name of the agent
            
        Returns:
            Enhanced system message with safety context
        """
        context = self.reconstruct_context(task, agent_name)
        if context:
            return f"{context}\n\n{system_message}"
        return system_message


class GlobalConsistencyValidator:
    """
    Validates that the overall behavior of a multi-agent system is consistent
    with the original intent labels.
    
    Checks for:
    1. Unexpected tool calls that don't match original intent
    2. Inconsistent execution paths
    3. Potential data exfiltration patterns
    4. Unexpected combinations of actions
    """

    def __init__(self):
        # Define dangerous action combinations
        self.dangerous_combinations = [
            (TaintCategory.FILE_READ, TaintCategory.EMAIL_SENDING),
            (TaintCategory.FILE_READ, TaintCategory.NETWORK_ACCESS),
            (TaintCategory.CODE_EXECUTION, TaintCategory.SYSTEM_COMMAND),
            (TaintCategory.FILE_READ, TaintCategory.DATA_EXFILTRATION),
            (TaintCategory.SOCIAL_ENGINEERING, TaintCategory.PHISHING),
        ]

    def validate(
        self,
        root_task: TaintedTask,
        execution_trace: Dict[str, Any],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate global consistency of execution.
        
        Args:
            root_task: The original task with taint labels
            execution_trace: Execution trace from the team run
            threshold: Confidence threshold for triggering block
            
        Returns:
            Validation result with status, reason, and recommendations
        """
        violations = []
        warnings = []
        
        # Check for dangerous combinations in executed tasks
        violations.extend(self._check_dangerous_combinations(root_task))
        
        # Check for unexpected tool calls
        tool_violations = self._check_unexpected_tools(root_task, execution_trace)
        violations.extend(tool_violations)
        
        # Check for escalation of intent
        escalation_warnings = self._check_intent_escalation(root_task)
        warnings.extend(escalation_warnings)
        
        # Determine overall status
        if violations:
            return {
                'valid': False,
                'status': 'BLOCKED',
                'violations': violations,
                'warnings': warnings,
                'reason': f"Detected {len(violations)} critical violation(s)",
                'recommendation': "Block execution and review the detected violations"
            }
        elif warnings:
            return {
                'valid': True,
                'status': 'WARNING',
                'violations': [],
                'warnings': warnings,
                'reason': f"Execution allowed but with {len(warnings)} warning(s)",
                'recommendation': "Monitor execution closely"
            }
        else:
            return {
                'valid': True,
                'status': 'SAFE',
                'violations': [],
                'warnings': [],
                'reason': "No security violations detected",
                'recommendation': "Execution can proceed safely"
            }

    def _check_dangerous_combinations(self, root_task: TaintedTask) -> List[str]:
        """Check for dangerous combinations of taint categories."""
        violations = []
        all_labels = self._collect_all_labels(root_task)
        label_categories = {label.category for label in all_labels}
        
        for cat1, cat2 in self.dangerous_combinations:
            if cat1 in label_categories and cat2 in label_categories:
                violations.append(
                    f"Dangerous combination detected: {cat1.value} + {cat2.value}. "
                    f"This pattern may indicate an attack."
                )
        
        return violations

    def _check_unexpected_tools(
        self,
        root_task: TaintedTask,
        execution_trace: Dict[str, Any]
    ) -> List[str]:
        """Check for tool calls that don't match the expected intent."""
        violations = []
        tool_calls = execution_trace.get('tool_calls', [])
        executed_tools = {call['name'] for call in tool_calls}
        
        # Expected tools based on taint categories
        expected_tools_by_category = {
            TaintCategory.FILE_DELETION: {'delete_file', 'remove_file', 'rm'},
            TaintCategory.FILE_MODIFICATION: {'write_file', 'update_file', 'save_file'},
            TaintCategory.FILE_READ: {'read_file', 'view_file', 'cat'},
            TaintCategory.EMAIL_SENDING: {'send_email', 'send_message'},
            TaintCategory.CODE_EXECUTION: {'execute_code', 'run_python', 'run_bash'},
        }
        
        # Get expected tools from root task labels
        expected_tools = set()
        for label in root_task.taint_labels:
            expected_tools.update(expected_tools_by_category.get(label.category, set()))
        
        # Also consider propagated labels
        for subtask in root_task.subtasks:
            for label in subtask.taint_labels:
                expected_tools.update(expected_tools_by_category.get(label.category, set()))
        
        # Check for unexpected tools
        for tool in executed_tools:
            if tool not in expected_tools:
                violations.append(
                    f"Unexpected tool call: '{tool}'. "
                    f"This tool was not anticipated based on the task intent."
                )
        
        return violations

    def _check_intent_escalation(self, root_task: TaintedTask) -> List[str]:
        """Check if subtasks have higher-risk intent than parent."""
        warnings = []
        
        # Define risk levels (higher = more dangerous)
        risk_levels = {
            TaintCategory.FILE_READ: 1,
            TaintCategory.FILE_MODIFICATION: 2,
            TaintCategory.FILE_DELETION: 3,
            TaintCategory.NETWORK_ACCESS: 2,
            TaintCategory.EMAIL_SENDING: 2,
            TaintCategory.CODE_EXECUTION: 3,
            TaintCategory.SYSTEM_COMMAND: 3,
            TaintCategory.DATA_EXFILTRATION: 4,
            TaintCategory.PRIVILEGE_ESCALATION: 4,
            TaintCategory.SOCIAL_ENGINEERING: 4,
            TaintCategory.PHISHING: 4,
            TaintCategory.MALWARE: 5,
            TaintCategory.DENIAL_OF_SERVICE: 5,
        }
        
        parent_max_risk = max(
            (risk_levels.get(l.category, 0) for l in root_task.taint_labels),
            default=0
        )
        
        for subtask in root_task.subtasks:
            subtask_max_risk = max(
                (risk_levels.get(l.category, 0) for l in subtask.taint_labels),
                default=0
            )
            
            if subtask_max_risk > parent_max_risk + 1:
                warnings.append(
                    f"Intent escalation detected: subtask '{subtask.task[:30]}...' "
                    f"has higher risk level ({subtask_max_risk}) than parent ({parent_max_risk})"
                )
        
        return warnings

    def _collect_all_labels(self, task: TaintedTask) -> List[TaintLabel]:
        """Recursively collect all taint labels from task and its subtasks."""
        labels = list(task.taint_labels)
        for subtask in task.subtasks:
            labels.extend(self._collect_all_labels(subtask))
        return labels


class SafeFlow:
    """
    Main SafeFlow orchestrator that integrates all components.
    
    Provides a unified interface for running tasks with information flow control.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SafeFlow with optional configuration.
        
        Args:
            config: Configuration dictionary with thresholds and settings
        """
        self.config = config or {}
        
        # Initialize components
        self.annotator = IntentTaintAnnotator(
            confidence_threshold=self.config.get('confidence_threshold', 0.3)
        )
        self.propagation_tracker = TaintPropagationTracker()
        self.context_reconstructor = ContextReconstructor()
        self.validator = GlobalConsistencyValidator()
        
        # Store execution history
        self.execution_history = []

    async def run_safe_task(
        self,
        task: str,
        team,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a task with SafeFlow protection.
        
        Args:
            task: The user task string
            team: Pre-created Team object from SafeAgents
            verbose: Whether to print detailed logs
            
        Returns:
            Result dictionary with safety status and execution details
        """
        if verbose:
            print("\n" + "="*80)
            print("SAFEFLOW SECURITY ANALYSIS")
            print("="*80 + "\n")
        
        # Step 1: Intent Taint Annotation
        if verbose:
            print("Step 1: Intent Taint Annotation...")
        
        taint_labels = self.annotator.annotate(task)
        root_task = TaintedTask(task=task, taint_labels=taint_labels)
        
        if verbose:
            print(f"  Detected {len(taint_labels)} taint label(s):")
            for label in taint_labels:
                print(f"    - {label}")
        
        # Check if we should block before execution
        if self._should_block_immediately(root_task):
            return {
                'blocked': True,
                'stage': 'pre-execution',
                'reason': "High-risk intent detected before execution",
                'taint_labels': [str(l) for l in taint_labels],
                'trace_report': self._generate_trace_report(root_task, None),
                'execution_result': None
            }
        
        # Step 2: Inject safety context into agent system messages
        if verbose:
            print("\nStep 2: Context Reconstruction...")
        
        self._inject_safety_context(root_task, team)
        
        # Step 3: Execute the task
        if verbose:
            print("\nStep 3: Executing Task...")
        
        try:
            execution_result = await team.run(task=task, verbose=verbose, **kwargs)
        except Exception as e:
            return {
                'blocked': False,
                'stage': 'execution',
                'reason': f"Execution failed: {str(e)}",
                'taint_labels': [str(l) for l in taint_labels],
                'trace_report': self._generate_trace_report(root_task, None),
                'execution_result': None,
                'error': str(e)
            }
        
        # Step 4: Global Consistency Validation
        if verbose:
            print("\nStep 4: Global Consistency Validation...")
        
        validation = self.validator.validate(
            root_task,
            execution_result.get('execution_trace', {}),
            threshold=self.config.get('validation_threshold', 0.7)
        )
        
        if verbose:
            print(f"  Validation Status: {validation['status']}")
            if validation['violations']:
                print("  Violations:")
                for v in validation['violations']:
                    print(f"    - {v}")
            if validation['warnings']:
                print("  Warnings:")
                for w in validation['warnings']:
                    print(f"    - {w}")
        
        # Determine final result
        if validation['status'] == 'BLOCKED':
            return {
                'blocked': True,
                'stage': 'post-execution',
                'reason': validation['reason'],
                'taint_labels': [str(l) for l in taint_labels],
                'trace_report': self._generate_trace_report(root_task, execution_result),
                'execution_result': execution_result,
                'validation': validation
            }
        else:
            return {
                'blocked': False,
                'stage': 'completed',
                'reason': validation['reason'],
                'taint_labels': [str(l) for l in taint_labels],
                'trace_report': self._generate_trace_report(root_task, execution_result),
                'execution_result': execution_result,
                'validation': validation
            }

    def _should_block_immediately(self, task: TaintedTask) -> bool:
        """Check if task should be blocked before execution."""
        # Block if any label has high confidence for severe categories
        severe_categories = {
            TaintCategory.MALWARE,
            TaintCategory.PHISHING,
            TaintCategory.DENIAL_OF_SERVICE,
            TaintCategory.PRIVILEGE_ESCALATION,
            TaintCategory.DATA_EXFILTRATION
        }
        
        for label in task.taint_labels:
            if label.category in severe_categories and label.confidence > 0.8:
                return True
        
        return False

    def _inject_safety_context(self, root_task: TaintedTask, team):
        """Inject safety context into team agents."""
        # This would modify agent system messages with reconstructed context
        # Implementation depends on the specific framework
        pass

    def _generate_trace_report(
        self,
        root_task: TaintedTask,
        execution_result: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a comprehensive trace report."""
        report_parts = []
        
        report_parts.append("="*80)
        report_parts.append("SAFEFLOW EXECUTION TRACE REPORT")
        report_parts.append("="*80)
        
        # Task information
        report_parts.append("\n1. Task Information")
        report_parts.append("-"*40)
        report_parts.append(f"Task: {root_task.task}")
        report_parts.append(f"Taint Labels: {[str(l) for l in root_task.taint_labels]}")
        
        # Subtask information
        if root_task.subtasks:
            report_parts.append("\n2. Subtasks")
            report_parts.append("-"*40)
            for i, subtask in enumerate(root_task.subtasks, 1):
                report_parts.append(f"\nSubtask {i}: {subtask.task}")
                report_parts.append(f"   Labels: {[str(l) for l in subtask.taint_labels]}")
                report_parts.append(f"   Executed: {subtask.executed}")
        
        # Execution trace
        if execution_result:
            report_parts.append("\n3. Execution Details")
            report_parts.append("-"*40)
            
            trace = execution_result.get('execution_trace', {})
            if trace.get('tool_calls'):
                report_parts.append("\nTools Called:")
                for call in trace['tool_calls']:
                    report_parts.append(f"  - {call['name']} (args: {call['args']})")
            
            if trace.get('bash_commands'):
                report_parts.append("\nBash Commands:")
                for cmd in trace['bash_commands']:
                    report_parts.append(f"  - {cmd['command']}")
            
            if execution_result.get('assessment'):
                report_parts.append("\nAssessment Results:")
                for key, value in execution_result['assessment'].items():
                    report_parts.append(f"  - {key}: {value}")
        
        report_parts.append("\n" + "="*80)
        report_parts.append("END OF REPORT")
        report_parts.append("="*80)
        
        return '\n'.join(report_parts)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of SafeFlow statistics."""
        return {
            'total_executions': len(self.execution_history),
            'blocked_executions': sum(1 for e in self.execution_history if e.get('blocked')),
            'propagation_events': len(self.propagation_tracker.propagation_history),
            'avg_taint_labels': sum(
                len(e.get('taint_labels', [])) for e in self.execution_history
            ) / max(len(self.execution_history), 1)
        }
