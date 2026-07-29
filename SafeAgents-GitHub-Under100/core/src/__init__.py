"""
SafeAgents Core - Framework for building safe multi-agent systems.
"""

from .models import (
    Agent,
    AgentConfig,
    Task,
    Tool,
    tool,
    Prompt,
    DesignChoices,
    AutonomyLevel,
    PlanningStrategy,
)

from .frameworks import (
    Team,
    TeamRegistry,
    register_framework,
    Framework,
    Architecture,
    TeamAutogen,
    TeamLanggraph,
    TeamOpenAIAgents,
)

from .evaluation import (
    Assessment,
    aria,
    dharma,
    ARIA_PROMPT,
    DHARMA_PROMPT,
)

from .datasets import (
    Dataset,
    DatasetRegistry,
)

from .clients import (
    get_llm_provider,
    get_llm_config,
    get_openai_client_sync,
    get_openai_client_async,
    get_openai_compatible_config,
    get_azure_config,
    get_azure_openai_client_sync,
    get_azure_openai_client_async,
    Model,
    ModelConfig,
)

from .config import (
    EnvironmentSetup,
)

from .safety import (
    Mitigation,
    SafeFlow,
    IntentTaintAnnotator,
    TaintPropagationTracker,
    ContextReconstructor,
    GlobalConsistencyValidator,
    TaintLabel,
    TaintedTask,
    TaintCategory,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "Task",
    "Tool",
    "tool",
    "Prompt",
    "DesignChoices",
    "AutonomyLevel",
    "PlanningStrategy",
    "Team",
    "TeamRegistry",
    "register_framework",
    "Framework",
    "Architecture",
    "TeamAutogen",
    "TeamLanggraph",
    "TeamOpenAIAgents",
    "Assessment",
    "aria",
    "dharma",
    "ARIA_PROMPT",
    "DHARMA_PROMPT",
    "Dataset",
    "DatasetRegistry",
    "get_llm_provider",
    "get_llm_config",
    "get_openai_client_sync",
    "get_openai_client_async",
    "get_openai_compatible_config",
    "get_azure_config",
    "get_azure_openai_client_sync",
    "get_azure_openai_client_async",
    "Model",
    "ModelConfig",
    "EnvironmentSetup",
    "Mitigation",
    "SafeFlow",
    "IntentTaintAnnotator",
    "TaintPropagationTracker",
    "ContextReconstructor",
    "GlobalConsistencyValidator",
    "TaintLabel",
    "TaintedTask",
    "TaintCategory",
]

try:
    import safeagents.datasets  # noqa: F401
except ImportError:
    pass
