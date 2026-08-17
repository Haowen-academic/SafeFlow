"""
SafeAgents dataset handlers.

Handlers are imported here so they auto-register with DatasetRegistry.
"""

from . import agentharm
from . import asb
from . import unified_benchmarks

try:
    from . import safeflow_synthetic
except ImportError:
    safeflow_synthetic = None

__all__ = ['agentharm', 'asb', 'unified_benchmarks', 'safeflow_synthetic']
