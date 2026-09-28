"""Multiagent capabilities for Strands Agents.

This module provides support for multiagent systems, including agent-to-agent (A2A)
communication protocols and coordination mechanisms.

Submodules:
    a2a: Implementation of the Agent-to-Agent (A2A) protocol, which enables
         standardized communication between agents.
"""

from .base import MultiAgentBase, MultiAgentResult, Status
from .graph import EdgeCondition, EdgeConditionWithContext, GraphBuilder, GraphResult
from .swarm import Handoff, HandoffContext, HandoffStrategy, Swarm, SwarmResult

__all__ = [
    "EdgeCondition",
    "EdgeConditionWithContext",
    "GraphBuilder",
    "GraphResult",
    "Handoff",
    "HandoffContext",
    "HandoffStrategy",
    "MultiAgentBase",
    "MultiAgentResult",
    "Status",
    "Swarm",
    "SwarmResult",
]
