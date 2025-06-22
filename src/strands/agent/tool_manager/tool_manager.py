"""Base tool manager interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ToolManager(ABC):
    """Base interface for tool managers that select tools based on context."""

    @abstractmethod
    def select_tools(
        self,
        prompt: str,
        available_tools: Dict[str, Any],
        context_messages: List[Dict[str, Any]],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select tools based on the given context.
        
        Args:
            prompt: The current user prompt
            available_tools: All available tools
            context_messages: Recent conversation messages
            agent_context: Agent configuration context
            
        Returns:
            Tool configuration with selected tools and choice strategy
        """
        pass

    def set_agent(self, agent: Any) -> None:
        """Set agent reference for managers that need it.
        
        Args:
            agent: The agent instance
        """
        pass
