"""Static tool manager that returns all available tools."""

from typing import Any, Dict, List

from .tool_manager import ToolManager


class StaticToolManager(ToolManager):
    """Tool manager that returns all available tools (current behavior)."""

    def select_tools(
        self,
        prompt: str,
        available_tools: Dict[str, Any],
        context_messages: List[Dict[str, Any]],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return all available tools.
        
        Args:
            prompt: The current user prompt (unused)
            available_tools: All available tools
            context_messages: Recent conversation messages (unused)
            agent_context: Agent configuration context (unused)
            
        Returns:
            Tool configuration with all available tools
        """
        # Convert available_tools dict to list of tool specs
        tools = list(available_tools.values())
        
        return {
            "tools": tools,
            "toolChoice": {"auto": {}},
        }
