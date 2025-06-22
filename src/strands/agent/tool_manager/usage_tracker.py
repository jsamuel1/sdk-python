"""Tool usage tracking for learning patterns."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolUsage:
    """Record of tool usage for learning patterns."""
    tool_name: str
    timestamp: str
    user_prompt: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    success: bool
    execution_time_ms: float
    context_messages: List[Dict[str, Any]]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUsage":
        """Create from dictionary."""
        return cls(**data)
    
    def to_searchable_content(self) -> str:
        """Create searchable content for KB storage."""
        parts = [
            f"Tool: {self.tool_name}",
            f"Success: {self.success}",
            f"Execution time: {self.execution_time_ms}ms",
            f"User prompt: {self.user_prompt}",
        ]
        
        # Add input/output summary
        if self.tool_input:
            input_keys = list(self.tool_input.keys())
            parts.append(f"Input parameters: {', '.join(input_keys)}")
        
        if self.success and self.tool_output:
            parts.append("Tool executed successfully")
        elif self.error_message:
            parts.append(f"Error: {self.error_message}")
        
        return "\n".join(parts)


class ToolUsageTracker:
    """Tracks tool usage patterns for learning and improvement."""
    
    def __init__(self, agent=None, kb_namespace: str = "tool_usage"):
        """Initialize usage tracker.
        
        Args:
            agent: Agent instance for KB storage
            kb_namespace: Namespace for storing usage data in KB
        """
        self.agent = agent
        self.kb_namespace = kb_namespace
        self._usage_cache: List[ToolUsage] = []
        self._cache_limit = 100  # Keep recent usage in memory
    
    def record_usage(
        self,
        tool_name: str,
        user_prompt: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        success: bool,
        execution_time_ms: float,
        context_messages: List[Dict[str, Any]],
        error_message: Optional[str] = None,
    ) -> None:
        """Record a tool usage event.
        
        Args:
            tool_name: Name of the tool used
            user_prompt: The user's original prompt
            tool_input: Input parameters passed to the tool
            tool_output: Output returned by the tool
            success: Whether the tool execution succeeded
            execution_time_ms: Execution time in milliseconds
            context_messages: Recent conversation context
            error_message: Error message if execution failed
        """
        usage = ToolUsage(
            tool_name=tool_name,
            timestamp=datetime.now().isoformat(),
            user_prompt=user_prompt,
            tool_input=tool_input,
            tool_output=tool_output,
            success=success,
            execution_time_ms=execution_time_ms,
            context_messages=context_messages[-3:],  # Keep last 3 messages
            error_message=error_message,
        )
        
        # Add to cache
        self._usage_cache.append(usage)
        if len(self._usage_cache) > self._cache_limit:
            self._usage_cache.pop(0)  # Remove oldest
        
        # Store in KB if agent available
        self._store_usage_in_kb(usage)
    
    def _store_usage_in_kb(self, usage: ToolUsage) -> None:
        """Store usage record in knowledge base."""
        if not self.agent:
            return
        
        try:
            # Only store successful usage for learning positive patterns
            if usage.success and hasattr(self.agent, 'tool') and hasattr(self.agent.tool, 'store_in_kb'):
                content = usage.to_searchable_content()
                
                self.agent.tool.store_in_kb(
                    content=content,
                    namespace=self.kb_namespace,
                    metadata={
                        "tool_name": usage.tool_name,
                        "success": usage.success,
                        "execution_time_ms": usage.execution_time_ms,
                        "timestamp": usage.timestamp,
                        "usage_type": "successful_execution",
                    }
                )
                
        except Exception as e:
            logger.warning(f"Failed to store usage for {usage.tool_name}: {e}")
    
    def get_successful_patterns(self, tool_name: Optional[str] = None, limit: int = 10) -> List[ToolUsage]:
        """Get successful usage patterns for learning.
        
        Args:
            tool_name: Optional tool name to filter by
            limit: Maximum number of patterns to return
            
        Returns:
            List of successful tool usage patterns
        """
        if not self.agent or not hasattr(self.agent, 'tool') or not hasattr(self.agent.tool, 'retrieve'):
            # Fallback to cache
            patterns = [u for u in self._usage_cache if u.success]
            if tool_name:
                patterns = [u for u in patterns if u.tool_name == tool_name]
            return patterns[-limit:]
        
        try:
            # Build search query
            query = "successful tool execution"
            if tool_name:
                query = f"Tool: {tool_name} successful execution"
            
            # Retrieve from KB
            results = self.agent.tool.retrieve(
                query=query,
                namespace=self.kb_namespace,
                max_results=limit,
                similarity_threshold=0.5,
            )
            
            # Convert results back to ToolUsage objects
            patterns = []
            for result in results:
                if result.get("metadata", {}).get("success"):
                    # For now, create simplified usage records from KB results
                    # In a full implementation, we'd store the full ToolUsage data
                    usage = ToolUsage(
                        tool_name=result["metadata"]["tool_name"],
                        timestamp=result["metadata"]["timestamp"],
                        user_prompt=result.get("content", "").split("User prompt: ")[-1].split("\n")[0] if "User prompt:" in result.get("content", "") else "",
                        tool_input={},  # Would need to store full data
                        tool_output={},  # Would need to store full data
                        success=True,
                        execution_time_ms=result["metadata"]["execution_time_ms"],
                        context_messages=[],
                    )
                    patterns.append(usage)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Failed to retrieve usage patterns: {e}")
            return []
    
    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get success rate for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Success rate between 0.0 and 1.0
        """
        tool_usages = [u for u in self._usage_cache if u.tool_name == tool_name]
        if not tool_usages:
            return 0.5  # Default neutral rate
        
        successful = sum(1 for u in tool_usages if u.success)
        return successful / len(tool_usages)
    
    def get_average_execution_time(self, tool_name: str) -> float:
        """Get average execution time for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Average execution time in milliseconds
        """
        tool_usages = [u for u in self._usage_cache if u.tool_name == tool_name and u.success]
        if not tool_usages:
            return 0.0
        
        total_time = sum(u.execution_time_ms for u in tool_usages)
        return total_time / len(tool_usages)
    
    def clear_cache(self) -> None:
        """Clear the usage cache."""
        self._usage_cache.clear()
