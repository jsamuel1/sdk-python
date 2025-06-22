"""Retrieval-based tool manager using CRUDL categorization and KB storage."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .metadata_extractor import MetadataExtractor
from .tool_manager import ToolManager
from .tool_metadata import CRUDLOperation, ToolMetadata

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval-based tool selection."""
    max_tools: int = 8
    similarity_threshold: float = 0.6
    kb_namespace: str = "agent_tools"
    usage_boost_factor: float = 0.2  # How much to boost tools with good usage history


class RetrievalToolManager(ToolManager):
    """Tool manager that uses CRUDL categorization and KB retrieval for smart tool selection."""
    
    def __init__(self, config: RetrievalConfig = None):
        """Initialize retrieval tool manager.
        
        Args:
            config: Configuration for tool selection
        """
        self.config = config or RetrievalConfig()
        self.agent = None
        self.metadata_extractor = None
        self.usage_tracker = None  # Will be initialized when agent is set
        self._tool_metadata_cache: Dict[str, ToolMetadata] = {}
    
    def set_agent(self, agent) -> None:
        """Set agent reference and initialize components."""
        self.agent = agent
        self.metadata_extractor = MetadataExtractor(agent)
    
    def select_tools(
        self,
        prompt: str,
        available_tools: Dict[str, Any],
        context_messages: List[Dict[str, Any]],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select tools using CRUDL categorization and semantic retrieval.
        
        Args:
            prompt: The current user prompt
            available_tools: All available tools
            context_messages: Recent conversation messages
            agent_context: Agent configuration context
            
        Returns:
            Tool configuration with selected tools
        """
        if not self.agent:
            logger.warning("Agent not set, falling back to all tools")
            return {"tools": list(available_tools.values()), "toolChoice": {"auto": {}}}
        
        try:
            # Step 1: Index tools with metadata if not already done
            self._index_tools(available_tools)
            
            # Step 2: Build search query from prompt
            search_query = self._build_search_query(prompt, context_messages)
            
            # Step 3: Retrieve relevant tools using KB
            selected_tools = self._retrieve_tools(search_query, available_tools)
            
            # Step 4: Enhance selection with KB-based usage patterns
            enhanced_tools = self._enhance_with_kb_usage_patterns(selected_tools, prompt, available_tools)
            
            # Step 5: Return tool configuration
            return {
                "tools": enhanced_tools,
                "toolChoice": {"auto": {}},
            }
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            # Fallback to all tools
            return {"tools": list(available_tools.values()), "toolChoice": {"auto": {}}}
    
    def record_tool_execution(
        self,
        tool_name: str,
        user_prompt: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        success: bool,
        execution_time_ms: float,
        context_messages: List[Dict[str, Any]],
        error_message: str = None,
    ) -> None:
        """Record tool execution for learning patterns.
        
        Args:
            tool_name: Name of the executed tool
            user_prompt: Original user prompt
            tool_input: Input passed to the tool
            tool_output: Output from the tool
            success: Whether execution succeeded
            execution_time_ms: Execution time in milliseconds
            context_messages: Conversation context
            error_message: Error message if failed
        """
        if self.usage_tracker:
            self.usage_tracker.record_usage(
                tool_name=tool_name,
                user_prompt=user_prompt,
                tool_input=tool_input,
                tool_output=tool_output,
                success=success,
                execution_time_ms=execution_time_ms,
                context_messages=context_messages,
                error_message=error_message,
            )
    
    def _index_tools(self, available_tools: Dict[str, Any]) -> None:
        """Index tools with metadata in the knowledge base."""
        for tool_name, tool_spec in available_tools.items():
            if tool_name not in self._tool_metadata_cache:
                # Extract metadata using LLM
                metadata = self.metadata_extractor.extract_metadata(tool_name, tool_spec)
                self._tool_metadata_cache[tool_name] = metadata
                
                # Store in KB for retrieval
                self._store_tool_metadata(metadata)
    
    def _store_tool_metadata(self, metadata: ToolMetadata) -> None:
        """Store tool metadata in the knowledge base."""
        try:
            # Create searchable content for the tool
            content = self._create_searchable_content(metadata)
            
            # Store in KB using agent's tool
            if hasattr(self.agent, 'tool') and hasattr(self.agent.tool, 'store_in_kb'):
                self.agent.tool.store_in_kb(
                    content=content,
                    namespace=self.config.kb_namespace,
                    metadata={
                        "tool_name": metadata.name,
                        "category": metadata.category,
                        "crudl_operations": [op.value for op in metadata.crudl_operations],
                        "complexity": metadata.complexity,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to store metadata for {metadata.name}: {e}")
    
    def _create_searchable_content(self, metadata: ToolMetadata) -> str:
        """Create searchable content from tool metadata."""
        parts = [
            f"Tool: {metadata.name}",
            f"Category: {metadata.category}",
            f"Operations: {', '.join(op.value for op in metadata.crudl_operations)}",
            f"Complexity: {metadata.complexity}",
            f"Keywords: {', '.join(metadata.keywords)}",
            f"Use cases: {'; '.join(metadata.use_cases)}",
        ]
        
        if metadata.description:
            parts.append(f"Description: {metadata.description}")
        
        return "\n".join(parts)
    
    def _build_search_query(self, prompt: str, context_messages: List[Dict[str, Any]]) -> str:
        """Build search query from user prompt and context."""
        # Start with the current prompt
        query_parts = [prompt]
        
        # Add context from recent messages
        for message in context_messages[-3:]:  # Last 3 messages for context
            if message.get("role") == "user":
                for block in message.get("content", []):
                    if text := block.get("text"):
                        query_parts.append(text)
        
        # Combine and clean up
        query = " ".join(query_parts)
        return query[:500]  # Limit query length
    
    def _retrieve_tools(self, search_query: str, available_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant tools using KB search."""
        try:
            # Use agent's KB retrieval if available
            if hasattr(self.agent, 'tool') and hasattr(self.agent.tool, 'retrieve'):
                results = self.agent.tool.retrieve(
                    query=search_query,
                    namespace=self.config.kb_namespace,
                    max_results=self.config.max_tools,
                    similarity_threshold=self.config.similarity_threshold,
                )
                
                # Extract tool names from results
                selected_tool_names = set()
                for result in results:
                    if "tool_name" in result.get("metadata", {}):
                        selected_tool_names.add(result["metadata"]["tool_name"])
                
                # Return corresponding tool specs
                selected_tools = []
                for tool_name in selected_tool_names:
                    if tool_name in available_tools:
                        selected_tools.append(available_tools[tool_name])
                
                # If we got results, return them
                if selected_tools:
                    return selected_tools
            
        except Exception as e:
            logger.warning(f"KB retrieval failed: {e}")
        
        # Fallback: use CRUDL-based selection
        return self._fallback_crudl_selection(search_query, available_tools)
    
    def _fallback_crudl_selection(self, search_query: str, available_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback tool selection using CRUDL operations."""
        # Analyze query for CRUDL operations
        query_lower = search_query.lower()
        needed_operations = set()
        
        # Simple keyword matching for CRUDL operations
        if any(word in query_lower for word in ["create", "make", "generate", "build", "new"]):
            needed_operations.add(CRUDLOperation.CREATE)
        if any(word in query_lower for word in ["read", "get", "fetch", "show", "display", "find"]):
            needed_operations.add(CRUDLOperation.READ)
        if any(word in query_lower for word in ["update", "modify", "edit", "change"]):
            needed_operations.add(CRUDLOperation.UPDATE)
        if any(word in query_lower for word in ["delete", "remove", "clear"]):
            needed_operations.add(CRUDLOperation.DELETE)
        if any(word in query_lower for word in ["list", "search", "query", "all"]):
            needed_operations.add(CRUDLOperation.LIST)
        
        # If no operations detected, default to READ
        if not needed_operations:
            needed_operations.add(CRUDLOperation.READ)
        
        # Select tools that match needed operations
        selected_tools = []
        for tool_name, tool_spec in available_tools.items():
            if tool_name in self._tool_metadata_cache:
                metadata = self._tool_metadata_cache[tool_name]
                # Check if tool supports any needed operations
                if any(op in metadata.crudl_operations for op in needed_operations):
                    selected_tools.append(tool_spec)
                    if len(selected_tools) >= self.config.max_tools:
                        break
        
        # If still no tools selected, return a few random ones
        if not selected_tools:
            selected_tools = list(available_tools.values())[:self.config.max_tools]
        
        return selected_tools
    
    def _enhance_with_kb_usage_patterns(
        self, 
        selected_tools: List[Dict[str, Any]], 
        prompt: str,
        available_tools: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enhance tool selection using KB-based historical usage patterns."""
        if not self.usage_tracker or not self.agent:
            return selected_tools
        
        try:
            # Get tool names from selected tools
            selected_tool_names = self._extract_tool_names(selected_tools)
            
            # Single KB call that combines query and statistical analysis
            tool_scores = self._query_and_analyze_usage_patterns(prompt, available_tools, selected_tool_names)
            
            # Select top tools based on enhanced scores
            enhanced_tools = self._select_top_scored_tools(tool_scores, available_tools)
            
            return enhanced_tools if enhanced_tools else selected_tools
            
        except Exception as e:
            logger.warning(f"Failed to enhance with KB usage patterns: {e}")
            return selected_tools
    
    def _query_and_analyze_usage_patterns(
        self, 
        prompt: str, 
        available_tools: Dict[str, Any], 
        selected_tool_names: set
    ) -> Dict[str, float]:
        """Single KB call that queries usage patterns and calculates enhanced scores."""
        # Initialize base scores
        tool_scores = {}
        for tool_name in available_tools.keys():
            base_score = 1.0 if tool_name in selected_tool_names else 0.0
            tool_scores[tool_name] = base_score
        
        if not hasattr(self.agent, 'tool') or not hasattr(self.agent.tool, 'retrieve'):
            return tool_scores
        
        try:
            # Build semantic query for similar successful tool usage
            usage_query = f"successful tool execution similar to: {prompt}"
            
            # Single KB retrieval call
            usage_patterns = self.agent.tool.retrieve(
                query=usage_query,
                namespace=f"{self.config.kb_namespace}_usage",
                max_results=20,  # Get more results for better statistics
                similarity_threshold=0.4,  # Lower threshold for usage patterns
            )
            
            if not usage_patterns:
                return tool_scores
            
            # Analyze patterns and calculate scores in one pass
            tool_usage_stats = {}
            
            # First pass: collect raw statistics
            for pattern in usage_patterns:
                metadata = pattern.get('metadata', {})
                tool_name = metadata.get('tool_name')
                
                if not tool_name or tool_name not in available_tools:
                    continue
                
                if tool_name not in tool_usage_stats:
                    tool_usage_stats[tool_name] = {
                        'usage_count': 0,
                        'success_count': 0,
                        'total_execution_time': 0.0,
                        'similarities': [],
                        'timestamps': [],
                    }
                
                stats = tool_usage_stats[tool_name]
                stats['usage_count'] += 1
                
                # Track success
                if metadata.get('success', False):
                    stats['success_count'] += 1
                
                # Track execution time
                exec_time = metadata.get('execution_time_ms', 0)
                stats['total_execution_time'] += exec_time
                
                # Track similarity score (from KB retrieval)
                similarity = pattern.get('similarity_score', 0.5)
                stats['similarities'].append(similarity)
                
                # Track timestamp for recency
                timestamp = metadata.get('timestamp', '')
                if timestamp:
                    stats['timestamps'].append(timestamp)
            
            # Second pass: calculate enhanced scores directly
            for tool_name, raw_stats in tool_usage_stats.items():
                usage_count = raw_stats['usage_count']
                if usage_count == 0:
                    continue
                
                # Calculate metrics and apply boosts in one step
                success_rate = raw_stats['success_count'] / usage_count
                avg_similarity = sum(raw_stats['similarities']) / len(raw_stats['similarities']) if raw_stats['similarities'] else 0.0
                recency_score = self._calculate_recency_score(raw_stats['timestamps'])
                
                # Apply all boosts
                success_boost = success_rate * self.config.usage_boost_factor
                frequency_boost = min(usage_count / 10.0, 0.1)  # Cap at 0.1
                recency_boost = recency_score * 0.05  # Small boost for recent usage
                similarity_boost = avg_similarity * 0.1
                
                total_boost = success_boost + frequency_boost + recency_boost + similarity_boost
                tool_scores[tool_name] += total_boost
                
                logger.debug(f"Tool {tool_name}: base={tool_scores[tool_name]:.3f}, "
                           f"success={success_boost:.3f}, freq={frequency_boost:.3f}, "
                           f"recency={recency_boost:.3f}, similarity={similarity_boost:.3f}")
            
            return tool_scores
            
        except Exception as e:
            logger.warning(f"Failed to query and analyze usage patterns: {e}")
            return tool_scores
    
    def _calculate_recency_score(self, timestamps: List[str]) -> float:
        """Calculate recency score based on timestamps (1.0 = very recent, 0.0 = old)."""
        if not timestamps:
            return 0.0
        
        try:
            from datetime import datetime, timedelta
            
            now = datetime.now()
            recent_scores = []
            
            for timestamp_str in timestamps:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age_hours = (now - timestamp).total_seconds() / 3600
                    
                    # Score: 1.0 for < 1 hour, 0.5 for < 24 hours, 0.0 for > 7 days
                    if age_hours < 1:
                        score = 1.0
                    elif age_hours < 24:
                        score = 0.8
                    elif age_hours < 24 * 7:  # 1 week
                        score = 0.3
                    else:
                        score = 0.0
                    
                    recent_scores.append(score)
                    
                except (ValueError, TypeError):
                    continue
            
            return max(recent_scores) if recent_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _extract_tool_names(self, selected_tools: List[Dict[str, Any]]) -> set:
        """Extract tool names from tool specifications."""
        selected_tool_names = set()
        for tool_spec in selected_tools:
            # Extract tool name from spec
            if "name" in tool_spec:
                selected_tool_names.add(tool_spec["name"])
            elif "toolSpec" in tool_spec and "name" in tool_spec["toolSpec"]:
                selected_tool_names.add(tool_spec["toolSpec"]["name"])
        return selected_tool_names
    
    def _select_top_scored_tools(self, tool_scores: Dict[str, float], available_tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select top-scored tools up to max_tools limit."""
        # Sort tools by score (highest first)
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top tools with positive scores
        top_tool_names = []
        for tool_name, score in sorted_tools:
            if score > 0 and len(top_tool_names) < self.config.max_tools:
                top_tool_names.append(tool_name)
        
        # Return corresponding tool specs
        enhanced_tools = []
        for tool_name in top_tool_names:
            if tool_name in available_tools:
                enhanced_tools.append(available_tools[tool_name])
        
        return enhanced_tools
