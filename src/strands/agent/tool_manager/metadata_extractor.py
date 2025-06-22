"""LLM-based tool metadata extraction."""

import json
import logging
from typing import Any, Dict, Optional

from .tool_metadata import CRUDLOperation, ToolMetadata

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts tool metadata using LLM analysis."""
    
    def __init__(self, agent):
        """Initialize with agent reference for LLM calls.
        
        Args:
            agent: The agent instance to use for LLM calls
        """
        self.agent = agent
        self.cache: Dict[str, ToolMetadata] = {}
    
    def extract_metadata(self, tool_name: str, tool_spec: Dict[str, Any]) -> ToolMetadata:
        """Extract metadata for a tool using LLM analysis.
        
        Args:
            tool_name: Name of the tool
            tool_spec: Tool specification dictionary
            
        Returns:
            ToolMetadata with extracted information
        """
        # Check cache first
        if tool_name in self.cache:
            return self.cache[tool_name]
        
        # Build description from tool spec
        description = self._build_tool_description(tool_name, tool_spec)
        
        # Use LLM to analyze the tool
        try:
            metadata = self._analyze_tool_with_llm(tool_name, description)
            self.cache[tool_name] = metadata
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {tool_name}: {e}")
            # Return basic metadata as fallback
            return self._create_fallback_metadata(tool_name, description)
    
    def _build_tool_description(self, tool_name: str, tool_spec: Dict[str, Any]) -> str:
        """Build a comprehensive description from tool specification."""
        parts = [f"Tool: {tool_name}"]
        
        # Extract description from various possible locations
        if "description" in tool_spec:
            parts.append(f"Description: {tool_spec['description']}")
        elif "toolSpec" in tool_spec and "description" in tool_spec["toolSpec"]:
            parts.append(f"Description: {tool_spec['toolSpec']['description']}")
        
        # Extract input schema information
        if "toolSpec" in tool_spec and "inputSchema" in tool_spec["toolSpec"]:
            schema = tool_spec["toolSpec"]["inputSchema"]
            if "properties" in schema:
                params = list(schema["properties"].keys())
                parts.append(f"Parameters: {', '.join(params)}")
        
        return "\n".join(parts)
    
    def _analyze_tool_with_llm(self, tool_name: str, description: str) -> ToolMetadata:
        """Use LLM to analyze tool and extract metadata."""
        prompt = f"""Analyze this tool and extract metadata. Respond with valid JSON only.

{description}

Analyze what CRUDL operations this tool performs:
- CREATE: Creates new data, files, resources, calculations, or content
- READ: Reads, retrieves, fetches, or gets existing information
- UPDATE: Modifies, edits, or changes existing data
- DELETE: Removes, deletes, or destroys data/resources  
- LIST: Lists, searches, queries, or enumerates collections

JSON format (respond with ONLY this JSON, no other text):
{{
  "category": "primary_category_like_calculation_or_communication_or_file_system",
  "crudl_operations": ["create", "read"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "use_cases": ["specific use case 1", "specific use case 2"],
  "complexity": "simple"
}}

Complexity levels:
- simple: Basic operations, few parameters
- moderate: Multiple parameters or steps
- complex: Advanced logic or many dependencies"""

        # Use the agent to analyze the tool
        response = self.agent(prompt)
        
        # Extract JSON from response
        response_text = str(response)
        json_str = self._extract_json_from_response(response_text)
        
        # Parse the JSON
        try:
            data = json.loads(json_str)
            return ToolMetadata(
                name=tool_name,
                category=data["category"],
                crudl_operations=[CRUDLOperation(op) for op in data["crudl_operations"]],
                keywords=data["keywords"],
                use_cases=data["use_cases"],
                complexity=data["complexity"],
                description=description,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response for {tool_name}: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling various formats."""
        # Try to find JSON in the response
        response = response.strip()
        
        # Look for JSON block markers
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Look for JSON object markers
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return response[start:end]
        
        # If no markers found, assume entire response is JSON
        return response
    
    def _create_fallback_metadata(self, tool_name: str, description: str) -> ToolMetadata:
        """Create basic metadata when LLM analysis fails."""
        # Simple heuristics for fallback
        operations = [CRUDLOperation.READ]  # Default to READ
        category = "utility"
        keywords = [tool_name.lower()]
        use_cases = [f"Use {tool_name} for specific tasks"]
        
        # Basic heuristics based on tool name
        name_lower = tool_name.lower()
        if any(word in name_lower for word in ["create", "make", "generate", "build"]):
            operations.append(CRUDLOperation.CREATE)
            category = "creation"
        elif any(word in name_lower for word in ["update", "modify", "edit", "change"]):
            operations.append(CRUDLOperation.UPDATE)
            category = "modification"
        elif any(word in name_lower for word in ["delete", "remove", "clear"]):
            operations.append(CRUDLOperation.DELETE)
            category = "deletion"
        elif any(word in name_lower for word in ["list", "search", "find", "query"]):
            operations.append(CRUDLOperation.LIST)
            category = "search"
        
        return ToolMetadata(
            name=tool_name,
            category=category,
            crudl_operations=operations,
            keywords=keywords,
            use_cases=use_cases,
            complexity="simple",
            description=description,
        )
