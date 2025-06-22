"""Tool metadata and CRUDL operations for intelligent tool selection."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class CRUDLOperation(Enum):
    """CRUDL operations that tools can perform."""
    CREATE = "create"    # Creates new data/resources
    READ = "read"        # Reads/retrieves information  
    UPDATE = "update"    # Modifies existing data
    DELETE = "delete"    # Removes data/resources
    LIST = "list"        # Lists/searches collections


@dataclass
class ToolMetadata:
    """Rich metadata about a tool for intelligent selection."""
    name: str
    category: str                           # "calculation", "communication", "file_system"
    crudl_operations: List[CRUDLOperation]  # Operations this tool performs
    keywords: List[str]                     # Search keywords
    use_cases: List[str]                    # Specific use cases
    complexity: str                         # "simple", "moderate", "complex"
    description: str = ""                   # Tool description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in KB."""
        return {
            "name": self.name,
            "category": self.category,
            "crudl_operations": [op.value for op in self.crudl_operations],
            "keywords": self.keywords,
            "use_cases": self.use_cases,
            "complexity": self.complexity,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        """Create from dictionary loaded from KB."""
        return cls(
            name=data["name"],
            category=data["category"],
            crudl_operations=[CRUDLOperation(op) for op in data["crudl_operations"]],
            keywords=data["keywords"],
            use_cases=data["use_cases"],
            complexity=data["complexity"],
            description=data.get("description", ""),
        )
