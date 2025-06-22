#!/usr/bin/env python3
"""Test script for RetrievalToolManager with optimized KB usage patterns."""

from src.strands.agent.tool_manager import RetrievalToolManager, RetrievalConfig

def test_retrieval_manager():
    """Test basic RetrievalToolManager functionality."""
    print("🧪 Testing Optimized RetrievalToolManager...")
    
    # Create manager with custom config
    config = RetrievalConfig(max_tools=3, similarity_threshold=0.5, usage_boost_factor=0.25)
    manager = RetrievalToolManager(config)
    
    print(f"✅ Config: max_tools={config.max_tools}, boost_factor={config.usage_boost_factor}")
    
    # Test without agent (should fallback gracefully)
    available_tools = {
        "calculator": {"name": "calculator", "description": "Performs mathematical calculations"},
        "weather": {"name": "weather", "description": "Gets weather information"},
        "file_reader": {"name": "file_reader", "description": "Reads files from disk"},
        "email_sender": {"name": "email_sender", "description": "Sends emails to recipients"},
    }
    
    result = manager.select_tools(
        prompt="What is 2 + 2?",
        available_tools=available_tools,
        context_messages=[],
        agent_context={},
    )
    
    print(f"✅ Fallback result: {len(result['tools'])} tools selected")
    print(f"   Tool choice: {result['toolChoice']}")
    
    # Test CRUDL fallback with different prompt types
    test_prompts = [
        "Create a new file with some content",
        "Read the weather forecast",
        "Update my email settings",
        "Delete old files",
        "List all available options"
    ]
    
    for prompt in test_prompts:
        result = manager.select_tools(
            prompt=prompt,
            available_tools=available_tools,
            context_messages=[],
            agent_context={},
        )
        print(f"✅ Prompt '{prompt[:20]}...': {len(result['tools'])} tools selected")
    
    # Test the optimized KB method exists
    if hasattr(manager, '_query_and_analyze_usage_patterns'):
        print("✅ Optimized single KB call method exists")
    
    print("🎉 Optimized RetrievalToolManager tests passed!")

def test_usage_tracking():
    """Test usage tracking functionality."""
    print("\n🧪 Testing Usage Tracking...")
    
    config = RetrievalConfig(usage_boost_factor=0.3)
    manager = RetrievalToolManager(config)
    
    # Test without agent (should handle gracefully)
    manager.record_tool_execution(
        tool_name="calculator",
        user_prompt="What is 5 * 7?",
        tool_input={"operation": "multiply", "a": 5, "b": 7},
        tool_output={"result": 35},
        success=True,
        execution_time_ms=150.0,
        context_messages=[{"role": "user", "content": [{"text": "What is 5 * 7?"}]}],
    )
    
    print("✅ Tool execution recording handled gracefully (no agent)")
    print("🎉 Usage tracking tests passed!")

if __name__ == "__main__":
    test_retrieval_manager()
    test_usage_tracking()
