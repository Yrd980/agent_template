"""
Comprehensive tests for tool_manager.py module.

Tests include ToolFunction, MCPClient, ToolManager, and integration tests
for Claude MCP and DeepSeek API call tools.
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from agent_template.services.tool_manager import (
    ToolFunction, MCPClient, ToolManager
)
from agent_template.models.tools import (
    ToolDefinition, ToolParameter, ToolCall, ToolResult,
    MCPServer, ToolExecutionContext, ToolType, ToolStatus,
    MCPServerStatus, ToolMetrics
)


class TestToolFunction:
    """Test cases for ToolFunction class."""

    @pytest.fixture
    def sample_tool_definition(self):
        """Create a sample tool definition."""
        return ToolDefinition(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.FUNCTION,
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="Input text",
                    required=True,
                    min_length=1,
                    max_length=100
                ),
                ToolParameter(
                    name="count",
                    type="number",
                    description="Count parameter",
                    required=False,
                    minimum=0,
                    maximum=10
                )
            ]
        )

    @pytest.fixture
    def sync_function(self):
        """Create a synchronous test function."""
        def test_func(text: str, count: int = 1) -> str:
            return f"{text} x {count}"
        return test_func

    @pytest.fixture
    def async_function(self):
        """Create an asynchronous test function."""
        async def test_func(text: str, count: int = 1) -> str:
            await asyncio.sleep(0.001)  # Simulate async work
            return f"{text} x {count}"
        return test_func

    @pytest.fixture
    def context_function(self):
        """Create a function that requires context."""
        def test_func(context: ToolExecutionContext, text: str) -> str:
            return f"Session {context.session_id}: {text}"
        return test_func

    def test_tool_function_init(self, sample_tool_definition, sync_function):
        """Test ToolFunction initialization."""
        tool_func = ToolFunction(
            func=sync_function,
            definition=sample_tool_definition,
            context_required=False
        )
        
        assert tool_func.func == sync_function
        assert tool_func.definition == sample_tool_definition
        assert tool_func.context_required is False
        assert isinstance(tool_func.metrics, ToolMetrics)
        assert tool_func.metrics.tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, sample_tool_definition, sync_function):
        """Test executing synchronous function."""
        tool_func = ToolFunction(sync_function, sample_tool_definition)
        
        parameters = {"text": "hello", "count": 3}
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.COMPLETED
        assert result.tool_name == "test_tool"
        assert result.data["result"] == "hello x 3"
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_async_function(self, sample_tool_definition, async_function):
        """Test executing asynchronous function."""
        tool_func = ToolFunction(async_function, sample_tool_definition)
        
        parameters = {"text": "world", "count": 2}
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.COMPLETED
        assert result.tool_name == "test_tool"
        assert result.data["result"] == "world x 2"
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_with_context(self, sample_tool_definition, context_function):
        """Test executing function with context."""
        tool_func = ToolFunction(context_function, sample_tool_definition, context_required=True)
        
        context = ToolExecutionContext(
            call_id="test-call",
            session_id="test-session"
        )
        parameters = {"text": "test"}
        
        result = await tool_func.execute(parameters, context)
        
        assert result.status == ToolStatus.COMPLETED
        assert result.data["result"] == "Session test-session: test"

    @pytest.mark.asyncio
    async def test_parameter_validation_missing_required(self, sample_tool_definition, sync_function):
        """Test parameter validation with missing required parameter."""
        tool_func = ToolFunction(sync_function, sample_tool_definition)
        
        parameters = {"count": 5}  # Missing required 'text' parameter
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.FAILED
        assert "Required parameter 'text' is missing" in result.error

    @pytest.mark.asyncio
    async def test_parameter_validation_type_error(self, sample_tool_definition, sync_function):
        """Test parameter validation with wrong type."""
        tool_func = ToolFunction(sync_function, sample_tool_definition)
        
        parameters = {"text": 123, "count": 5}  # text should be string
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.FAILED
        assert "must be a string" in result.error

    @pytest.mark.asyncio
    async def test_parameter_validation_constraint_violation(self, sample_tool_definition, sync_function):
        """Test parameter validation with constraint violation."""
        tool_func = ToolFunction(sync_function, sample_tool_definition)
        
        parameters = {"text": "a" * 150, "count": 5}  # text too long (max 100)
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.FAILED
        assert "must be at most 100 characters" in result.error

    @pytest.mark.asyncio
    async def test_function_exception_handling(self, sample_tool_definition):
        """Test exception handling in function execution."""
        def failing_func(text: str) -> str:
            raise ValueError("Test error")

        tool_func = ToolFunction(failing_func, sample_tool_definition)
        
        parameters = {"text": "test"}
        result = await tool_func.execute(parameters)
        
        assert result.status == ToolStatus.FAILED
        assert result.error == "Test error"
        assert result.execution_time > 0


class TestMCPClient:
    """Test cases for MCPClient class."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock MCP server."""
        return MCPServer(
            name="test-server",
            url="http://localhost:8080",
            timeout=30.0,
            auth_type="bearer",
            auth_data={"token": "test-token"}
        )

    @pytest.fixture
    def mcp_client(self, mock_server):
        """Create MCP client with mock server."""
        # Create client but don't initialize httpx client yet
        client = MCPClient.__new__(MCPClient)
        client.server = mock_server
        client.client = AsyncMock()  # Use mock client
        client._tools_cache = None
        client._last_ping = None
        return client

    @pytest.mark.asyncio
    async def test_mcp_client_init(self, mcp_client, mock_server):
        """Test MCPClient initialization."""
        assert mcp_client.server == mock_server
        assert mcp_client.client is not None  # Should be AsyncMock in tests
        assert mcp_client._tools_cache is None

    @pytest.mark.asyncio
    async def test_connect_success(self, mcp_client):
        """Test successful connection to MCP server."""
        # Mock the health check response
        mock_health_response = MagicMock()
        mock_health_response.raise_for_status.return_value = None

        # Mock the tools discovery response
        mock_tools_response = MagicMock()
        mock_tools_response.json.return_value = {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "input": {
                            "type": "string",
                            "description": "Input text",
                            "required": True
                        }
                    }
                }
            ]
        }
        mock_tools_response.raise_for_status.return_value = None

        mcp_client.client.get.side_effect = [mock_health_response, mock_tools_response]
        
        result = await mcp_client.connect()
        
        assert result is True
        assert mcp_client.server.status == MCPServerStatus.CONNECTED
        assert mcp_client.server.error is None
        assert len(mcp_client.server.tools) == 1
        assert mcp_client.server.tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_connect_failure(self, mcp_client):
        """Test failed connection to MCP server."""
        mcp_client.client.get.side_effect = httpx.ConnectError("Connection failed")
        
        result = await mcp_client.connect()
        
        assert result is False
        assert mcp_client.server.status == MCPServerStatus.ERROR
        assert "Connection failed" in mcp_client.server.error

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mcp_client):
        """Test successful tool call."""
        context = ToolExecutionContext(call_id="test-call", session_id="test-session")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"output": "Tool executed successfully"},
            "output": "Success"
        }
        mock_response.raise_for_status.return_value = None

        mcp_client.client.post.return_value = mock_response
        
        result = await mcp_client.call_tool(
            tool_name="test_tool",
            parameters={"input": "test"},
            context=context
        )
        
        assert result.status == ToolStatus.COMPLETED
        assert result.tool_name == "test_tool"
        assert result.call_id == "test-call"
        assert result.data["output"] == "Tool executed successfully"
        assert mcp_client.server.successful_calls == 1

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self, mcp_client):
        """Test tool call timeout."""
        context = ToolExecutionContext(call_id="test-call", session_id="test-session")

        mcp_client.client.post.side_effect = httpx.TimeoutException("Request timed out")
        
        result = await mcp_client.call_tool(
            tool_name="test_tool",
            parameters={"input": "test"},
            context=context
        )
        
        assert result.status == ToolStatus.TIMEOUT
        assert result.error == "Tool execution timed out"
        assert mcp_client.server.failed_calls == 1

    @pytest.mark.asyncio
    async def test_ping_success(self, mcp_client):
        """Test successful server ping."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        mcp_client.client.get.return_value = mock_response
        
        result = await mcp_client.ping()
        
        assert result is True
        assert mcp_client.server.last_ping is not None

    @pytest.mark.asyncio
    async def test_ping_failure(self, mcp_client):
        """Test failed server ping."""
        mcp_client.client.get.side_effect = httpx.ConnectError("Ping failed")
        
        result = await mcp_client.ping()
        
        assert result is False


class TestToolManager:
    """Test cases for ToolManager class."""

    @pytest.fixture
    async def tool_manager(self):
        """Create a tool manager instance."""
        manager = ToolManager()
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_tool_manager_init(self):
        """Test ToolManager initialization."""
        manager = ToolManager()
        
        assert manager.registry is not None
        assert manager.discovery is not None
        assert len(manager._functions) == 0
        assert len(manager._mcp_clients) == 0
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_register_function(self, tool_manager):
        """Test function tool registration."""
        def test_function(text: str) -> str:
            """A test function."""
            return f"Result: {text}"

        parameters = [
            ToolParameter(
                name="text",
                type="string",
                description="Input text",
                required=True
            )
        ]

        tool_name = tool_manager.register_function(
            func=test_function,
            name="test_func",
            description="Test function tool",
            parameters=parameters
        )
        
        assert tool_name == "test_func"
        assert "test_func" in tool_manager._functions
        assert tool_manager.get_tool("test_func") is not None

    @pytest.mark.asyncio
    async def test_unregister_tool(self, tool_manager):
        """Test tool unregistration."""
        def test_function(text: str) -> str:
            return text

        tool_name = tool_manager.register_function(test_function, "test_func")
        assert tool_manager.get_tool("test_func") is not None
        
        result = tool_manager.unregister_tool("test_func")
        assert result is True
        assert tool_manager.get_tool("test_func") is None

    @pytest.mark.asyncio
    async def test_call_function_tool(self, tool_manager):
        """Test calling a registered function tool."""
        def test_function(text: str, multiplier: int = 1) -> str:
            return text * multiplier

        parameters = [
            ToolParameter(name="text", type="string", required=True),
            ToolParameter(name="multiplier", type="number", required=False)
        ]

        tool_manager.register_function(
            test_function, "repeat_text", parameters=parameters
        )

        call = await tool_manager.call_tool(
            tool_name="repeat_text",
            parameters={"text": "hello", "multiplier": 3},
            session_id="test-session"
        )
        
        # Wait a bit for execution
        await asyncio.sleep(0.1)
        
        assert call.tool_name == "repeat_text"
        assert call.session_id == "test-session"


class TestClaudeMCPIntegration:
    """Test Claude MCP tool integration."""

    @pytest.fixture
    def claude_mcp_server_config(self):
        """Configuration for Claude MCP server."""
        return {
            "name": "claude-mcp",
            "url": "http://localhost:3000",
            "auth_type": "api_key",
            "auth_data": {"key": "claude-mcp-key"}
        }

    @pytest.mark.asyncio
    async def test_claude_mcp_tool_registration(self, claude_mcp_server_config):
        """Test registering Claude MCP tools."""
        manager = ToolManager()
        await manager.start()
        
        try:
            # Mock the MCP server responses
            mock_health = MagicMock()
            mock_health.raise_for_status.return_value = None
            
            mock_tools = MagicMock()
            mock_tools.json.return_value = {
                "tools": [
                    {
                        "name": "claude_reasoning",
                        "description": "Claude reasoning and analysis tool",
                        "parameters": {
                            "query": {
                                "type": "string",
                                "description": "Query to analyze",
                                "required": True
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context",
                                "required": False
                            }
                        }
                    }
                ]
            }
            mock_tools.raise_for_status.return_value = None

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.get.side_effect = [mock_health, mock_tools]
                
                success = await manager.add_mcp_server(**claude_mcp_server_config)
                
                assert success is True
                
                # Verify tool is registered
                claude_tool = manager.get_tool("claude_reasoning")
                assert claude_tool is not None
                assert claude_tool.tool_type == ToolType.MCP_TOOL
                assert claude_tool.server_name == "claude-mcp"
                
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_claude_mcp_tool_execution(self, claude_mcp_server_config):
        """Test executing Claude MCP tool."""
        manager = ToolManager()
        await manager.start()
        
        try:
            # Mock server setup
            mock_health = MagicMock()
            mock_health.raise_for_status.return_value = None
            
            mock_tools = MagicMock()
            mock_tools.json.return_value = {
                "tools": [{
                    "name": "claude_reasoning",
                    "description": "Claude reasoning tool",
                    "parameters": {
                        "query": {"type": "string", "required": True}
                    }
                }]
            }
            mock_tools.raise_for_status.return_value = None
            
            mock_execute = MagicMock()
            mock_execute.json.return_value = {
                "result": {
                    "reasoning": "Analysis of the query shows...",
                    "conclusion": "Based on the evidence..."
                }
            }
            mock_execute.raise_for_status.return_value = None

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client
                mock_client.get.side_effect = [mock_health, mock_tools]
                mock_client.post.return_value = mock_execute
                
                # Add server and execute tool
                await manager.add_mcp_server(**claude_mcp_server_config)
                
                call = await manager.call_tool(
                    tool_name="claude_reasoning",
                    parameters={"query": "What is the meaning of life?"},
                    session_id="test-session"
                )
                
                # Wait for execution
                await asyncio.sleep(0.1)
                
                assert call.tool_name == "claude_reasoning"
                
        finally:
            await manager.stop()


class TestDeepSeekAPITool:
    """Test DeepSeek API call tool integration."""

    @pytest.mark.asyncio
    async def test_deepseek_api_tool_registration(self):
        """Test registering DeepSeek API tool."""
        manager = ToolManager()
        await manager.start()
        
        try:
            async def deepseek_chat_completion(
                context: ToolExecutionContext,
                messages: List[Dict[str, str]],
                model: str = "deepseek-chat",
                temperature: float = 0.7,
                max_tokens: int = 1000
            ) -> Dict[str, Any]:
                """Call DeepSeek API for chat completion."""
                # Mock API call
                return {
                    "model": model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "This is a mock DeepSeek response."
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18
                    }
                }

            parameters = [
                ToolParameter(
                    name="messages",
                    type="array",
                    description="Chat messages",
                    required=True
                ),
                ToolParameter(
                    name="model",
                    type="string",
                    description="Model name",
                    required=False,
                    enum=["deepseek-chat", "deepseek-coder"]
                ),
                ToolParameter(
                    name="temperature",
                    type="number",
                    description="Response creativity",
                    required=False,
                    minimum=0.0,
                    maximum=2.0
                ),
                ToolParameter(
                    name="max_tokens",
                    type="number",
                    description="Maximum tokens",
                    required=False,
                    minimum=1,
                    maximum=4000
                )
            ]

            tool_name = manager.register_function(
                func=deepseek_chat_completion,
                name="deepseek_chat",
                description="DeepSeek API chat completion",
                parameters=parameters,
                context_required=True
            )
            
            assert tool_name == "deepseek_chat"
            
            # Verify tool registration
            tool_def = manager.get_tool("deepseek_chat")
            assert tool_def is not None
            assert tool_def.tool_type == ToolType.FUNCTION
            assert len(tool_def.parameters) == 4
            
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_deepseek_api_tool_execution(self):
        """Test executing DeepSeek API tool."""
        manager = ToolManager()
        await manager.start()
        
        try:
            async def deepseek_api_call(
                context: ToolExecutionContext,
                prompt: str,
                model: str = "deepseek-chat"
            ) -> Dict[str, Any]:
                """Mock DeepSeek API call."""
                await asyncio.sleep(0.01)  # Simulate API delay
                
                return {
                    "response": f"DeepSeek processed: {prompt}",
                    "model_used": model,
                    "session_id": context.session_id,
                    "tokens_used": 25
                }

            parameters = [
                ToolParameter(name="prompt", type="string", required=True),
                ToolParameter(name="model", type="string", required=False)
            ]

            manager.register_function(
                func=deepseek_api_call,
                name="deepseek_api",
                description="DeepSeek API integration",
                parameters=parameters,
                context_required=True
            )

            call = await manager.call_tool(
                tool_name="deepseek_api",
                parameters={
                    "prompt": "Explain quantum computing",
                    "model": "deepseek-chat"
                },
                session_id="test-session"
            )
            
            # Wait for execution
            await asyncio.sleep(0.1)
            
            assert call.tool_name == "deepseek_api"
            assert call.session_id == "test-session"
            
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_deepseek_parameter_validation(self):
        """Test DeepSeek tool parameter validation."""
        manager = ToolManager()
        await manager.start()
        
        try:
            def deepseek_tool(prompt: str, temperature: float) -> str:
                return f"Response to: {prompt}"

            parameters = [
                ToolParameter(
                    name="prompt",
                    type="string",
                    required=True,
                    min_length=1,
                    max_length=2000
                ),
                ToolParameter(
                    name="temperature",
                    type="number",
                    required=True,
                    minimum=0.0,
                    maximum=2.0
                )
            ]

            manager.register_function(
                deepseek_tool, "deepseek_test", parameters=parameters
            )

            # Test with valid parameters
            call1 = await manager.call_tool(
                "deepseek_test",
                {"prompt": "Hello", "temperature": 0.7},
                "session1"
            )
            await asyncio.sleep(0.1)
            assert call1.tool_name == "deepseek_test"

            # Test with invalid temperature (too high)
            call2 = await manager.call_tool(
                "deepseek_test",
                {"prompt": "Hello", "temperature": 3.0},  # Invalid: > 2.0
                "session2"
            )
            await asyncio.sleep(0.1)
            # The validation error would be caught in the execution
            
        finally:
            await manager.stop()


@pytest.mark.integration
class TestToolManagerIntegration:
    """Integration tests for complete tool manager functionality."""

    @pytest.mark.asyncio
    async def test_mixed_tool_execution(self):
        """Test executing both function and MCP tools."""
        manager = ToolManager()
        await manager.start()
        
        try:
            # Register a function tool
            def simple_calculator(operation: str, a: float, b: float) -> float:
                """Simple calculator function."""
                if operation == "add":
                    return a + b
                elif operation == "multiply":
                    return a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            calc_params = [
                ToolParameter(name="operation", type="string", required=True),
                ToolParameter(name="a", type="number", required=True),
                ToolParameter(name="b", type="number", required=True)
            ]

            manager.register_function(
                simple_calculator, "calculator", parameters=calc_params
            )

            # Test function tool
            calc_call = await manager.call_tool(
                "calculator",
                {"operation": "add", "a": 5.0, "b": 3.0},
                "session1"
            )
            
            await asyncio.sleep(0.1)
            assert calc_call.tool_name == "calculator"

            # Test stats
            stats = manager.get_stats()
            assert stats["registered_tools"] >= 1
            assert stats["function_tools"] >= 1
            
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_tool_manager_lifecycle(self):
        """Test complete tool manager lifecycle."""
        manager = ToolManager()
        
        # Test initial state
        assert not manager._running
        assert len(manager._functions) == 0
        
        # Start manager
        await manager.start()
        assert manager._running
        
        # Register tools and test functionality
        def echo_tool(message: str) -> str:
            return f"Echo: {message}"
        
        manager.register_function(echo_tool, "echo")
        assert len(manager._functions) == 1
        
        # Test tool execution
        call = await manager.call_tool("echo", {"message": "test"}, "session")
        assert call is not None
        
        # Stop manager
        await manager.stop()
        assert not manager._running