"""
Tool management and MCP (Model Context Protocol) integration service.

This module provides comprehensive tool calling capabilities including
function execution, MCP server integration, and dynamic tool discovery.
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable, Union
from weakref import WeakValueDictionary

import httpx
import structlog

from ..config import settings
from ..core.async_queue import AsyncQueue, QueueItem
from ..models.tools import (
    ToolDefinition, ToolCall, ToolResult, ToolRegistry, ToolMetrics,
    MCPServer, ToolExecutionContext, ToolDiscovery,
    ToolType, ToolStatus, MCPServerStatus, ToolParameter
)


logger = structlog.get_logger(__name__)


class ToolFunction:
    """Wrapper for tool functions with metadata and execution context."""
    
    def __init__(
        self,
        func: Callable,
        definition: ToolDefinition,
        context_required: bool = False
    ):
        self.func = func
        self.definition = definition
        self.context_required = context_required
        self.metrics = ToolMetrics(tool_name=definition.name)
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        """Execute the tool function."""
        start_time = time.time()
        call_id = context.call_id if context else "unknown"
        
        try:
            # Validate parameters
            self._validate_parameters(parameters)
            
            # Prepare function call
            if self.context_required and context:
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(context=context, **parameters)
                else:
                    result = self.func(context=context, **parameters)
            else:
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(**parameters)
                else:
                    result = self.func(**parameters)
            
            execution_time = time.time() - start_time
            
            # Record successful execution
            self.metrics.record_call(True, execution_time)
            
            return ToolResult(
                call_id=call_id,
                tool_name=self.definition.name,
                status=ToolStatus.COMPLETED,
                data={"result": result} if not isinstance(result, dict) else result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Record failed execution
            self.metrics.record_call(False, execution_time, error_msg)
            
            return ToolResult(
                call_id=call_id,
                tool_name=self.definition.name,
                status=ToolStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate function parameters."""
        required_params = [p.name for p in self.definition.parameters if p.required]
        
        # Check required parameters
        for param_name in required_params:
            if param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' is missing")
        
        # Validate parameter types and constraints
        for param in self.definition.parameters:
            if param.name not in parameters:
                continue
                
            value = parameters[param.name]
            self._validate_parameter_value(param, value)
    
    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> None:
        """Validate a single parameter value."""
        # Type validation
        if param.type == "string" and not isinstance(value, str):
            raise ValueError(f"Parameter '{param.name}' must be a string")
        elif param.type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{param.name}' must be a number")
        elif param.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Parameter '{param.name}' must be a boolean")
        elif param.type == "array" and not isinstance(value, list):
            raise ValueError(f"Parameter '{param.name}' must be an array")
        elif param.type == "object" and not isinstance(value, dict):
            raise ValueError(f"Parameter '{param.name}' must be an object")
        
        # Constraint validation
        if param.enum and value not in param.enum:
            raise ValueError(f"Parameter '{param.name}' must be one of {param.enum}")
        
        if isinstance(value, (int, float)):
            if param.minimum is not None and value < param.minimum:
                raise ValueError(f"Parameter '{param.name}' must be >= {param.minimum}")
            if param.maximum is not None and value > param.maximum:
                raise ValueError(f"Parameter '{param.name}' must be <= {param.maximum}")
        
        if isinstance(value, str):
            if param.min_length is not None and len(value) < param.min_length:
                raise ValueError(f"Parameter '{param.name}' must be at least {param.min_length} characters")
            if param.max_length is not None and len(value) > param.max_length:
                raise ValueError(f"Parameter '{param.name}' must be at most {param.max_length} characters")


class MCPClient:
    """Client for communicating with MCP servers."""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.client = httpx.AsyncClient(timeout=server.timeout)
        self._tools_cache: Optional[List[ToolDefinition]] = None
        self._last_ping: Optional[float] = None
    
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            self.server.status = MCPServerStatus.CONNECTING
            
            # Test connection with ping
            response = await self.client.get(f"{self.server.url}/health")
            response.raise_for_status()
            
            # Discover tools
            await self._discover_tools()
            
            self.server.status = MCPServerStatus.CONNECTED
            self.server.connected_at = datetime.utcnow()
            self.server.error = None
            
            logger.info(f"Connected to MCP server '{self.server.name}'", 
                       url=self.server.url, tools=len(self.server.tools))
            return True
            
        except Exception as e:
            self.server.status = MCPServerStatus.ERROR
            self.server.error = str(e)
            logger.error(f"Failed to connect to MCP server '{self.server.name}'", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            await self.client.aclose()
            self.server.status = MCPServerStatus.DISCONNECTED
            self.server.connected_at = None
            logger.info(f"Disconnected from MCP server '{self.server.name}'")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        """Call a tool on the MCP server."""
        start_time = time.time()
        call_id = context.call_id if context else "unknown"
        
        try:
            # Prepare request
            request_data = {
                "tool": tool_name,
                "parameters": parameters,
                "context": {
                    "call_id": call_id,
                    "session_id": context.session_id if context else "unknown"
                }
            }
            
            # Add authentication if configured
            headers = {}
            if self.server.auth_type == "bearer" and "token" in self.server.auth_data:
                headers["Authorization"] = f"Bearer {self.server.auth_data['token']}"
            elif self.server.auth_type == "api_key" and "key" in self.server.auth_data:
                headers["X-API-Key"] = self.server.auth_data["key"]
            
            # Make the call
            response = await self.client.post(
                f"{self.server.url}/tools/{tool_name}/execute",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            result_data = response.json()
            execution_time = time.time() - start_time
            
            # Update server stats
            self.server.total_calls += 1
            self.server.successful_calls += 1
            
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.COMPLETED,
                data=result_data.get("result", {}),
                output=result_data.get("output"),
                execution_time=execution_time,
                metadata={"server": self.server.name}
            )
            
        except httpx.TimeoutException:
            execution_time = time.time() - start_time
            self.server.total_calls += 1
            self.server.failed_calls += 1
            
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                error="Tool execution timed out",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.server.total_calls += 1
            self.server.failed_calls += 1
            
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
    
    async def ping(self) -> bool:
        """Ping the MCP server to check health."""
        try:
            response = await self.client.get(f"{self.server.url}/health", timeout=5)
            response.raise_for_status()
            
            self.server.last_ping = datetime.utcnow()
            self._last_ping = time.time()
            
            return True
            
        except Exception as e:
            logger.warning(f"Ping failed for MCP server '{self.server.name}'", error=str(e))
            return False
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        try:
            response = await self.client.get(f"{self.server.url}/tools")
            response.raise_for_status()
            
            tools_data = response.json()
            tools = []
            
            for tool_data in tools_data.get("tools", []):
                # Convert to ToolDefinition
                parameters = []
                
                if "parameters" in tool_data:
                    for param_name, param_info in tool_data["parameters"].items():
                        parameters.append(ToolParameter(
                            name=param_name,
                            type=param_info.get("type", "string"),
                            description=param_info.get("description"),
                            required=param_info.get("required", True)
                        ))
                
                tool = ToolDefinition(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    tool_type=ToolType.MCP_TOOL,
                    parameters=parameters,
                    server_name=self.server.name,
                    server_url=self.server.url
                )
                
                tools.append(tool)
            
            self.server.tools = tools
            self._tools_cache = tools
            
            logger.info(f"Discovered {len(tools)} tools from MCP server '{self.server.name}'")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from MCP server '{self.server.name}'", error=str(e))
            self.server.tools = []


class ToolManager:
    """
    Comprehensive tool management system.
    
    Manages tool registration, execution, MCP server integration,
    and dynamic tool discovery.
    """
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.discovery = ToolDiscovery()
        
        # Function registry
        self._functions: Dict[str, ToolFunction] = {}
        
        # MCP clients
        self._mcp_clients: Dict[str, MCPClient] = {}
        
        # Execution tracking
        self._active_calls: Dict[str, ToolCall] = {}
        self._execution_contexts: WeakValueDictionary[str, ToolExecutionContext] = WeakValueDictionary()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Execution queue
        self._execution_queue = AsyncQueue(name="tool_execution", maxsize=100)
        
        logger.info("ToolManager initialized")
    
    async def start(self) -> None:
        """Start the tool manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._background_tasks.add(
            asyncio.create_task(self._tool_executor())
        )
        
        if self.discovery.enabled:
            self._background_tasks.add(
                asyncio.create_task(self._discovery_loop())
            )
            self._background_tasks.add(
                asyncio.create_task(self._health_monitor())
            )
        
        # Connect to configured MCP servers
        await self._connect_mcp_servers()
        
        logger.info("ToolManager started")
    
    async def stop(self) -> None:
        """Stop the tool manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Disconnect MCP servers
        for client in self._mcp_clients.values():
            await client.disconnect()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        # Close execution queue
        await self._execution_queue.close()
        
        logger.info("ToolManager stopped")
    
    # Tool registration
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
        context_required: bool = False
    ) -> str:
        """Register a Python function as a tool."""
        tool_name = name or func.__name__
        
        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=description or func.__doc__ or "",
            tool_type=ToolType.FUNCTION,
            parameters=parameters or []
        )
        
        # Create tool function wrapper
        tool_func = ToolFunction(func, tool_def, context_required)
        
        # Register
        self.registry.add_tool(tool_def)
        self._functions[tool_name] = tool_func
        
        logger.info(f"Registered function tool '{tool_name}'")
        return tool_name
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        # Remove from registry
        if not self.registry.remove_tool(tool_name):
            return False
        
        # Remove function if it exists
        if tool_name in self._functions:
            del self._functions[tool_name]
        
        logger.info(f"Unregistered tool '{tool_name}'")
        return True
    
    # MCP server management
    async def add_mcp_server(
        self,
        name: str,
        url: str,
        auth_type: Optional[str] = None,
        auth_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an MCP server."""
        server = MCPServer(
            name=name,
            url=url,
            auth_type=auth_type,
            auth_data=auth_data or {}
        )
        
        # Create client and try to connect
        client = MCPClient(server)
        
        if await client.connect():
            self.registry.add_mcp_server(server)
            self._mcp_clients[name] = client
            
            # Register discovered tools
            for tool in server.tools:
                self.registry.add_tool(tool)
            
            logger.info(f"Added MCP server '{name}' with {len(server.tools)} tools")
            return True
        else:
            logger.error(f"Failed to add MCP server '{name}'")
            return False
    
    async def remove_mcp_server(self, server_name: str) -> bool:
        """Remove an MCP server."""
        if server_name not in self._mcp_clients:
            return False
        
        # Disconnect client
        client = self._mcp_clients[server_name]
        await client.disconnect()
        del self._mcp_clients[server_name]
        
        # Remove from registry
        self.registry.remove_mcp_server(server_name)
        
        # Remove server tools from registry
        server_tools = [
            tool for tool in self.registry.tools.values()
            if tool.server_name == server_name
        ]
        
        for tool in server_tools:
            self.registry.remove_tool(tool.name)
        
        logger.info(f"Removed MCP server '{server_name}'")
        return True
    
    # Tool execution
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: str,
        message_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolCall:
        """Call a tool asynchronously."""
        # Create tool call
        call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            session_id=session_id,
            message_id=message_id
        )
        
        # Create execution context
        exec_context = ToolExecutionContext(
            call_id=call.id,
            session_id=session_id,
            environment=context or {}
        )
        
        # Store context and call
        self._execution_contexts[call.id] = exec_context
        self._active_calls[call.id] = call
        
        # Queue for execution
        await self._execution_queue.put(call)
        
        logger.info(f"Queued tool call '{tool_name}'", call_id=call.id)
        return call
    
    async def get_tool_call(self, call_id: str) -> Optional[ToolCall]:
        """Get a tool call by ID."""
        return self._active_calls.get(call_id)
    
    async def cancel_tool_call(self, call_id: str) -> bool:
        """Cancel a tool call."""
        if call_id in self._active_calls:
            call = self._active_calls[call_id]
            call.status = ToolStatus.CANCELLED
            logger.info(f"Cancelled tool call", call_id=call_id)
            return True
        return False
    
    # Tool information
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[ToolDefinition]:
        """List available tools."""
        return self.registry.list_tools(tool_type)
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition."""
        return self.registry.get_tool(tool_name)
    
    def get_tool_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """Get tool execution metrics."""
        if tool_name in self._functions:
            return self._functions[tool_name].metrics
        return None
    
    # Tool execution (internal)
    async def _execute_tool(
        self,
        call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute a single tool call."""
        call.status = ToolStatus.RUNNING
        call.started_at = datetime.utcnow()
        
        try:
            tool_def = self.registry.get_tool(call.tool_name)
            if not tool_def:
                raise ValueError(f"Tool '{call.tool_name}' not found")
            
            # Route to appropriate executor
            if tool_def.tool_type == ToolType.FUNCTION:
                # Execute Python function
                if call.tool_name not in self._functions:
                    raise ValueError(f"Function '{call.tool_name}' not registered")
                
                tool_func = self._functions[call.tool_name]
                result = await tool_func.execute(call.parameters, context)
                
            elif tool_def.tool_type == ToolType.MCP_TOOL:
                # Execute via MCP server
                if not tool_def.server_name or tool_def.server_name not in self._mcp_clients:
                    raise ValueError(f"MCP server for tool '{call.tool_name}' not available")
                
                client = self._mcp_clients[tool_def.server_name]
                result = await client.call_tool(call.tool_name, call.parameters, context)
                
            else:
                raise ValueError(f"Unsupported tool type: {tool_def.tool_type}")
            
            # Update call with result
            call.status = result.status
            call.result = result.data
            call.error = result.error
            call.completed_at = datetime.utcnow()
            call.execution_time = result.execution_time
            
            # Update registry stats
            self.registry.update_call_stats(result.status == ToolStatus.COMPLETED)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            call.status = ToolStatus.FAILED
            call.error = error_msg
            call.completed_at = datetime.utcnow()
            
            # Update registry stats
            self.registry.update_call_stats(False)
            
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                status=ToolStatus.FAILED,
                error=error_msg,
                execution_time=0.0
            )
    
    # Background tasks
    async def _tool_executor(self) -> None:
        """Background tool executor."""
        while self._running:
            try:
                # Get tool call from queue
                queue_item = await self._execution_queue.get(timeout=1.0)
                if queue_item is None:
                    continue
                
                call = queue_item.data
                context = self._execution_contexts.get(call.id)
                
                if context is None:
                    logger.error(f"Missing execution context for call {call.id}")
                    await self._execution_queue.task_done(queue_item, success=False)
                    continue
                
                # Execute the tool
                result = await self._execute_tool(call, context)
                
                # Mark task as done
                await self._execution_queue.task_done(queue_item, success=True)
                
                logger.info(f"Executed tool '{call.tool_name}'", 
                           call_id=call.id, status=result.status)
                
            except Exception as e:
                logger.error("Error in tool executor", error=str(e))
                if 'queue_item' in locals():
                    await self._execution_queue.task_done(queue_item, success=False)
                await asyncio.sleep(1)
    
    async def _discovery_loop(self) -> None:
        """Periodic tool discovery."""
        while self._running:
            try:
                await self._discover_tools()
                await asyncio.sleep(self.discovery.discovery_interval)
            except Exception as e:
                logger.error("Error in discovery loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _health_monitor(self) -> None:
        """Monitor MCP server health."""
        while self._running:
            try:
                for client in self._mcp_clients.values():
                    if client.server.status == MCPServerStatus.CONNECTED:
                        healthy = await client.ping()
                        if not healthy and client.server.status == MCPServerStatus.CONNECTED:
                            client.server.status = MCPServerStatus.ERROR
                            logger.warning(f"MCP server '{client.server.name}' health check failed")
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error("Error in health monitor", error=str(e))
                await asyncio.sleep(60)
    
    async def _connect_mcp_servers(self) -> None:
        """Connect to configured MCP servers."""
        # This would typically load from configuration
        # For now, it's a placeholder
        pass
    
    async def _discover_tools(self) -> None:
        """Discover new tools from various sources."""
        discovered = 0
        errors = []
        
        try:
            # Rediscover tools from MCP servers
            for client in self._mcp_clients.values():
                if client.server.status == MCPServerStatus.CONNECTED:
                    try:
                        await client._discover_tools()
                        
                        # Update registry with new tools
                        for tool in client.server.tools:
                            if tool.name not in self.registry.tools:
                                self.registry.add_tool(tool)
                                discovered += 1
                        
                    except Exception as e:
                        errors.append(f"Error discovering from {client.server.name}: {e}")
            
            # Update discovery info
            self.discovery.last_discovery = datetime.utcnow()
            self.discovery.discovered_tools = discovered
            self.discovery.discovery_errors = errors
            
            if discovered > 0:
                logger.info(f"Discovered {discovered} new tools")
            
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get tool manager statistics."""
        return {
            "registry": self.registry.stats,
            "active_calls": len(self._active_calls),
            "execution_queue_size": self._execution_queue.qsize(),
            "mcp_servers": len(self._mcp_clients),
            "connected_servers": len([
                c for c in self._mcp_clients.values() 
                if c.server.status == MCPServerStatus.CONNECTED
            ]),
            "functions": len(self._functions),
            "discovery": {
                "enabled": self.discovery.enabled,
                "last_discovery": self.discovery.last_discovery,
                "discovered_tools": self.discovery.discovered_tools,
                "errors": len(self.discovery.discovery_errors)
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool manager statistics."""
        return {
            "registered_tools": len(self.registry.tools),
            "active_calls": len(self._active_calls),
            "total_calls": self.registry.total_calls,
            "successful_calls": self.registry.successful_calls,
            "failed_calls": self.registry.failed_calls,
            "function_tools": len(self._functions),
            "mcp_clients": len(self._mcp_clients),
            "execution_contexts": len(self._execution_contexts)
        }
    
    # Context manager for tool execution
    @asynccontextmanager
    async def execution_context(
        self,
        session_id: str,
        permissions: Optional[Dict[str, bool]] = None,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        """Create an execution context for tool calls."""
        context = ToolExecutionContext(
            call_id=str(uuid.uuid4()),
            session_id=session_id,
            permissions=permissions or {},
            resource_limits=resource_limits or {}
        )
        
        self._execution_contexts[context.call_id] = context
        
        try:
            yield context
        finally:
            # Context will be automatically removed from WeakValueDictionary
            pass


# Global tool manager instance
tool_manager = ToolManager()