"""Tool and MCP (Model Context Protocol) models."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field, field_validator


class ToolType(str, Enum):
    """Types of tools available."""
    
    FUNCTION = "function"
    MCP_TOOL = "mcp_tool"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"
    SUBAGENT = "subagent"


class ToolStatus(str, Enum):
    """Tool execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class MCPServerStatus(str, Enum):
    """MCP server connection status."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class ToolParameter(BaseModel):
    """Tool parameter definition."""
    
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None
    
    # Validation constraints
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


class ToolDefinition(BaseModel):
    """Tool definition and metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    tool_type: ToolType = ToolType.FUNCTION
    
    # Parameters
    parameters: List[ToolParameter] = Field(default_factory=list)
    return_type: Optional[str] = None
    
    # Execution info
    timeout: int = 60  # seconds
    max_retries: int = 3
    
    # MCP-specific
    server_name: Optional[str] = None
    server_url: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_def = {
                "type": param.type,
                "description": param.description or ""
            }
            
            if param.enum:
                prop_def["enum"] = param.enum
            if param.minimum is not None:
                prop_def["minimum"] = param.minimum
            if param.maximum is not None:
                prop_def["maximum"] = param.maximum
            
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_def = {
                "type": param.type,
                "description": param.description or ""
            }
            
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class ToolCall(BaseModel):
    """Tool call request."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    status: ToolStatus = ToolStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Context
    session_id: str
    message_id: Optional[str] = None
    parent_call_id: Optional[str] = None  # For nested tool calls
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Metadata
    execution_time: Optional[float] = None  # seconds
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ToolResult(BaseModel):
    """Tool execution result."""
    
    call_id: str
    tool_name: str
    status: ToolStatus
    
    # Results
    data: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    error: Optional[str] = None
    
    # Execution info
    execution_time: float
    retry_count: int = 0
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class MCPServer(BaseModel):
    """MCP server configuration and status."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    url: str
    status: MCPServerStatus = MCPServerStatus.DISCONNECTED
    
    # Configuration
    timeout: int = 30
    max_retries: int = 3
    retry_interval: int = 5  # seconds
    
    # Authentication
    auth_type: Optional[str] = None  # "bearer", "api_key", etc.
    auth_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Available tools
    tools: List[ToolDefinition] = Field(default_factory=list)
    
    # Connection info
    connected_at: Optional[datetime] = None
    last_ping: Optional[datetime] = None
    error: Optional[str] = None
    
    # Statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Metadata
    version: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy."""
        return (
            self.status == MCPServerStatus.CONNECTED and
            self.success_rate >= 0.8 and  # 80% success rate threshold
            (self.last_ping is None or 
             (datetime.utcnow() - self.last_ping).total_seconds() < 300)  # 5 min threshold
        )


class ToolRegistry(BaseModel):
    """Registry of available tools."""
    
    tools: Dict[str, ToolDefinition] = Field(default_factory=dict)
    mcp_servers: Dict[str, MCPServer] = Field(default_factory=dict)
    
    # Statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    def add_tool(self, tool: ToolDefinition) -> None:
        """Add a tool to the registry."""
        self.tools[tool.name] = tool
        self.updated_at = datetime.utcnow()
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get a tool definition."""
        return self.tools.get(tool_name)
    
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[ToolDefinition]:
        """List tools, optionally filtered by type."""
        tools = list(self.tools.values())
        if tool_type:
            tools = [tool for tool in tools if tool.tool_type == tool_type]
        return tools
    
    def add_mcp_server(self, server: MCPServer) -> None:
        """Add an MCP server to the registry."""
        self.mcp_servers[server.name] = server
        self.updated_at = datetime.utcnow()
    
    def remove_mcp_server(self, server_name: str) -> bool:
        """Remove an MCP server from the registry."""
        if server_name in self.mcp_servers:
            del self.mcp_servers[server_name]
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_mcp_server(self, server_name: str) -> Optional[MCPServer]:
        """Get an MCP server."""
        return self.mcp_servers.get(server_name)
    
    def list_mcp_servers(self, status: Optional[MCPServerStatus] = None) -> List[MCPServer]:
        """List MCP servers, optionally filtered by status."""
        servers = list(self.mcp_servers.values())
        if status:
            servers = [server for server in servers if server.status == status]
        return servers
    
    def update_call_stats(self, success: bool) -> None:
        """Update call statistics."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.updated_at = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self.tools),
            "mcp_servers": len(self.mcp_servers),
            "connected_servers": len([
                s for s in self.mcp_servers.values() 
                if s.status == MCPServerStatus.CONNECTED
            ]),
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "tools_by_type": {
                tool_type.value: len([
                    t for t in self.tools.values() 
                    if t.tool_type == tool_type
                ])
                for tool_type in ToolType
            }
        }


class ToolDiscovery(BaseModel):
    """Tool discovery configuration and results."""
    
    enabled: bool = True
    discovery_interval: int = 300  # seconds (5 minutes)
    
    # Discovery sources
    mcp_servers: List[str] = Field(default_factory=list)
    local_paths: List[str] = Field(default_factory=list)
    remote_registries: List[str] = Field(default_factory=list)
    
    # Last discovery
    last_discovery: Optional[datetime] = None
    discovered_tools: int = 0
    discovery_errors: List[str] = Field(default_factory=list)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class ToolExecutionContext(BaseModel):
    """Context for tool execution."""
    
    call_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Execution environment
    environment: Dict[str, Any] = Field(default_factory=dict)
    permissions: Dict[str, bool] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    
    # Parent context (for nested calls)
    parent_context: Optional[str] = None
    depth: int = 0
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    @property
    def is_expired(self) -> bool:
        """Check if context is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission."""
        return self.permissions.get(permission, False)


class ToolMetrics(BaseModel):
    """Tool execution metrics."""
    
    tool_name: str
    
    # Call statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    
    # Timing statistics
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    
    # Recent performance
    recent_calls: List[Dict[str, Any]] = Field(default_factory=list)  # Last 100 calls
    
    # Time tracking
    first_call: Optional[datetime] = None
    last_call: Optional[datetime] = None
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    def record_call(
        self,
        success: bool,
        execution_time: float,
        error: Optional[str] = None
    ) -> None:
        """Record a tool call."""
        now = datetime.utcnow()
        
        # Update counters
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error and "timeout" in error.lower():
                self.timeout_calls += 1
        
        # Update timing
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_calls
        
        if self.min_execution_time == 0 or execution_time < self.min_execution_time:
            self.min_execution_time = execution_time
        
        if execution_time > self.max_execution_time:
            self.max_execution_time = execution_time
        
        # Update time tracking
        if self.first_call is None:
            self.first_call = now
        self.last_call = now
        
        # Add to recent calls (keep last 100)
        self.recent_calls.append({
            "timestamp": now.isoformat(),
            "success": success,
            "execution_time": execution_time,
            "error": error
        })
        
        if len(self.recent_calls) > 100:
            self.recent_calls = self.recent_calls[-100:]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def recent_success_rate(self) -> float:
        """Calculate recent success rate (last 20 calls)."""
        recent_subset = self.recent_calls[-20:] if len(self.recent_calls) >= 20 else self.recent_calls
        if not recent_subset:
            return 0.0
        
        successful = sum(1 for call in recent_subset if call["success"])
        return successful / len(recent_subset)
    
    @property
    def is_healthy(self) -> bool:
        """Check if tool is performing well."""
        return (
            self.success_rate >= 0.8 and  # 80% overall success rate
            self.recent_success_rate >= 0.7 and  # 70% recent success rate
            self.average_execution_time < 60.0  # Average under 1 minute
        )
