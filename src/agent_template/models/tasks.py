"""Task and state models for the agent system."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task types supported by the agent."""
    
    CHAT = "chat"
    COMPLETION = "completion"
    TOOL_CALL = "tool_call"
    SUBAGENT = "subagent"
    COMPRESSION = "compression"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Task execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class Task(BaseModel):
    """Core task model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Content
    content: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution info
    session_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[int] = None  # seconds
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress tracking
    progress: float = 0.0
    progress_message: Optional[str] = None
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class AgentState(str, Enum):
    """Agent state enumeration."""
    
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class StateSnapshot(BaseModel):
    """Agent state snapshot."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state: AgentState
    
    # Metrics
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    memory_usage: Optional[int] = None  # bytes
    cpu_usage: Optional[float] = None  # percentage
    
    # Context
    session_count: int = 0
    subagent_count: int = 0
    tool_calls: int = 0
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class TaskResult(BaseModel):
    """Task execution result."""
    
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None  # seconds
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "use_enum_values": True
    }


class SubagentRequest(BaseModel):
    """Request to spawn a subagent."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    task: Task
    parent_session_id: str
    timeout: Optional[int] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }


class ToolCall(BaseModel):
    """Tool call representation."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution info
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }