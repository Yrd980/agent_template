"""Message and session models for the agent system."""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Message roles in the conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Legacy support


class MessageType(str, Enum):
    """Message types for different content formats."""
    
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_EVENT = "system_event"


class SessionStatus(str, Enum):
    """Session status enumeration."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class Message(BaseModel):
    """Core message model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]
    message_type: MessageType = MessageType.TEXT
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Content metadata
    tokens: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Tool-related
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing info
    processing_time: Optional[float] = None  # seconds
    compressed: bool = False
    compression_ratio: Optional[float] = None
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('content')
    def validate_content(cls, v, values):
        """Validate content based on message type."""
        message_type = values.get('message_type')
        
        if message_type == MessageType.TEXT and not isinstance(v, str):
            # Allow list format for complex content
            if not isinstance(v, (list, dict)):
                raise ValueError("Text messages must have string, list, or dict content")
        
        return v
    
    def get_text_content(self) -> str:
        """Extract text content from various content formats."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Extract text from list format (e.g., OpenAI format)
            text_parts = []
            for item in self.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            return '\n'.join(text_parts)
        elif isinstance(self.content, dict):
            return self.content.get('text', str(self.content))
        
        return str(self.content)
    
    def estimate_tokens(self) -> int:
        """Rough token estimation for the message."""
        if self.tokens:
            return self.tokens
        
        text = self.get_text_content()
        # Rough estimation: ~4 characters per token
        return max(1, len(text) // 4)


class Session(BaseModel):
    """Conversation session model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Status and timing
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    model_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    message_count: int = 0
    total_tokens: int = 0
    max_tokens: Optional[int] = None
    
    # Context management
    context_window: int = 8000  # Default context window size
    compression_enabled: bool = True
    
    # Metadata
    user_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    def is_expired(self) -> bool:
        """Check if the session is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def extend_expiry(self, hours: int = 24) -> None:
        """Extend session expiry time."""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.updated_at = datetime.utcnow()
    
    def add_message_stats(self, message: Message) -> None:
        """Update session stats when a message is added."""
        self.message_count += 1
        self.total_tokens += message.estimate_tokens()
        self.updated_at = datetime.utcnow()


class Context(BaseModel):
    """Context window management."""
    
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    
    # Context management
    max_tokens: int = 8000
    current_tokens: int = 0
    compression_threshold: float = 0.8  # Compress when 80% full
    
    # System context
    system_message: Optional[str] = None
    persistent_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    def add_message(self, message: Message) -> bool:
        """
        Add a message to the context.
        
        Returns:
            bool: True if message was added, False if context is full
        """
        message_tokens = message.estimate_tokens()
        
        # Check if we need compression
        if (self.current_tokens + message_tokens) > (self.max_tokens * self.compression_threshold):
            return False
        
        self.messages.append(message)
        self.current_tokens += message_tokens
        self.updated_at = datetime.utcnow()
        return True
    
    def get_context_messages(self, include_system: bool = True) -> List[Message]:
        """Get messages for sending to AI model."""
        messages = []
        
        # Add system message if specified
        if include_system and self.system_message:
            system_msg = Message(
                role=MessageRole.SYSTEM,
                content=self.system_message,
                session_id=self.session_id,
                message_type=MessageType.SYSTEM_EVENT
            )
            messages.append(system_msg)
        
        # Add conversation messages
        messages.extend(self.messages)
        
        return messages
    
    def needs_compression(self) -> bool:
        """Check if context needs compression."""
        return self.current_tokens >= (self.max_tokens * self.compression_threshold)
    
    def clear_old_messages(self, keep_count: int = 10) -> List[Message]:
        """
        Clear old messages, keeping the most recent ones.
        
        Returns:
            List of removed messages
        """
        if len(self.messages) <= keep_count:
            return []
        
        removed_messages = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]
        
        # Recalculate token count
        self.current_tokens = sum(msg.estimate_tokens() for msg in self.messages)
        self.updated_at = datetime.utcnow()
        
        return removed_messages


class ConversationThread(BaseModel):
    """Thread within a session for branching conversations."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    parent_message_id: Optional[str] = None
    title: Optional[str] = None
    
    # Messages in this thread
    messages: List[Message] = Field(default_factory=list)
    
    # Status
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MessageHistory(BaseModel):
    """Historical record of messages with compression."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    
    # Original messages
    original_messages: List[Message] = Field(default_factory=list)
    
    # Compressed summary
    summary: str
    key_points: List[str] = Field(default_factory=list)
    important_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Compression metadata
    compression_ratio: float
    original_tokens: int
    compressed_tokens: int
    compression_method: str
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MessageFilter(BaseModel):
    """Filter criteria for message queries."""
    
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    role: Optional[MessageRole] = None
    message_type: Optional[MessageType] = None
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Content filters
    content_contains: Optional[str] = None
    has_tool_calls: Optional[bool] = None
    
    # Metadata filters
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Pagination
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"
    descending: bool = True
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MessageStats(BaseModel):
    """Message statistics for sessions and contexts."""
    
    session_id: str
    
    # Message counts by role
    user_messages: int = 0
    assistant_messages: int = 0
    system_messages: int = 0
    tool_messages: int = 0
    
    # Token statistics
    total_tokens: int = 0
    average_tokens_per_message: float = 0.0
    max_tokens_in_message: int = 0
    
    # Timing statistics
    average_response_time: float = 0.0  # seconds
    fastest_response: float = 0.0
    slowest_response: float = 0.0
    
    # Compression statistics
    compressed_messages: int = 0
    total_compression_ratio: float = 0.0
    
    # Time range
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    def update_with_message(self, message: Message) -> None:
        """Update statistics with a new message."""
        # Update counts
        if message.role == MessageRole.USER:
            self.user_messages += 1
        elif message.role == MessageRole.ASSISTANT:
            self.assistant_messages += 1
        elif message.role == MessageRole.SYSTEM:
            self.system_messages += 1
        elif message.role == MessageRole.TOOL:
            self.tool_messages += 1
        
        # Update token stats
        message_tokens = message.estimate_tokens()
        self.total_tokens += message_tokens
        self.max_tokens_in_message = max(self.max_tokens_in_message, message_tokens)
        
        total_messages = (
            self.user_messages + self.assistant_messages + 
            self.system_messages + self.tool_messages
        )
        self.average_tokens_per_message = self.total_tokens / max(1, total_messages)
        
        # Update compression stats
        if message.compressed:
            self.compressed_messages += 1
            if message.compression_ratio:
                self.total_compression_ratio += message.compression_ratio
        
        # Update timing
        if message.processing_time:
            if self.average_response_time == 0:
                self.average_response_time = message.processing_time
                self.fastest_response = message.processing_time
                self.slowest_response = message.processing_time
            else:
                # Simple moving average (could be improved)
                self.average_response_time = (
                    self.average_response_time + message.processing_time
                ) / 2
                self.fastest_response = min(self.fastest_response, message.processing_time)
                self.slowest_response = max(self.slowest_response, message.processing_time)
        
        # Update time range
        if self.first_message_at is None or message.timestamp < self.first_message_at:
            self.first_message_at = message.timestamp
        
        if self.last_message_at is None or message.timestamp > self.last_message_at:
            self.last_message_at = message.timestamp