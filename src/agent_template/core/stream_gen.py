"""
StreamGen - Real-time response generation with streaming output.

This module provides advanced streaming capabilities for AI model responses
with token-by-token output, real-time processing, and WebSocket streaming.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Callable
from uuid import uuid4

import structlog

from ..config import settings
from .async_queue import StreamQueue, QueueItem


logger = structlog.get_logger(__name__)


class StreamType(str, Enum):
    """Types of streams supported."""
    
    TOKEN = "token"           # Token-by-token streaming
    CHUNK = "chunk"           # Chunk-based streaming  
    DELTA = "delta"           # Delta/diff streaming
    EVENT = "event"           # Event-based streaming
    RAW = "raw"               # Raw data streaming


class StreamStatus(str, Enum):
    """Stream status enumeration."""
    
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamToken:
    """Individual token in a stream."""
    
    content: str
    token_id: Optional[int] = None
    position: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "token_id": self.token_id,
            "position": self.position,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class StreamChunk:
    """Chunk of streamed data."""
    
    data: Union[str, bytes, Dict[str, Any]]
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    sequence: int = 0
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "chunk_id": self.chunk_id,
            "sequence": self.sequence,
            "is_final": self.is_final,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class StreamEvent:
    """Stream event for event-based streaming."""
    
    event_type: str
    data: Any
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process(self, data: Any) -> AsyncGenerator[Any, None]:
        """Process input data and yield stream items."""
        pass
    
    @abstractmethod
    async def finalize(self) -> Optional[Any]:
        """Finalize processing and return any final data."""
        pass


class TokenProcessor(StreamProcessor):
    """Processor for token-based streaming."""
    
    def __init__(self, tokenizer: Optional[Callable] = None):
        self.tokenizer = tokenizer
        self.position = 0
    
    async def process(self, data: str) -> AsyncGenerator[StreamToken, None]:
        """Process text data into tokens."""
        if self.tokenizer:
            # Use custom tokenizer
            tokens = await self._tokenize(data)
            for token in tokens:
                yield StreamToken(
                    content=token,
                    position=self.position,
                )
                self.position += 1
        else:
            # Simple word-based tokenization
            words = data.split()
            for word in words:
                yield StreamToken(
                    content=word + " ",
                    position=self.position,
                )
                self.position += 1
    
    async def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the configured tokenizer."""
        if asyncio.iscoroutinefunction(self.tokenizer):
            return await self.tokenizer(text)
        else:
            return self.tokenizer(text)
    
    async def finalize(self) -> Optional[StreamToken]:
        """Finalize token processing."""
        return None


class ChunkProcessor(StreamProcessor):
    """Processor for chunk-based streaming."""
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.sequence = 0
    
    async def process(self, data: Union[str, bytes]) -> AsyncGenerator[StreamChunk, None]:
        """Process data into chunks."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        for i in range(0, len(data), self.chunk_size):
            chunk_data = data[i:i + self.chunk_size]
            is_final = (i + self.chunk_size) >= len(data)
            
            yield StreamChunk(
                data=chunk_data.decode('utf-8') if isinstance(chunk_data, bytes) else chunk_data,
                sequence=self.sequence,
                is_final=is_final
            )
            self.sequence += 1
    
    async def finalize(self) -> Optional[StreamChunk]:
        """Finalize chunk processing."""
        return StreamChunk(
            data="",
            sequence=self.sequence,
            is_final=True,
            metadata={"type": "end_of_stream"}
        )


class StreamGenerator:
    """
    Core stream generation engine.
    
    Manages multiple concurrent streams with different processing strategies,
    real-time output, and WebSocket integration.
    """
    
    def __init__(self, name: str = "stream_gen"):
        self.name = name
        self.status = StreamStatus.INITIALIZING
        self._streams: Dict[str, "Stream"] = {}
        self._processors: Dict[StreamType, StreamProcessor] = {}
        self._output_queue = StreamQueue(name=f"{name}_output")
        self._active_connections: Dict[str, Any] = {}
        self._metrics = {
            "streams_created": 0,
            "streams_completed": 0,
            "streams_failed": 0,
            "total_tokens": 0,
            "total_chunks": 0,
        }
        
        # Initialize default processors
        self._init_default_processors()
        
        self.status = StreamStatus.ACTIVE
        logger.info(f"StreamGenerator '{name}' initialized")
    
    def _init_default_processors(self):
        """Initialize default stream processors."""
        self._processors[StreamType.TOKEN] = TokenProcessor()
        self._processors[StreamType.CHUNK] = ChunkProcessor(
            chunk_size=settings.agent.stream_chunk_size
        )
    
    def register_processor(self, stream_type: StreamType, processor: StreamProcessor):
        """Register a custom stream processor."""
        self._processors[stream_type] = processor
        logger.info(f"Registered processor for {stream_type}")
    
    async def create_stream(
        self,
        stream_id: str,
        stream_type: StreamType = StreamType.TOKEN,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Stream":
        """Create a new stream."""
        if stream_id in self._streams:
            raise ValueError(f"Stream '{stream_id}' already exists")
        
        stream = Stream(
            stream_id=stream_id,
            stream_type=stream_type,
            generator=self,
            metadata=metadata or {}
        )
        
        self._streams[stream_id] = stream
        self._metrics["streams_created"] += 1
        
        logger.info(f"Created stream '{stream_id}' of type {stream_type}")
        return stream
    
    async def get_stream(self, stream_id: str) -> Optional["Stream"]:
        """Get an existing stream."""
        return self._streams.get(stream_id)
    
    async def close_stream(self, stream_id: str) -> bool:
        """Close and remove a stream."""
        if stream_id not in self._streams:
            return False
        
        stream = self._streams[stream_id]
        await stream.close()
        del self._streams[stream_id]
        
        logger.info(f"Closed stream '{stream_id}'")
        return True
    
    async def process_stream(
        self,
        stream_id: str,
        data: Any
    ) -> AsyncGenerator[Union[StreamToken, StreamChunk, StreamEvent], None]:
        """Process data through a stream and yield results."""
        stream = self._streams.get(stream_id)
        if not stream:
            raise ValueError(f"Stream '{stream_id}' not found")
        
        processor = self._processors.get(stream.stream_type)
        if not processor:
            raise ValueError(f"No processor for stream type {stream.stream_type}")
        
        try:
            stream.status = StreamStatus.ACTIVE
            
            async for item in processor.process(data):
                # Update metrics
                if isinstance(item, StreamToken):
                    self._metrics["total_tokens"] += 1
                elif isinstance(item, StreamChunk):
                    self._metrics["total_chunks"] += 1
                
                # Add to output queue for WebSocket distribution
                await self._output_queue.put_chunk({
                    "stream_id": stream_id,
                    "item": item.to_dict(),
                    "timestamp": time.time()
                })
                
                yield item
            
            # Finalize processing
            final_item = await processor.finalize()
            if final_item:
                yield final_item
            
            stream.status = StreamStatus.COMPLETED
            self._metrics["streams_completed"] += 1
            
        except Exception as e:
            stream.status = StreamStatus.ERROR
            self._metrics["streams_failed"] += 1
            logger.error(f"Error processing stream '{stream_id}'", error=str(e))
            raise
    
    async def stream_to_websocket(
        self,
        websocket: Any,
        stream_id: str,
        data: Any
    ) -> None:
        """Stream data directly to a WebSocket connection."""
        connection_id = str(uuid4())
        self._active_connections[connection_id] = websocket
        
        try:
            async for item in self.process_stream(stream_id, data):
                message = {
                    "stream_id": stream_id,
                    "type": "stream_data",
                    "data": item.to_dict(),
                    "timestamp": time.time()
                }
                
                # Send to WebSocket
                await websocket.send_text(json.dumps(message))
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.001)
            
            # Send completion message
            completion_message = {
                "stream_id": stream_id,
                "type": "stream_complete",
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(completion_message))
            
        except Exception as e:
            error_message = {
                "stream_id": stream_id,
                "type": "stream_error",
                "error": str(e),
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(error_message))
            raise
        finally:
            if connection_id in self._active_connections:
                del self._active_connections[connection_id]
    
    async def broadcast_to_connections(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all active WebSocket connections."""
        if not self._active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection_id, websocket in self._active_connections.items():
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send to connection {connection_id}", error=str(e))
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            del self._active_connections[connection_id]
    
    @asynccontextmanager
    async def stream_context(self, stream_id: str, stream_type: StreamType = StreamType.TOKEN):
        """Context manager for stream lifecycle."""
        stream = await self.create_stream(stream_id, stream_type)
        try:
            yield stream
        finally:
            await self.close_stream(stream_id)
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self._metrics,
            "active_streams": len(self._streams),
            "active_connections": len(self._active_connections),
            "output_queue_size": self._output_queue.qsize()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the stream generator."""
        logger.info(f"Shutting down StreamGenerator '{self.name}'")
        self.status = StreamStatus.COMPLETING
        
        # Close all streams
        for stream_id in list(self._streams.keys()):
            await self.close_stream(stream_id)
        
        # Close output queue
        await self._output_queue.close()
        
        # Clear connections
        self._active_connections.clear()
        
        self.status = StreamStatus.COMPLETED
        logger.info(f"StreamGenerator '{self.name}' shutdown complete")


class Stream:
    """
    Individual stream instance.
    
    Represents a single streaming session with its own state,
    configuration, and processing pipeline.
    """
    
    def __init__(
        self,
        stream_id: str,
        stream_type: StreamType,
        generator: StreamGenerator,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.stream_id = stream_id
        self.stream_type = stream_type
        self.generator = generator
        self.metadata = metadata or {}
        self.status = StreamStatus.INITIALIZING
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        
        # Stream-specific metrics
        self._items_generated = 0
        self._bytes_processed = 0
        
        self.status = StreamStatus.ACTIVE
    
    async def process(self, data: Any) -> AsyncGenerator[Any, None]:
        """Process data through this stream."""
        self.started_at = time.time()
        
        async for item in self.generator.process_stream(self.stream_id, data):
            self._items_generated += 1
            if hasattr(item, 'content') and item.content:
                self._bytes_processed += len(str(item.content).encode('utf-8'))
            yield item
        
        self.completed_at = time.time()
    
    async def close(self) -> None:
        """Close the stream."""
        if self.status not in [StreamStatus.COMPLETED, StreamStatus.CANCELLED]:
            self.status = StreamStatus.CANCELLED
            self.completed_at = time.time()
    
    @property
    def duration(self) -> Optional[float]:
        """Get stream duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get stream metrics."""
        return {
            "stream_id": self.stream_id,
            "stream_type": self.stream_type,
            "status": self.status,
            "duration": self.duration,
            "items_generated": self._items_generated,
            "bytes_processed": self._bytes_processed,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class StreamManager:
    """
    Manager for multiple stream generators.
    
    Provides centralized management, load balancing, and coordination
    across multiple stream generators.
    """
    
    def __init__(self):
        self._generators: Dict[str, StreamGenerator] = {}
        self._default_generator: Optional[StreamGenerator] = None
        self._lock = asyncio.Lock()
    
    async def create_generator(self, name: str) -> StreamGenerator:
        """Create a new stream generator."""
        async with self._lock:
            if name in self._generators:
                raise ValueError(f"Generator '{name}' already exists")
            
            generator = StreamGenerator(name)
            self._generators[name] = generator
            
            # Set as default if it's the first one
            if self._default_generator is None:
                self._default_generator = generator
            
            logger.info(f"Created stream generator '{name}'")
            return generator
    
    async def get_generator(self, name: Optional[str] = None) -> Optional[StreamGenerator]:
        """Get a generator by name, or the default generator."""
        if name is None:
            return self._default_generator
        return self._generators.get(name)
    
    async def create_stream(
        self,
        stream_id: str,
        stream_type: StreamType = StreamType.TOKEN,
        generator_name: Optional[str] = None,
        **kwargs
    ) -> Stream:
        """Create a stream using the specified or default generator."""
        generator = await self.get_generator(generator_name)
        if generator is None:
            # Create default generator if none exists
            generator = await self.create_generator("default")
        
        return await generator.create_stream(stream_id, stream_type, **kwargs)
    
    async def shutdown_all(self) -> None:
        """Shutdown all generators."""
        logger.info("Shutting down all stream generators")
        
        for generator in self._generators.values():
            await generator.shutdown()
        
        self._generators.clear()
        self._default_generator = None
        
        logger.info("All stream generators shut down")
    
    @property
    def global_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all generators."""
        total_metrics = {
            "generators": len(self._generators),
            "total_streams": 0,
            "total_connections": 0,
            "streams_created": 0,
            "streams_completed": 0,
            "streams_failed": 0,
            "total_tokens": 0,
            "total_chunks": 0
        }
        
        for generator in self._generators.values():
            metrics = generator.metrics
            total_metrics["total_streams"] += metrics.get("active_streams", 0)
            total_metrics["total_connections"] += metrics.get("active_connections", 0)
            total_metrics["streams_created"] += metrics.get("streams_created", 0)
            total_metrics["streams_completed"] += metrics.get("streams_completed", 0)
            total_metrics["streams_failed"] += metrics.get("streams_failed", 0)
            total_metrics["total_tokens"] += metrics.get("total_tokens", 0)
            total_metrics["total_chunks"] += metrics.get("total_chunks", 0)
        
        return total_metrics


# Global stream manager instance
stream_manager = StreamManager()