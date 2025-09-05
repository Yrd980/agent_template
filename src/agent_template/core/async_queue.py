"""
AsyncQueue - Async communication pipeline with stream handling and flow control.

This module provides advanced queue systems for handling async communication
with backpressure, flow control, and cancellation support.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generic, List, Optional, Set, TypeVar, Union
from weakref import WeakSet

import structlog

from ..config import settings


logger = structlog.get_logger(__name__)

T = TypeVar('T')


class QueueState(str, Enum):
    """Queue state enumeration."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    CLOSED = "closed"


class FlowControlStrategy(str, Enum):
    """Flow control strategies."""
    
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    BACKPRESSURE = "backpressure"


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    
    total_items: int = 0
    processed_items: int = 0
    dropped_items: int = 0
    current_size: int = 0
    max_size_reached: int = 0
    average_wait_time: float = 0.0
    peak_processing_time: float = 0.0
    errors: int = 0
    
    def reset(self) -> None:
        """Reset metrics counters."""
        self.__dict__.update(QueueMetrics().__dict__)


class QueueItem(Generic[T]):
    """Wrapper for queue items with metadata."""
    
    def __init__(self, data: T, priority: int = 0, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.id = f"{id(self)}_{self.created_at}"
        self.attempts = 0
        self.last_error: Optional[str] = None
    
    def __lt__(self, other: 'QueueItem') -> bool:
        """Compare items by priority (higher priority first)."""
        return self.priority > other.priority


class AsyncQueue(Generic[T]):
    """
    Advanced async queue with flow control and backpressure.
    
    Features:
    - Priority-based ordering
    - Backpressure handling
    - Flow control strategies
    - Metrics collection
    - Graceful shutdown
    - Error handling and retries
    """
    
    def __init__(
        self,
        maxsize: int = 0,
        flow_control: FlowControlStrategy = FlowControlStrategy.BACKPRESSURE,
        name: str = "async_queue"
    ):
        self.maxsize = maxsize
        self.flow_control = flow_control
        self.name = name
        self.state = QueueState.ACTIVE
        
        # Core queue structures
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        self._processing: Set[str] = set()
        self._completed: Set[str] = set()
        
        # Flow control
        self._consumers: WeakSet = WeakSet()
        self._backpressure_event = asyncio.Event()
        self._backpressure_event.set()  # Initially no backpressure
        
        # Metrics
        self.metrics = QueueMetrics()
        self._wait_times: deque = deque(maxlen=100)
        self._processing_times: deque = deque(maxlen=100)
        
        # Shutdown
        self._shutdown_event = asyncio.Event()
        self._drain_complete = asyncio.Event()
        
        # Locks
        self._state_lock = asyncio.Lock()
        
        logger.info(f"AsyncQueue '{name}' initialized", maxsize=maxsize, flow_control=flow_control)
    
    async def put(self, item: T, priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Put an item in the queue.
        
        Returns:
            bool: True if item was added, False if dropped or rejected
        """
        if self.state == QueueState.CLOSED:
            logger.warning(f"Attempted to put item in closed queue '{self.name}'")
            return False
        
        queue_item = QueueItem(item, priority, metadata)
        start_time = time.time()
        
        try:
            # Handle backpressure
            if not await self._handle_backpressure(queue_item):
                return False
            
            # Try to put the item
            if self.maxsize == 0:  # Unlimited queue
                await self._queue.put(queue_item)
            else:
                try:
                    self._queue.put_nowait(queue_item)
                except asyncio.QueueFull:
                    if not await self._handle_full_queue(queue_item):
                        return False
            
            # Update metrics
            self.metrics.total_items += 1
            self.metrics.current_size = self._queue.qsize()
            self.metrics.max_size_reached = max(
                self.metrics.max_size_reached, 
                self.metrics.current_size
            )
            
            wait_time = time.time() - start_time
            self._wait_times.append(wait_time)
            self.metrics.average_wait_time = sum(self._wait_times) / len(self._wait_times)
            
            logger.debug(f"Item added to queue '{self.name}'", 
                        item_id=queue_item.id, priority=priority)
            return True
            
        except Exception as e:
            logger.error(f"Error putting item in queue '{self.name}'", error=str(e))
            self.metrics.errors += 1
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueItem[T]]:
        """
        Get an item from the queue.
        
        Args:
            timeout: Maximum time to wait for an item
            
        Returns:
            QueueItem or None if timeout or closed
        """
        if self.state == QueueState.CLOSED:
            return None
        
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                item = await self._queue.get()
            
            # Mark as processing
            self._processing.add(item.id)
            self.metrics.current_size = self._queue.qsize()
            
            logger.debug(f"Item retrieved from queue '{self.name}'", item_id=item.id)
            return item
            
        except asyncio.TimeoutError:
            logger.debug(f"Queue get timeout in '{self.name}'")
            return None
        except Exception as e:
            logger.error(f"Error getting item from queue '{self.name}'", error=str(e))
            self.metrics.errors += 1
            return None
    
    async def task_done(self, item: QueueItem[T], success: bool = True) -> None:
        """
        Mark a task as done.
        
        Args:
            item: The completed queue item
            success: Whether the task completed successfully
        """
        if item.id in self._processing:
            self._processing.remove(item.id)
            
            if success:
                self._completed.add(item.id)
                self.metrics.processed_items += 1
                
                # Calculate processing time
                processing_time = time.time() - item.created_at
                self._processing_times.append(processing_time)
                self.metrics.peak_processing_time = max(
                    self.metrics.peak_processing_time,
                    processing_time
                )
            else:
                self.metrics.errors += 1
            
            # Signal queue task done (for asyncio.Queue compatibility)
            self._queue.task_done()
            
            # Check if draining is complete
            if (self.state == QueueState.DRAINING and 
                self._queue.empty() and 
                not self._processing):
                self._drain_complete.set()
            
            logger.debug(f"Task marked as done in queue '{self.name}'", 
                        item_id=item.id, success=success)
    
    async def _handle_backpressure(self, item: QueueItem[T]) -> bool:
        """Handle backpressure scenarios."""
        if self.flow_control != FlowControlStrategy.BACKPRESSURE:
            return True
        
        # Wait for backpressure to clear
        await self._backpressure_event.wait()
        return True
    
    async def _handle_full_queue(self, item: QueueItem[T]) -> bool:
        """Handle full queue scenarios based on flow control strategy."""
        if self.flow_control == FlowControlStrategy.DROP_NEWEST:
            logger.debug(f"Dropping newest item in queue '{self.name}'", item_id=item.id)
            self.metrics.dropped_items += 1
            return False
        
        elif self.flow_control == FlowControlStrategy.DROP_OLDEST:
            try:
                # Remove oldest item (lowest priority)
                old_item = self._queue.get_nowait()
                logger.debug(f"Dropped oldest item in queue '{self.name}'", 
                           old_item_id=old_item.id)
                self.metrics.dropped_items += 1
                await self._queue.put(item)
                return True
            except asyncio.QueueEmpty:
                return False
        
        elif self.flow_control == FlowControlStrategy.BLOCK:
            # Block until space is available
            await self._queue.put(item)
            return True
        
        elif self.flow_control == FlowControlStrategy.BACKPRESSURE:
            # Apply backpressure
            self._backpressure_event.clear()
            logger.warning(f"Backpressure applied to queue '{self.name}'")
            return False
        
        return False
    
    async def pause(self) -> None:
        """Pause the queue (stop processing new items)."""
        async with self._state_lock:
            if self.state == QueueState.ACTIVE:
                self.state = QueueState.PAUSED
                logger.info(f"Queue '{self.name}' paused")
    
    async def resume(self) -> None:
        """Resume the queue."""
        async with self._state_lock:
            if self.state == QueueState.PAUSED:
                self.state = QueueState.ACTIVE
                self._backpressure_event.set()  # Clear any backpressure
                logger.info(f"Queue '{self.name}' resumed")
    
    async def drain(self, timeout: Optional[float] = None) -> bool:
        """
        Drain the queue (process remaining items but don't accept new ones).
        
        Returns:
            bool: True if drain completed, False if timeout
        """
        async with self._state_lock:
            if self.state in [QueueState.DRAINING, QueueState.CLOSED]:
                return True
            
            self.state = QueueState.DRAINING
            logger.info(f"Draining queue '{self.name}'")
        
        try:
            if timeout is not None:
                await asyncio.wait_for(self._drain_complete.wait(), timeout=timeout)
            else:
                await self._drain_complete.wait()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timeout for '{self.name}'")
            return False
    
    async def close(self) -> None:
        """Close the queue and reject any new items."""
        async with self._state_lock:
            if self.state != QueueState.CLOSED:
                self.state = QueueState.CLOSED
                self._shutdown_event.set()
                logger.info(f"Queue '{self.name}' closed")
    
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty() and not self._processing
    
    def qsize(self) -> int:
        """Get the current queue size."""
        return self._queue.qsize()
    
    def processing_count(self) -> int:
        """Get the number of items currently being processed."""
        return len(self._processing)
    
    @property
    def is_healthy(self) -> bool:
        """Check if the queue is in a healthy state."""
        return (
            self.state in [QueueState.ACTIVE, QueueState.PAUSED] and
            self.metrics.errors < 10  # Arbitrary threshold
        )


class StreamQueue(AsyncQueue[T]):
    """
    Specialized queue for streaming data with automatic flow control.
    
    Features additional stream-specific functionality like chunking,
    buffering, and automatic backpressure based on consumer speed.
    """
    
    def __init__(
        self,
        maxsize: int = 1000,
        chunk_size: int = 1024,
        buffer_timeout: float = 0.1,
        **kwargs
    ):
        super().__init__(maxsize=maxsize, **kwargs)
        self.chunk_size = chunk_size
        self.buffer_timeout = buffer_timeout
        self._buffer: List[T] = []
        self._buffer_lock = asyncio.Lock()
        self._last_flush = time.time()
    
    async def put_chunk(self, chunk: T) -> bool:
        """Put a chunk of data, potentially buffering it."""
        async with self._buffer_lock:
            self._buffer.append(chunk)
            
            # Flush buffer if conditions are met
            should_flush = (
                len(self._buffer) >= self.chunk_size or
                time.time() - self._last_flush >= self.buffer_timeout
            )
            
            if should_flush:
                return await self._flush_buffer()
            
            return True
    
    async def _flush_buffer(self) -> bool:
        """Flush the internal buffer to the queue."""
        if not self._buffer:
            return True
        
        # Create a combined chunk
        combined_chunk = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        
        return await self.put(combined_chunk)
    
    async def force_flush(self) -> bool:
        """Force flush the buffer."""
        async with self._buffer_lock:
            return await self._flush_buffer()
    
    @asynccontextmanager
    async def stream_consumer(self, consumer_id: str):
        """Context manager for stream consumers with automatic cleanup."""
        class StreamConsumer:
            def __init__(self, queue: StreamQueue, consumer_id: str):
                self.queue = queue
                self.consumer_id = consumer_id
                self.processed = 0
                self.start_time = time.time()
        
        consumer = StreamConsumer(self, consumer_id)
        self._consumers.add(consumer)
        
        try:
            logger.info(f"Stream consumer '{consumer_id}' connected to queue '{self.name}'")
            yield consumer
        finally:
            # Consumer cleanup is automatic via WeakSet
            duration = time.time() - consumer.start_time
            logger.info(f"Stream consumer '{consumer_id}' disconnected", 
                       processed=consumer.processed, duration=duration)


class QueueManager:
    """
    Manager for multiple async queues with load balancing and coordination.
    """
    
    def __init__(self):
        self._queues: Dict[str, AsyncQueue] = {}
        self._round_robin_index: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self.metrics: Dict[str, QueueMetrics] = {}
    
    async def create_queue(
        self, 
        name: str, 
        queue_type: str = "standard",
        **kwargs
    ) -> AsyncQueue:
        """Create a new queue."""
        async with self._lock:
            if name in self._queues:
                raise ValueError(f"Queue '{name}' already exists")
            
            if queue_type == "stream":
                queue = StreamQueue(name=name, **kwargs)
            else:
                queue = AsyncQueue(name=name, **kwargs)
            
            self._queues[name] = queue
            self.metrics[name] = queue.metrics
            
            logger.info(f"Created {queue_type} queue '{name}'")
            return queue
    
    async def get_queue(self, name: str) -> Optional[AsyncQueue]:
        """Get a queue by name."""
        return self._queues.get(name)
    
    async def delete_queue(self, name: str) -> bool:
        """Delete a queue."""
        async with self._lock:
            if name not in self._queues:
                return False
            
            queue = self._queues.pop(name)
            await queue.close()
            
            if name in self.metrics:
                del self.metrics[name]
            
            logger.info(f"Deleted queue '{name}'")
            return True
    
    async def load_balance_put(
        self, 
        queue_pattern: str, 
        item: Any, 
        **kwargs
    ) -> Optional[str]:
        """
        Put an item using load balancing across matching queues.
        
        Returns the name of the queue that accepted the item.
        """
        matching_queues = [
            name for name in self._queues.keys() 
            if queue_pattern in name and self._queues[name].is_healthy
        ]
        
        if not matching_queues:
            return None
        
        # Round-robin load balancing
        index = self._round_robin_index[queue_pattern] % len(matching_queues)
        queue_name = matching_queues[index]
        self._round_robin_index[queue_pattern] += 1
        
        queue = self._queues[queue_name]
        success = await queue.put(item, **kwargs)
        
        return queue_name if success else None
    
    async def shutdown_all(self, drain_timeout: float = 30.0) -> None:
        """Shutdown all managed queues."""
        logger.info("Shutting down all queues")
        
        # First drain all queues
        drain_tasks = [
            queue.drain(timeout=drain_timeout) 
            for queue in self._queues.values()
        ]
        
        await asyncio.gather(*drain_tasks, return_exceptions=True)
        
        # Then close all queues
        for queue in self._queues.values():
            await queue.close()
        
        logger.info("All queues shut down")
    
    @property
    def health_status(self) -> Dict[str, Any]:
        """Get health status of all queues."""
        return {
            name: {
                "healthy": queue.is_healthy,
                "state": queue.state,
                "size": queue.qsize(),
                "processing": queue.processing_count(),
                "metrics": {
                    "total_items": queue.metrics.total_items,
                    "processed_items": queue.metrics.processed_items,
                    "dropped_items": queue.metrics.dropped_items,
                    "errors": queue.metrics.errors
                }
            }
            for name, queue in self._queues.items()
        }