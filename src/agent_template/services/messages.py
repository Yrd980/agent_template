"""
Message management service with session handling, context queues, and caching.

This module provides comprehensive message management including session lifecycle,
context window management, temporary caching, and message persistence.
"""

import asyncio
import json
import time
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from weakref import WeakValueDictionary

import structlog

from ..config import settings
from ..core.async_queue import AsyncQueue, QueueItem
from ..models.messages import (
    Message, Session, Context, ConversationThread, MessageHistory,
    MessageFilter, MessageStats, MessageRole, SessionStatus
)


logger = structlog.get_logger(__name__)


class MessageCache:
    """LRU cache for messages with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if time.time() - self._access_times[key] > self.ttl_seconds:
                await self._remove_item(key)
                return None
            
            # Move to end (mark as recently accessed)
            self._cache.move_to_end(key)
            self._access_times[key] = time.time()
            
            return self._cache[key]['data']
    
    async def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        async with self._lock:
            current_time = time.time()
            
            # Remove oldest items if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                await self._remove_item(oldest_key)
            
            self._cache[key] = {'data': value, 'created_at': current_time}
            self._access_times[key] = current_time
            
            # Move to end
            self._cache.move_to_end(key)
    
    async def remove(self, key: str) -> bool:
        """Remove item from cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_item(key)
                return True
            return False
    
    async def _remove_item(self, key: str) -> None:
        """Internal method to remove an item."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
    
    async def clear_expired(self) -> int:
        """Clear expired items and return count of cleared items."""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, access_time in self._access_times.items()
                if current_time - access_time > self.ttl_seconds
            ]
            
            for key in expired_keys:
                await self._remove_item(key)
            
            return len(expired_keys)
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for access_time in self._access_times.values()
            if current_time - access_time > self.ttl_seconds
        )
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "expired_items": expired_count,
            "ttl_seconds": self.ttl_seconds
        }


class ContextQueue:
    """Context-aware message queue with priority handling."""
    
    def __init__(self, session_id: str, max_context_tokens: int = 8000):
        self.session_id = session_id
        self.max_context_tokens = max_context_tokens
        self._messages: List[Message] = []
        self._current_tokens = 0
        self._lock = asyncio.Lock()
        
        # Context management
        self._compression_threshold = settings.agent.context_compression_threshold
        self._system_context: Optional[str] = None
        self._persistent_data: Dict[str, Any] = {}
    
    async def add_message(self, message: Message) -> bool:
        """
        Add message to the context queue.
        
        Returns:
            bool: True if message was added, False if compression needed
        """
        async with self._lock:
            message_tokens = message.estimate_tokens()
            
            # Check if we need compression
            if (self._current_tokens + message_tokens) > (
                self.max_context_tokens * self._compression_threshold
            ):
                return False
            
            self._messages.append(message)
            self._current_tokens += message_tokens
            
            logger.debug(
                f"Message added to context queue",
                session_id=self.session_id,
                message_id=message.id,
                current_tokens=self._current_tokens,
                max_tokens=self.max_context_tokens
            )
            
            return True
    
    async def get_context_messages(self, max_tokens: Optional[int] = None) -> List[Message]:
        """Get messages that fit within the context window."""
        async with self._lock:
            if max_tokens is None:
                max_tokens = self.max_context_tokens
            
            # Start from the end (most recent messages)
            context_messages = []
            token_count = 0
            
            # Add system message if present
            if self._system_context:
                system_msg = Message(
                    role=MessageRole.SYSTEM,
                    content=self._system_context,
                    session_id=self.session_id
                )
                context_messages.append(system_msg)
                token_count += system_msg.estimate_tokens()
            
            # Add messages from newest to oldest until we hit the limit
            for message in reversed(self._messages):
                message_tokens = message.estimate_tokens()
                if token_count + message_tokens > max_tokens:
                    break
                
                context_messages.insert(-len(context_messages) if self._system_context else 0, message)
                token_count += message_tokens
            
            return context_messages
    
    async def compress_old_messages(self, keep_count: int = 10) -> List[Message]:
        """
        Remove old messages for compression, keeping the most recent ones.
        
        Returns:
            List of removed messages
        """
        async with self._lock:
            if len(self._messages) <= keep_count:
                return []
            
            # Remove older messages
            removed_messages = self._messages[:-keep_count]
            self._messages = self._messages[-keep_count:]
            
            # Recalculate token count
            self._current_tokens = sum(msg.estimate_tokens() for msg in self._messages)
            
            logger.info(
                f"Compressed context queue",
                session_id=self.session_id,
                removed_count=len(removed_messages),
                remaining_count=len(self._messages),
                current_tokens=self._current_tokens
            )
            
            return removed_messages
    
    async def set_system_context(self, context: str) -> None:
        """Set system context for the session."""
        async with self._lock:
            self._system_context = context
    
    async def update_persistent_data(self, key: str, value: Any) -> None:
        """Update persistent data for the session."""
        async with self._lock:
            self._persistent_data[key] = value
    
    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        return self._current_tokens
    
    @property
    def message_count(self) -> int:
        """Get message count."""
        return len(self._messages)
    
    @property
    def needs_compression(self) -> bool:
        """Check if compression is needed."""
        return self._current_tokens >= (self.max_context_tokens * self._compression_threshold)


class SessionManager:
    """Manages conversation sessions with lifecycle and cleanup."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._context_queues: Dict[str, ContextQueue] = {}
        self._session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self._stats: Dict[str, MessageStats] = {}
        
        logger.info("SessionManager initialized")
    
    async def start(self) -> None:
        """Start the session manager."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager started")
    
    async def stop(self) -> None:
        """Stop the session manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SessionManager stopped")
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Session:
        """Create a new session."""
        session = Session(
            id=session_id or str(uuid.uuid4()),
            config=config or {},
            user_id=user_id
        )
        
        async with self._session_locks[session.id]:
            if session.id in self._sessions:
                raise ValueError(f"Session {session.id} already exists")
            
            self._sessions[session.id] = session
            self._context_queues[session.id] = ContextQueue(
                session.id,
                max_context_tokens=session.config.get(
                    'context_window', 
                    settings.agent.max_context_length
                )
            )
            self._stats[session.id] = MessageStats(session_id=session.id)
            
            # Set expiry
            session.extend_expiry(hours=24)
        
        logger.info(f"Created session {session.id}", user_id=user_id)
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        session = self._sessions.get(session_id)
        
        if session and session.is_expired():
            await self.close_session(session_id)
            return None
        
        return session
    
    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session."""
        async with self._session_locks[session_id]:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            session.status = SessionStatus.COMPLETED
            
            # Cleanup
            del self._sessions[session_id]
            if session_id in self._context_queues:
                del self._context_queues[session_id]
            if session_id in self._stats:
                del self._stats[session_id]
        
        logger.info(f"Closed session {session_id}")
        return True
    
    async def add_message(self, session_id: str, message: Message) -> bool:
        """
        Add a message to a session.
        
        Returns:
            bool: True if message was added, False if compression needed
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Update session stats
        session.add_message_stats(message)
        self._stats[session_id].update_with_message(message)
        
        # Add to context queue
        context_queue = self._context_queues[session_id]
        return await context_queue.add_message(message)
    
    async def get_context_messages(
        self, 
        session_id: str, 
        max_tokens: Optional[int] = None
    ) -> List[Message]:
        """Get context messages for a session."""
        if session_id not in self._context_queues:
            return []
        
        context_queue = self._context_queues[session_id]
        return await context_queue.get_context_messages(max_tokens)
    
    async def compress_session_context(self, session_id: str) -> List[Message]:
        """
        Compress session context by removing old messages.
        
        Returns:
            List of removed messages for archival
        """
        if session_id not in self._context_queues:
            return []
        
        context_queue = self._context_queues[session_id]
        return await context_queue.compress_old_messages()
    
    async def get_session_stats(self, session_id: str) -> Optional[MessageStats]:
        """Get statistics for a session."""
        return self._stats.get(session_id)
    
    async def list_sessions(
        self, 
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None
    ) -> List[Session]:
        """List sessions with optional filtering."""
        sessions = []
        
        for session in self._sessions.values():
            # Filter by user_id
            if user_id and session.user_id != user_id:
                continue
            
            # Filter by status
            if status and session.status != status:
                continue
            
            # Skip expired sessions
            if session.is_expired():
                continue
            
            sessions.append(session)
        
        return sessions
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired sessions."""
        while self._running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(60)  # Retry in 1 minute on error
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        active_sessions = len(self._sessions)
        total_messages = sum(stats.user_messages + stats.assistant_messages 
                           for stats in self._stats.values())
        total_tokens = sum(stats.total_tokens for stats in self._stats.values())
        
        return {
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "average_messages_per_session": total_messages / max(1, active_sessions),
            "average_tokens_per_session": total_tokens / max(1, active_sessions)
        }


class MessageService:
    """
    Comprehensive message management service.
    
    Provides high-level interface for message handling, session management,
    caching, and context management.
    """
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.message_cache = MessageCache(
            max_size=1000,
            ttl_seconds=3600  # 1 hour
        )
        
        # Message processing queue
        self.processing_queue = AsyncQueue(name="message_processing")
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        logger.info("MessageService initialized")
    
    async def start(self) -> None:
        """Start the message service."""
        if self._running:
            return
        
        self._running = True
        await self.session_manager.start()
        
        # Start background tasks
        self._background_tasks.add(
            asyncio.create_task(self._cache_cleanup_loop())
        )
        self._background_tasks.add(
            asyncio.create_task(self._message_processor())
        )
        
        logger.info("MessageService started")
    
    async def stop(self) -> None:
        """Stop the message service."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop session manager
        await self.session_manager.stop()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        # Close processing queue
        await self.processing_queue.close()
        
        logger.info("MessageService stopped")
    
    # Session management
    async def create_session(self, **kwargs) -> Session:
        """Create a new session."""
        return await self.session_manager.create_session(**kwargs)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session."""
        return await self.session_manager.get_session(session_id)
    
    async def close_session(self, session_id: str) -> bool:
        """Close a session."""
        return await self.session_manager.close_session(session_id)
    
    # Message handling
    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: Union[str, List[Dict[str, Any]]],
        **kwargs
    ) -> Message:
        """Add a message to a session."""
        message = Message(
            role=role,
            content=content,
            session_id=session_id,
            **kwargs
        )
        
        # Try to add to session
        added = await self.session_manager.add_message(session_id, message)
        
        if not added:
            # Context is full, need compression
            await self._handle_context_compression(session_id)
            # Retry adding the message
            added = await self.session_manager.add_message(session_id, message)
            
            if not added:
                raise RuntimeError(f"Failed to add message to session {session_id}")
        
        # Cache the message
        await self.message_cache.put(message.id, message)
        
        # Queue for background processing
        await self.processing_queue.put(message)
        
        return message
    
    async def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        return await self.message_cache.get(message_id)
    
    async def get_context_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None
    ) -> List[Message]:
        """Get context messages for a session."""
        return await self.session_manager.get_context_messages(session_id, max_tokens)
    
    async def search_messages(
        self,
        filter_criteria: MessageFilter
    ) -> List[Message]:
        """Search messages with filter criteria."""
        # This would typically query a database
        # For now, return empty list as placeholder
        logger.info("Message search requested", criteria=filter_criteria.dict())
        return []
    
    # Context management
    async def compress_context(self, session_id: str) -> bool:
        """Compress context for a session."""
        return await self._handle_context_compression(session_id)
    
    async def _handle_context_compression(self, session_id: str) -> bool:
        """Handle context compression for a session."""
        try:
            # Get old messages for compression
            old_messages = await self.session_manager.compress_session_context(session_id)
            
            if old_messages:
                logger.info(
                    f"Compressed context for session {session_id}",
                    compressed_messages=len(old_messages)
                )
                
                # TODO: Create compressed summary using AI model
                # This would typically summarize the old messages
                # and store the summary for later retrieval
            
            return True
            
        except Exception as e:
            logger.error(f"Context compression failed for session {session_id}", error=str(e))
            return False
    
    # Background processing
    async def _message_processor(self) -> None:
        """Background message processor."""
        while self._running:
            try:
                # Get message from queue
                queue_item = await self.processing_queue.get(timeout=1.0)
                if queue_item is None:
                    continue
                
                message = queue_item.data
                
                # Process message (e.g., extract metadata, analyze content)
                await self._process_message(message)
                
                # Mark task as done
                await self.processing_queue.task_done(queue_item, success=True)
                
            except Exception as e:
                logger.error("Error in message processor", error=str(e))
                if 'queue_item' in locals():
                    await self.processing_queue.task_done(queue_item, success=False)
    
    async def _process_message(self, message: Message) -> None:
        """Process a single message."""
        # Placeholder for message processing logic
        # Could include: sentiment analysis, entity extraction,
        # content classification, etc.
        logger.debug(f"Processing message {message.id}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Periodic cache cleanup."""
        while self._running:
            try:
                cleared = await self.message_cache.clear_expired()
                if cleared > 0:
                    logger.debug(f"Cleared {cleared} expired cache items")
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error("Error in cache cleanup", error=str(e))
                await asyncio.sleep(60)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "session_manager": self.session_manager.stats,
            "message_cache": self.message_cache.stats,
            "processing_queue": {
                "size": self.processing_queue.qsize(),
                "processing": self.processing_queue.processing_count()
            },
            "running": self._running,
            "background_tasks": len(self._background_tasks)
        }