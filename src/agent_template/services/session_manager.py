"""Session management for multi-user conversations and context."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog

from ..config import settings
from ..models.messages import Message, MessageType, Session, Context
from .state_cache import StateCache


logger = structlog.get_logger(__name__)


class SessionManager:
    """
    Manages user sessions, context preservation, and multi-user state.
    
    Provides session lifecycle management, context switching,
    and persistent storage of conversation history.
    """
    
    def __init__(self, state_cache: Optional[StateCache] = None):
        self.state_cache = state_cache or StateCache()
        self._sessions: Dict[str, Session] = {}
        self._active_contexts: Dict[str, Context] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.session_timeout = timedelta(hours=24)  # Default session timeout
        self.max_sessions = 1000  # Maximum concurrent sessions
        self.cleanup_interval = 3600  # Cleanup every hour
        
        # Metrics
        self._metrics = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "messages_processed": 0,
            "context_switches": 0,
        }
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("SessionManager initialized")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session."""
        if len(self._sessions) >= self.max_sessions:
            await self._cleanup_expired_sessions(force=True)
            
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("Maximum number of sessions reached")
        
        session_id = str(uuid4())
        context = Context(
            session_id=session_id,
            conversation_history=[],
            variables={},
            active_tools=set()
        )
        session = Session(
            id=session_id,
            user_id=user_id,
            metadata=session_metadata or {},
            context=context
        )
        
        # Store session
        self._sessions[session_id] = session
        self._session_locks[session_id] = asyncio.Lock()
        self._active_contexts[session_id] = session.context
        
        # Persist to cache
        await self.state_cache.put(f"session:{session_id}", session.model_dump())
        
        self._metrics["sessions_created"] += 1
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        # Check memory first
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_activity = datetime.utcnow()
            return session
        
        # Try to load from cache
        cached_data = await self.state_cache.get(f"session:{session_id}")
        if cached_data:
            session = Session(**cached_data)
            self._sessions[session_id] = session
            self._session_locks[session_id] = asyncio.Lock()
            self._active_contexts[session_id] = session.context
            
            logger.info(f"Loaded session {session_id} from cache")
            return session
        
        return None
    
    async def update_session(self, session: Session) -> None:
        """Update a session."""
        session.last_activity = datetime.utcnow()
        self._sessions[session.id] = session
        
        # Persist to cache
        await self.state_cache.put(f"session:{session.id}", session.model_dump())
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id not in self._sessions:
            return False
        
        # Remove from memory
        del self._sessions[session_id]
        if session_id in self._session_locks:
            del self._session_locks[session_id]
        if session_id in self._active_contexts:
            del self._active_contexts[session_id]
        
        # Remove from cache
        await self.state_cache.remove(f"session:{session_id}")
        
        logger.info(f"Deleted session {session_id}")
        return True
    
    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Session]:
        """List sessions, optionally filtered by user."""
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        # Sort by last activity (most recent first)
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        
        return sessions[offset:offset + limit]
    
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        async with self._session_locks[session_id]:
            session.messages.append(message)
            session.context.conversation_history.append({
                "role": message.message_type,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            })
            
            await self.update_session(session)
            self._metrics["messages_processed"] += 1
    
    async def get_context(self, session_id: str) -> Optional[Context]:
        """Get the context for a session."""
        session = await self.get_session(session_id)
        return session.context if session else None
    
    async def update_context(self, session_id: str, context: Context) -> None:
        """Update the context for a session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        async with self._session_locks[session_id]:
            session.context = context
            self._active_contexts[session_id] = context
            await self.update_session(session)
            self._metrics["context_switches"] += 1
    
    async def set_context_variable(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set a context variable for a session."""
        context = await self.get_context(session_id)
        if not context:
            raise ValueError(f"Session {session_id} not found")
        
        context.variables[key] = value
        await self.update_context(session_id, context)
    
    async def get_context_variable(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a context variable from a session."""
        context = await self.get_context(session_id)
        if not context:
            return default
        
        return context.variables.get(key, default)
    
    async def add_system_message(self, session_id: str, message: str) -> None:
        """Add a system message to session context."""
        context = await self.get_context(session_id)
        if not context:
            raise ValueError(f"Session {session_id} not found")
        
        context.system_messages.append({
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        await self.update_context(session_id, context)
    
    async def set_user_preference(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set a user preference for a session."""
        context = await self.get_context(session_id)
        if not context:
            raise ValueError(f"Session {session_id} not found")
        
        context.user_preferences[key] = value
        await self.update_context(session_id, context)
    
    async def activate_tool(self, session_id: str, tool_name: str) -> None:
        """Activate a tool for a session."""
        context = await self.get_context(session_id)
        if not context:
            raise ValueError(f"Session {session_id} not found")
        
        context.active_tools.add(tool_name)
        await self.update_context(session_id, context)
    
    async def deactivate_tool(self, session_id: str, tool_name: str) -> None:
        """Deactivate a tool for a session."""
        context = await self.get_context(session_id)
        if not context:
            return
        
        context.active_tools.discard(tool_name)
        await self.update_context(session_id, context)
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        now = datetime.utcnow()
        duration = (now - session.created_at).total_seconds()
        
        message_counts = {}
        for message in session.messages:
            msg_type = message.message_type.value
            message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "duration_seconds": duration,
            "total_messages": len(session.messages),
            "message_counts": message_counts,
            "context_variables": len(session.context.variables),
            "active_tools": len(session.context.active_tools),
            "last_activity": session.last_activity.isoformat()
        }
    
    async def _cleanup_expired_sessions(self, force: bool = False) -> None:
        """Clean up expired sessions."""
        while True:
            try:
                now = datetime.utcnow()
                expired_sessions = []
                
                for session_id, session in self._sessions.items():
                    age = now - session.last_activity
                    if age > self.session_timeout or force:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.delete_session(session_id)
                    self._metrics["sessions_expired"] += 1
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                if force:
                    break
                
                # Wait for next cleanup interval
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error during session cleanup", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get session manager metrics."""
        return {
            **self._metrics,
            "active_sessions": len(self._sessions),
            "active_contexts": len(self._active_contexts),
        }
    
    async def cleanup_expired_sessions(self) -> None:
        """Public method to trigger session cleanup."""
        await self._cleanup_expired_sessions(force=True)
    
    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        logger.info("Shutting down SessionManager")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Persist all active sessions
        for session in self._sessions.values():
            await self.state_cache.put(f"session:{session.id}", session.model_dump())
        
        logger.info("SessionManager shutdown complete")
