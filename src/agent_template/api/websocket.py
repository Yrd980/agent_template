"""WebSocket communication layer for real-time agent interaction."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from ..core.agent_loop import AgentLoop
from ..models.messages import Message, MessageType, MessageRole, Session
from ..core.stream_gen import StreamGenerator
from ..models.tasks import Task, TaskStatus
from ..services.session_manager import SessionManager


logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a WebSocket connection with state management."""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.session_id: Optional[str] = None
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_alive = True
        
        # Subscriptions
        self.subscribed_sessions: Set[str] = set()
        self.subscribed_tasks: Set[str] = set()
        
        # Client info
        self.client_info: Dict[str, Any] = {}
    
    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Send JSON data to the client."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(data)
                self.last_activity = datetime.utcnow()
                return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            self.is_alive = False
        return False
    
    async def send_text(self, message: str) -> bool:
        """Send text message to the client."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_text(message)
                self.last_activity = datetime.utcnow()
                return True
        except Exception as e:
            logger.error(f"Failed to send text to {self.connection_id}: {e}")
            self.is_alive = False
        return False
    
    async def close(self, code: int = 1000, reason: str = "Connection closed"):
        """Close the WebSocket connection."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.error(f"Failed to close connection {self.connection_id}: {e}")
        finally:
            self.is_alive = False


class WebSocketManager:
    """Manages WebSocket connections and message routing."""
    
    def __init__(self, agent_loop: AgentLoop, session_manager: SessionManager):
        self.agent_loop = agent_loop
        self.session_manager = session_manager
        self.connections: Dict[str, WebSocketConnection] = {}
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        
        # Stream generators for each connection
        self.stream_generators: Dict[str, StreamGenerator] = {}
        
        # Event handlers
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Set up event handlers for agent loop events."""
        # Task events
        self.agent_loop.add_event_handler("task_created", self._on_task_created)
        self.agent_loop.add_event_handler("task_updated", self._on_task_updated)
        self.agent_loop.add_event_handler("task_completed", self._on_task_completed)
        
        # Message events  
        self.agent_loop.add_event_handler("message_created", self._on_message_created)
        self.agent_loop.add_event_handler("stream_chunk", self._on_stream_chunk)
        
        # Session events
        self.agent_loop.add_event_handler("session_created", self._on_session_created)
        self.agent_loop.add_event_handler("session_updated", self._on_session_updated)
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id)
        self.connections[connection_id] = connection
        
        # Create stream generator for this connection
        self.stream_generators[connection_id] = StreamGenerator()
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send connection info
        await connection.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect and clean up a WebSocket connection."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Remove from session subscriptions
            if connection.session_id:
                self._unsubscribe_from_session(connection_id, connection.session_id)
            
            # Clean up stream generator
            if connection_id in self.stream_generators:
                del self.stream_generators[connection_id]
            
            # Remove connection
            del self.connections[connection_id]
            
            logger.info(f"WebSocket connection disconnected: {connection_id}")
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "message":
                await self._handle_user_message(connection, data)
            elif msg_type == "create_session":
                await self._handle_create_session(connection, data)
            elif msg_type == "join_session":
                await self._handle_join_session(connection, data)
            elif msg_type == "subscribe_tasks":
                await self._handle_subscribe_tasks(connection, data)
            elif msg_type == "get_session_history":
                await self._handle_get_session_history(connection, data)
            elif msg_type == "ping":
                await self._handle_ping(connection)
            else:
                await connection.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
        
        except json.JSONDecodeError:
            await connection.send_json({
                "type": "error", 
                "message": "Invalid JSON format"
            })
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await connection.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def _handle_user_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle user message."""
        message_data = data.get("data", {})
        
        # Create message object
        message = Message(
            type=MessageType.USER,
            content=message_data.get("content", ""),
            session_id=connection.session_id or "default",
            metadata=message_data.get("metadata", {})
        )
        
        # If no session, create one
        if not connection.session_id:
            session = await self.session_manager.create_session()
            connection.session_id = session.id
            self._subscribe_to_session(connection.connection_id, session.id)
            
            await connection.send_json({
                "type": "session_created",
                "session_id": session.id
            })
        
        # Add message to session
        await self.session_manager.add_message(connection.session_id, message)
        
        # Submit to agent loop for processing
        await self.agent_loop.process_message(message)
        
        # Start streaming response if available
        stream_gen = self.stream_generators.get(connection.connection_id)
        if stream_gen:
            await stream_gen.start_stream(message.id)
    
    async def _handle_create_session(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle create session request."""
        session = await self.session_manager.create_session()
        connection.session_id = session.id
        self._subscribe_to_session(connection.connection_id, session.id)
        
        await connection.send_json({
            "type": "session_created",
            "session_id": session.id,
            "data": session.model_dump()
        })
    
    async def _handle_join_session(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle join session request."""
        session_id = data.get("session_id")
        if not session_id:
            await connection.send_json({
                "type": "error",
                "message": "session_id required"
            })
            return
        
        # Get session
        session = await self.session_manager.get_session(session_id)
        if not session:
            await connection.send_json({
                "type": "error", 
                "message": f"Session {session_id} not found"
            })
            return
        
        # Unsubscribe from current session
        if connection.session_id:
            self._unsubscribe_from_session(connection.connection_id, connection.session_id)
        
        # Subscribe to new session
        connection.session_id = session_id
        self._subscribe_to_session(connection.connection_id, session_id)
        
        await connection.send_json({
            "type": "session_joined",
            "session_id": session_id,
            "data": session.model_dump()
        })
    
    async def _handle_subscribe_tasks(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle task subscription request."""
        session_id = data.get("session_id") or connection.session_id
        if session_id:
            # Send current tasks
            tasks = await self.agent_loop.get_session_tasks(session_id)
            await connection.send_json({
                "type": "tasks_update",
                "session_id": session_id,
                "data": [task.model_dump() for task in tasks]
            })
    
    async def _handle_get_session_history(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle get session history request."""
        session_id = data.get("session_id") or connection.session_id
        limit = data.get("limit", 50)
        
        if not session_id:
            await connection.send_json({
                "type": "error",
                "message": "session_id required"
            })
            return
        
        # Get messages from session
        messages = await self.session_manager.get_messages(session_id, limit=limit)
        
        await connection.send_json({
            "type": "session_history",
            "session_id": session_id,
            "data": [msg.model_dump() for msg in messages]
        })
    
    async def _handle_ping(self, connection: WebSocketConnection):
        """Handle ping request."""
        await connection.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _subscribe_to_session(self, connection_id: str, session_id: str):
        """Subscribe connection to session events."""
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(connection_id)
        
        # Add to connection's subscriptions
        connection = self.connections.get(connection_id)
        if connection:
            connection.subscribed_sessions.add(session_id)
    
    def _unsubscribe_from_session(self, connection_id: str, session_id: str):
        """Unsubscribe connection from session events."""
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        # Remove from connection's subscriptions
        connection = self.connections.get(connection_id)
        if connection:
            connection.subscribed_sessions.discard(session_id)
    
    async def broadcast_to_session(self, session_id: str, data: Dict[str, Any]):
        """Broadcast message to all connections in a session."""
        connection_ids = self.session_connections.get(session_id, set())
        
        # Send to all connections in parallel
        tasks = []
        for connection_id in list(connection_ids):  # Copy to avoid modification during iteration
            connection = self.connections.get(connection_id)
            if connection and connection.is_alive:
                tasks.append(connection.send_json(data))
            else:
                # Clean up dead connection
                self._unsubscribe_from_session(connection_id, session_id)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast message to all connections."""
        tasks = []
        for connection in list(self.connections.values()):
            if connection.is_alive:
                tasks.append(connection.send_json(data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    # Event handlers
    async def _on_task_created(self, task: Task):
        """Handle task created event."""
        if task.session_id:
            await self.broadcast_to_session(task.session_id, {
                "type": "task_created",
                "data": task.model_dump()
            })
    
    async def _on_task_updated(self, task: Task):
        """Handle task updated event."""
        if task.session_id:
            await self.broadcast_to_session(task.session_id, {
                "type": "task_updated", 
                "data": task.model_dump()
            })
    
    async def _on_task_completed(self, task: Task):
        """Handle task completed event."""
        if task.session_id:
            await self.broadcast_to_session(task.session_id, {
                "type": "task_completed",
                "data": task.model_dump()
            })
    
    async def _on_message_created(self, message: Message):
        """Handle message created event."""
        if message.session_id:
            await self.broadcast_to_session(message.session_id, {
                "type": "message",
                "data": message.model_dump()
            })
    
    async def _on_stream_chunk(self, chunk_data: Dict[str, Any]):
        """Handle stream chunk event."""
        session_id = chunk_data.get("session_id")
        if session_id:
            await self.broadcast_to_session(session_id, {
                "type": "stream_chunk",
                "data": chunk_data
            })
    
    async def _on_session_created(self, session: Session):
        """Handle session created event."""
        await self.broadcast_to_all({
            "type": "session_created",
            "data": session.model_dump()
        })
    
    async def _on_session_updated(self, session: Session):
        """Handle session updated event."""
        await self.broadcast_to_session(session.id, {
            "type": "session_updated",
            "data": session.model_dump()
        })
    
    async def cleanup_dead_connections(self):
        """Clean up dead or inactive connections."""
        current_time = datetime.utcnow()
        dead_connections = []
        
        for connection_id, connection in self.connections.items():
            # Check if connection is dead or inactive for too long
            if (not connection.is_alive or 
                (current_time - connection.last_activity).total_seconds() > 3600):  # 1 hour timeout
                dead_connections.append(connection_id)
        
        for connection_id in dead_connections:
            await self.disconnect(connection_id)
    
    async def send_heartbeat(self):
        """Send heartbeat to all connections."""
        heartbeat_data = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "connections": len(self.connections)
        }
        await self.broadcast_to_all(heartbeat_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self.connections),
            "active_sessions": len(self.session_connections),
            "stream_generators": len(self.stream_generators),
            "connections_by_session": {
                session_id: len(connection_ids)
                for session_id, connection_ids in self.session_connections.items()
            }
        }