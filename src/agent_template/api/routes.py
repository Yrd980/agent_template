"""FastAPI routes and endpoints for the agent API."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.agent_loop import AgentLoop
from ..models.messages import Message, MessageType, MessageRole
from ..core.stream_gen import StreamGenerator
from ..models.tasks import Task, TaskStatus, TaskType
from ..services.session_manager import SessionManager
from ..services.tool_manager import ToolManager
from ..services.state_cache import StateCache


logger = logging.getLogger(__name__)

# Request/Response models
class MessageRequest(BaseModel):
    content: str
    session_id: Optional[str] = None
    role: MessageRole = MessageRole.USER
    metadata: Dict[str, Any] = {}


class MessageResponse(BaseModel):
    id: str
    type: MessageType
    content: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class SessionRequest(BaseModel):
    metadata: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    metadata: Dict[str, Any] = {}


class TaskRequest(BaseModel):
    type: TaskType
    content: Dict[str, Any]
    session_id: Optional[str] = None
    priority: int = 2
    timeout: Optional[int] = None


class TaskResponse(BaseModel):
    id: str
    type: TaskType
    status: TaskStatus
    progress: float
    created_at: datetime
    session_id: Optional[str] = None


class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}
    session_id: Optional[str] = None


class StatsResponse(BaseModel):
    total_sessions: int
    active_tasks: int
    total_messages: int
    model_stats: Dict[str, Any]
    tool_stats: Dict[str, Any]


# Dependency injection
class APIDependencies:
    """Container for API dependencies."""
    
    def __init__(
        self,
        agent_loop: AgentLoop,
        session_manager: SessionManager,
        tool_manager: ToolManager,
        state_cache: StateCache
    ):
        self.agent_loop = agent_loop
        self.session_manager = session_manager
        self.tool_manager = tool_manager
        self.state_cache = state_cache


# Global dependencies (to be set by the server)
_dependencies: Optional[APIDependencies] = None


def get_dependencies() -> APIDependencies:
    """Get API dependencies."""
    if _dependencies is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return _dependencies


def set_dependencies(deps: APIDependencies):
    """Set API dependencies."""
    global _dependencies
    _dependencies = deps


# Create router
router = APIRouter(prefix="/api/v1", tags=["Agent API"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@router.get("/stats", response_model=StatsResponse)
async def get_stats(deps: APIDependencies = Depends(get_dependencies)):
    """Get system statistics."""
    try:
        # Get session stats
        session_stats = await deps.session_manager.get_stats()
        
        # Get task stats  
        task_stats = await deps.agent_loop.get_stats()
        
        # Get tool stats
        tool_stats = deps.tool_manager.get_stats()
        
        # Get model stats (placeholder)
        model_stats = {"requests": 0, "tokens": 0, "errors": 0}
        
        return StatsResponse(
            total_sessions=session_stats.get("total_sessions", 0),
            active_tasks=task_stats.get("active_tasks", 0),
            total_messages=session_stats.get("total_messages", 0),
            model_stats=model_stats,
            tool_stats=tool_stats
        )
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Session endpoints
@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Create a new session."""
    try:
        session = await deps.session_manager.create_session(metadata=request.metadata)
        
        return SessionResponse(
            id=session.id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages),
            metadata=session.metadata
        )
    
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    deps: APIDependencies = Depends(get_dependencies)
):
    """List recent sessions."""
    try:
        sessions = await deps.session_manager.list_sessions(limit=limit)
        
        return [
            SessionResponse(
                id=session.id,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=len(session.messages),
                metadata=session.metadata
            )
            for session in sessions
        ]
    
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Get session details."""
    try:
        session = await deps.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return SessionResponse(
            id=session.id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages),
            metadata=session.metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Delete a session."""
    try:
        success = await deps.session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return {"message": f"Session {session_id} deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Message endpoints
@router.post("/messages", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Send a message to the agent."""
    try:
        # Create session if not provided
        if not request.session_id:
            session = await deps.session_manager.create_session()
            session_id = session.id
        else:
            session_id = request.session_id
            # Verify session exists
            session = await deps.session_manager.get_session(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found"
                )
        
        # Create message
        message = Message(
            type=request.message_type,
            content=request.content,
            session_id=session_id,
            metadata=request.metadata
        )
        
        # Add to session
        await deps.session_manager.add_message(session_id, message)
        
        # Process with agent loop
        await deps.agent_loop.process_message(message)
        
        return MessageResponse(
            id=message.id,
            type=message.type,
            content=message.content,
            session_id=message.session_id,
            timestamp=message.timestamp,
            metadata=message.metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    deps: APIDependencies = Depends(get_dependencies)
):
    """Get messages from a session."""
    try:
        messages = await deps.session_manager.get_messages(session_id, limit=limit)
        
        return [
            MessageResponse(
                id=msg.id,
                type=msg.type,
                content=msg.content,
                session_id=msg.session_id,
                timestamp=msg.timestamp,
                metadata=msg.metadata
            )
            for msg in messages
        ]
    
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/messages/stream")
async def stream_message(
    request: MessageRequest,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Send a message and get streaming response."""
    try:
        # Create session if not provided
        if not request.session_id:
            session = await deps.session_manager.create_session()
            session_id = session.id
        else:
            session_id = request.session_id
        
        # Create message
        message = Message(
            type=request.message_type,
            content=request.content,
            session_id=session_id,
            metadata=request.metadata
        )
        
        # Add to session
        await deps.session_manager.add_message(session_id, message)
        
        # Create stream generator
        stream_gen = StreamGenerator()
        
        # Start streaming response
        async def generate_stream():
            try:
                # Process message
                await deps.agent_loop.process_message(message)
                
                # Stream the response
                async for chunk in stream_gen.stream_response(message.id):
                    yield f"data: {chunk}\n\n"
                
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Task endpoints
@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Create a new task."""
    try:
        task = await deps.agent_loop.create_task(
            task_type=request.type,
            content=request.content,
            session_id=request.session_id,
            priority=request.priority,
            timeout=request.timeout
        )
        
        return TaskResponse(
            id=task.id,
            type=task.type,
            status=task.status,
            progress=task.progress,
            created_at=task.created_at,
            session_id=task.session_id
        )
    
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    session_id: Optional[str] = Query(None),
    status: Optional[TaskStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    deps: APIDependencies = Depends(get_dependencies)
):
    """List tasks."""
    try:
        if session_id:
            tasks = await deps.agent_loop.get_session_tasks(session_id)
        else:
            tasks = await deps.agent_loop.get_all_tasks(limit=limit)
        
        # Filter by status if provided
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        return [
            TaskResponse(
                id=task.id,
                type=task.type,
                status=task.status,
                progress=task.progress,
                created_at=task.created_at,
                session_id=task.session_id
            )
            for task in tasks[:limit]
        ]
    
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Get task details."""
    try:
        task = await deps.agent_loop.get_task(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskResponse(
            id=task.id,
            type=task.type,
            status=task.status,
            progress=task.progress,
            created_at=task.created_at,
            session_id=task.session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Cancel a task."""
    try:
        success = await deps.agent_loop.cancel_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return {"message": f"Task {task_id} cancelled"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Tool endpoints
@router.get("/tools")
async def list_tools(deps: APIDependencies = Depends(get_dependencies)):
    """List available tools."""
    try:
        tools = deps.tool_manager.list_tools()
        return {"tools": [tool.model_dump() for tool in tools]}
    
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/tools/call")
async def call_tool(
    request: ToolCallRequest,
    deps: APIDependencies = Depends(get_dependencies)
):
    """Call a tool."""
    try:
        result = await deps.tool_manager.call_tool(
            tool_name=request.tool_name,
            parameters=request.parameters,
            session_id=request.session_id or "api_call"
        )
        
        return {"result": result.model_dump()}
    
    except Exception as e:
        logger.error(f"Failed to call tool {request.tool_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Cache endpoints
@router.get("/cache/stats")
async def get_cache_stats(deps: APIDependencies = Depends(get_dependencies)):
    """Get cache statistics."""
    try:
        stats = await deps.state_cache.get_stats()
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/cache")
async def clear_cache(deps: APIDependencies = Depends(get_dependencies)):
    """Clear the cache."""
    try:
        await deps.state_cache.clear()
        return {"message": "Cache cleared"}
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )