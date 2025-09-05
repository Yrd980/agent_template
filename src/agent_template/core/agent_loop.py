"""
AgentLoop - Main orchestration engine with task scheduling and state management.

This module provides the core event loop that manages task execution,
state transitions, and coordination between different components.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, AsyncGenerator
from weakref import WeakSet

import structlog

from ..config import settings
from ..models.tasks import (
    Task, TaskStatus, TaskPriority, TaskType, TaskResult, 
    AgentState, StateSnapshot
)


logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Priority-based task scheduler with dependency resolution."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._pending_tasks: List[Task] = []
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: Task) -> None:
        """Add a task to the scheduler."""
        async with self._lock:
            self._pending_tasks.append(task)
            
            # Track dependencies
            if task.dependencies:
                self._task_dependencies[task.id] = set(task.dependencies)
            
            # Sort by priority
            self._pending_tasks.sort(
                key=lambda t: (t.priority.value, t.created_at), 
                reverse=True
            )
            
            logger.info("Task added to scheduler", task_id=task.id, task_type=task.type)
    
    async def get_next_task(self) -> Optional[Task]:
        """Get the next task that's ready to execute."""
        async with self._lock:
            if len(self._running_tasks) >= self.max_concurrent:
                return None
            
            for i, task in enumerate(self._pending_tasks):
                # Check if dependencies are satisfied
                if await self._dependencies_satisfied(task):
                    # Remove from pending and add to running
                    self._pending_tasks.pop(i)
                    self._running_tasks[task.id] = task
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.utcnow()
                    return task
            
            return None
    
    async def complete_task(self, task_id: str, result: TaskResult) -> None:
        """Mark a task as completed."""
        async with self._lock:
            if task_id in self._running_tasks:
                task = self._running_tasks.pop(task_id)
                task.status = result.status
                task.completed_at = datetime.utcnow()
                task.result = result.result
                task.error = result.error
                
                self._completed_tasks[task_id] = result
                
                # Clean up dependencies
                if task_id in self._task_dependencies:
                    del self._task_dependencies[task_id]
                
                logger.info("Task completed", task_id=task_id, status=result.status)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        async with self._lock:
            # Check pending tasks
            for i, task in enumerate(self._pending_tasks):
                if task.id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self._pending_tasks.pop(i)
                    logger.info("Pending task cancelled", task_id=task_id)
                    return True
            
            # Check running tasks
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                logger.info("Running task marked for cancellation", task_id=task_id)
                return True
            
            return False
    
    async def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if a task's dependencies are satisfied."""
        if task.id not in self._task_dependencies:
            return True
        
        dependencies = self._task_dependencies[task.id]
        for dep_id in dependencies:
            if dep_id not in self._completed_tasks:
                return False
            
            # Check if dependency completed successfully
            result = self._completed_tasks[dep_id]
            if result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "pending": len(self._pending_tasks),
            "running": len(self._running_tasks),
            "completed": len(self._completed_tasks),
            "max_concurrent": self.max_concurrent
        }


class AgentLoop:
    """
    Main agent orchestration loop.
    
    Manages task execution, state transitions, and coordinates between
    different components of the agent system.
    """
    
    def __init__(self):
        self.state = AgentState.IDLE
        self.scheduler = TaskScheduler(max_concurrent=settings.agent.max_concurrent_tasks)
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state_snapshots: List[StateSnapshot] = []
        self._sessions: WeakSet = WeakSet()
        
        # Metrics
        self._start_time: Optional[datetime] = None
        self._task_count = 0
        self._error_count = 0
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(self.shutdown())
    
    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    async def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error("Error in event handler", event=event, error=str(e))
    
    async def create_task(self, task_data: Dict[str, Any]) -> Task:
        """Create and schedule a new task."""
        task = Task(**task_data)
        await self.scheduler.add_task(task)
        await self.emit("task_created", task)
        return task
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        success = await self.scheduler.cancel_task(task_id)
        if success:
            await self.emit("task_cancelled", task_id)
        return success
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        # Check running tasks
        if task_id in self.scheduler._running_tasks:
            return self.scheduler._running_tasks[task_id].status
        
        # Check completed tasks
        if task_id in self.scheduler._completed_tasks:
            return self.scheduler._completed_tasks[task_id].status
        
        # Check pending tasks
        for task in self.scheduler._pending_tasks:
            if task.id == task_id:
                return task.status
        
        return None
    
    async def run(self) -> None:
        """Start the agent loop."""
        if self.running:
            logger.warning("Agent loop is already running")
            return
        
        self.running = True
        self.state = AgentState.INITIALIZING
        self._start_time = datetime.utcnow()
        
        logger.info("Starting agent loop")
        await self.emit("agent_starting")
        
        try:
            # Start background tasks
            self._tasks.add(asyncio.create_task(self._main_loop()))
            self._tasks.add(asyncio.create_task(self._state_monitor()))
            self._tasks.add(asyncio.create_task(self._health_checker()))
            
            self.state = AgentState.RUNNING
            await self.emit("agent_started")
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error("Error in agent loop", error=str(e))
            self.state = AgentState.ERROR
            self._error_count += 1
            await self.emit("agent_error", str(e))
        finally:
            await self._cleanup()
    
    async def shutdown(self) -> None:
        """Shutdown the agent loop gracefully."""
        if not self.running:
            return
        
        logger.info("Shutting down agent loop")
        self.state = AgentState.SHUTTING_DOWN
        await self.emit("agent_shutting_down")
        
        self._shutdown_event.set()
    
    async def _main_loop(self) -> None:
        """Main task processing loop."""
        while self.running:
            try:
                # Get next task
                task = await self.scheduler.get_next_task()
                if task is None:
                    await asyncio.sleep(0.1)  # Brief pause if no tasks
                    continue
                
                # Execute task
                self.state = AgentState.PROCESSING
                result = await self._execute_task(task)
                await self.scheduler.complete_task(task.id, result)
                
                self._task_count += 1
                await self.emit("task_completed", {"task": task, "result": result})
                
                if self.scheduler.stats["running"] == 0:
                    self.state = AgentState.RUNNING
                
            except Exception as e:
                logger.error("Error in main loop", error=str(e))
                self._error_count += 1
                await asyncio.sleep(1)  # Prevent tight error loops
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = datetime.utcnow()
        
        try:
            logger.info("Executing task", task_id=task.id, task_type=task.type)
            await self.emit("task_started", task)
            
            # Task execution logic would be implemented by handlers
            result_data = await self._dispatch_task(task)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            logger.warning("Task timed out", task_id=task.id)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error="Task execution timed out"
            )
        except Exception as e:
            logger.error("Task execution failed", task_id=task.id, error=str(e))
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    async def _dispatch_task(self, task: Task) -> Dict[str, Any]:
        """Dispatch task to appropriate handler."""
        # This would be implemented by specific task handlers
        # For now, just emit an event for external handlers
        result = {}
        await self.emit(f"execute_{task.type.value}", {"task": task, "result": result})
        return result
    
    async def _state_monitor(self) -> None:
        """Monitor and record agent state."""
        while self.running:
            try:
                snapshot = StateSnapshot(
                    state=self.state,
                    active_tasks=len(self.scheduler._running_tasks),
                    completed_tasks=self._task_count,
                    failed_tasks=self._error_count,
                    session_count=len(self._sessions),
                    metadata={
                        "scheduler_stats": self.scheduler.stats,
                        "uptime": (
                            datetime.utcnow() - self._start_time
                        ).total_seconds() if self._start_time else 0
                    }
                )
                
                self._state_snapshots.append(snapshot)
                
                # Keep only last 100 snapshots
                if len(self._state_snapshots) > 100:
                    self._state_snapshots = self._state_snapshots[-100:]
                
                await self.emit("state_snapshot", snapshot)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error("Error in state monitor", error=str(e))
                await asyncio.sleep(5)
    
    async def _health_checker(self) -> None:
        """Periodic health checks."""
        while self.running:
            try:
                # Basic health checks
                health_status = {
                    "running": self.running,
                    "state": self.state,
                    "task_count": self._task_count,
                    "error_count": self._error_count,
                    "scheduler_stats": self.scheduler.stats
                }
                
                await self.emit("health_check", health_status)
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error("Error in health checker", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up agent loop")
        self.running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self.state = AgentState.IDLE
        await self.emit("agent_stopped")
        logger.info("Agent loop cleanup completed")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "state": self.state,
            "running": self.running,
            "uptime": (
                datetime.utcnow() - self._start_time
            ).total_seconds() if self._start_time else 0,
            "task_count": self._task_count,
            "error_count": self._error_count,
            "scheduler": self.scheduler.stats,
            "sessions": len(self._sessions)
        }
    
    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncGenerator[None, None]:
        """Context manager for session tracking."""
        class SessionTracker:
            def __init__(self, session_id: str):
                self.session_id = session_id
        
        tracker = SessionTracker(session_id)
        self._sessions.add(tracker)
        try:
            yield
        finally:
            # WeakSet will automatically clean up when tracker goes out of scope
            pass