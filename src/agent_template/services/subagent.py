"""
Subagent process management system.

This module provides capabilities for spawning, managing, and coordinating
specialized child agents for complex multi-step tasks.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from weakref import WeakSet

import structlog

from ..config import settings
from ..core.async_queue import AsyncQueue, StreamQueue
from ..models.tasks import Task, TaskStatus, TaskType, TaskResult
from ..models.messages import Message, MessageRole


logger = structlog.get_logger(__name__)


class SubagentType(str, Enum):
    """Types of subagents available."""
    
    GENERAL = "general"           # General-purpose agent
    RESEARCHER = "researcher"     # Research and analysis focused
    CODER = "coder"              # Code generation and analysis
    WRITER = "writer"            # Content writing and editing
    PLANNER = "planner"          # Task planning and coordination
    VALIDATOR = "validator"      # Validation and quality checks
    SPECIALIST = "specialist"    # Domain-specific specialist
    CUSTOM = "custom"            # Custom agent implementation


class SubagentStatus(str, Enum):
    """Subagent lifecycle status."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    ERROR = "error"


class CommunicationMode(str, Enum):
    """Communication modes between parent and subagent."""
    
    SYNC = "sync"                # Synchronous request-response
    ASYNC = "async"              # Asynchronous messaging
    STREAM = "stream"            # Real-time streaming
    BATCH = "batch"              # Batch processing


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""
    
    # Basic config
    agent_type: SubagentType
    name: str
    description: str = ""
    
    # Execution config
    timeout: int = 600           # 10 minutes default
    max_memory: int = 512        # MB
    max_cpu_percent: float = 50.0
    max_concurrent_tasks: int = 5
    
    # Communication config
    communication_mode: CommunicationMode = CommunicationMode.ASYNC
    message_queue_size: int = 100
    stream_buffer_size: int = 1024
    
    # Model config
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Capabilities
    tools_enabled: bool = True
    internet_access: bool = False
    file_system_access: bool = False
    subagent_spawning: bool = False
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None


@dataclass
class SubagentMetrics:
    """Performance metrics for a subagent."""
    
    # Task metrics
    tasks_received: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    # Timing metrics
    total_runtime: float = 0.0
    average_task_time: float = 0.0
    min_task_time: float = float('inf')
    max_task_time: float = 0.0
    
    # Resource metrics
    peak_memory_usage: float = 0.0  # MB
    average_cpu_usage: float = 0.0
    
    # Communication metrics
    messages_sent: int = 0
    messages_received: int = 0
    
    # Error metrics
    errors: int = 0
    timeouts: int = 0
    restarts: int = 0
    
    def update_task_completion(self, execution_time: float, success: bool) -> None:
        """Update task completion metrics."""
        self.tasks_received += 1
        
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        # Update timing
        self.min_task_time = min(self.min_task_time, execution_time)
        self.max_task_time = max(self.max_task_time, execution_time)
        
        # Update average
        total_completed = self.tasks_completed + self.tasks_failed
        if total_completed > 0:
            self.average_task_time = (
                (self.average_task_time * (total_completed - 1) + execution_time) / 
                total_completed
            )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.tasks_received == 0:
            return 0.0
        return self.tasks_completed / self.tasks_received
    
    @property
    def is_healthy(self) -> bool:
        """Check if subagent is performing well."""
        return (
            self.success_rate >= 0.8 and
            self.errors < 5 and
            self.timeouts < 3
        )


class SubagentProcess:
    """
    Individual subagent process wrapper.
    
    Manages the lifecycle, communication, and monitoring of a single subagent.
    """
    
    def __init__(
        self,
        subagent_id: str,
        config: SubagentConfig,
        parent_session_id: str
    ):
        self.id = subagent_id
        self.config = config
        self.parent_session_id = parent_session_id
        self.status = SubagentStatus.INITIALIZING
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        
        # Communication
        self.input_queue = AsyncQueue(f"subagent_{subagent_id}_input", maxsize=config.message_queue_size)
        self.output_queue = AsyncQueue(f"subagent_{subagent_id}_output", maxsize=config.message_queue_size)
        self.stream_queue: Optional[StreamQueue] = None
        
        if config.communication_mode == CommunicationMode.STREAM:
            self.stream_queue = StreamQueue(
                maxsize=config.stream_buffer_size,
                name=f"subagent_{subagent_id}_stream"
            )
        
        # Monitoring
        self.metrics = SubagentMetrics()
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Task tracking
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[TaskResult] = []
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"SubagentProcess '{self.id}' initialized", agent_type=config.agent_type)
    
    async def start(self) -> bool:
        """Start the subagent process."""
        try:
            self.status = SubagentStatus.INITIALIZING
            
            # Start the subprocess
            await self._start_process()
            
            # Start communication handlers
            self._background_tasks.add(
                asyncio.create_task(self._message_handler())
            )
            self._background_tasks.add(
                asyncio.create_task(self._process_monitor())
            )
            
            if self.config.communication_mode == CommunicationMode.STREAM:
                self._background_tasks.add(
                    asyncio.create_task(self._stream_handler())
                )
            
            self.status = SubagentStatus.READY
            self.started_at = datetime.utcnow()
            
            logger.info(f"Subagent '{self.id}' started successfully", pid=self.pid)
            return True
            
        except Exception as e:
            self.status = SubagentStatus.ERROR
            logger.error(f"Failed to start subagent '{self.id}'", error=str(e))
            return False
    
    async def stop(self, graceful: bool = True) -> None:
        """Stop the subagent process."""
        logger.info(f"Stopping subagent '{self.id}'", graceful=graceful)
        
        if graceful:
            # Send shutdown signal
            await self._send_command("shutdown", {})
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning(f"Subagent '{self.id}' did not shutdown gracefully")
        
        # Terminate process
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(2)
                
                if self.process.poll() is None:
                    self.process.kill()
                    
            except Exception as e:
                logger.error(f"Error terminating subagent process", error=str(e))
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close queues
        await self.input_queue.close()
        await self.output_queue.close()
        
        if self.stream_queue:
            await self.stream_queue.close()
        
        self.status = SubagentStatus.TERMINATED
        self.completed_at = datetime.utcnow()
        
        logger.info(f"Subagent '{self.id}' stopped")
    
    async def send_task(self, task: Task) -> bool:
        """Send a task to the subagent."""
        try:
            # Add to active tasks
            self.active_tasks[task.id] = task
            
            # Send via input queue
            message = {
                "type": "task",
                "task_id": task.id,
                "task_type": task.type,
                "content": task.content,
                "context": task.context,
                "metadata": task.metadata
            }
            
            success = await self.input_queue.put(message, priority=task.priority.value)
            
            if success:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                logger.debug(f"Sent task to subagent '{self.id}'", task_id=task.id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send task to subagent '{self.id}'", task_id=task.id, error=str(e))
            return False
    
    async def send_message(
        self,
        message: Message,
        priority: int = 0
    ) -> bool:
        """Send a message to the subagent."""
        try:
            msg_data = {
                "type": "message",
                "message": {
                    "id": message.id,
                    "role": message.role.value,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata
                }
            }
            
            success = await self.input_queue.put(msg_data, priority=priority)
            
            if success:
                self.metrics.messages_sent += 1
                logger.debug(f"Sent message to subagent '{self.id}'", message_id=message.id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message to subagent '{self.id}'", 
                        message_id=message.id, error=str(e))
            return False
    
    async def get_response(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get a response from the subagent."""
        try:
            queue_item = await self.output_queue.get(timeout=timeout)
            if queue_item:
                await self.output_queue.task_done(queue_item, success=True)
                return queue_item.data
            return None
            
        except Exception as e:
            logger.error(f"Error getting response from subagent '{self.id}'", error=str(e))
            return None
    
    async def _start_process(self) -> None:
        """Start the subagent subprocess."""
        # Build command
        cmd = [
            sys.executable, "-m", "agent_template.subagent.runner",
            "--config", json.dumps(self.config.__dict__),
            "--parent-session", self.parent_session_id,
            "--subagent-id", self.id
        ]
        
        # Prepare environment
        env = dict(os.environ)
        env.update(self.config.environment_vars)
        
        # Set working directory
        cwd = self.config.working_directory or Path.cwd()
        
        # Start process
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.pid = self.process.pid
        logger.info(f"Started subagent process", subagent_id=self.id, pid=self.pid)
    
    async def _message_handler(self) -> None:
        """Handle message communication with subagent."""
        while self.status not in [SubagentStatus.TERMINATED, SubagentStatus.ERROR]:
            try:
                # Send queued messages to process
                queue_item = await self.input_queue.get(timeout=1.0)
                if queue_item is None:
                    continue
                
                message = queue_item.data
                
                # Write to process stdin
                if self.process and self.process.stdin:
                    json_msg = json.dumps(message) + "\n"
                    self.process.stdin.write(json_msg)
                    self.process.stdin.flush()
                
                await self.input_queue.task_done(queue_item, success=True)
                
            except Exception as e:
                logger.error(f"Error in message handler for subagent '{self.id}'", error=str(e))
                if 'queue_item' in locals():
                    await self.input_queue.task_done(queue_item, success=False)
                await asyncio.sleep(1)
    
    async def _process_monitor(self) -> None:
        """Monitor the subagent process."""
        while self.status not in [SubagentStatus.TERMINATED, SubagentStatus.ERROR]:
            try:
                if not self.process:
                    await asyncio.sleep(1)
                    continue
                
                # Check if process is still running
                if self.process.poll() is not None:
                    logger.warning(f"Subagent process '{self.id}' terminated unexpectedly")
                    self.status = SubagentStatus.ERROR
                    break
                
                # Read stdout for responses
                if self.process.stdout:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, self.process.stdout.readline
                    )
                    
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            await self._handle_response(response)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON from subagent '{self.id}': {line}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in process monitor for subagent '{self.id}'", error=str(e))
                await asyncio.sleep(1)
    
    async def _stream_handler(self) -> None:
        """Handle streaming communication."""
        if not self.stream_queue:
            return
        
        while self.status not in [SubagentStatus.TERMINATED, SubagentStatus.ERROR]:
            try:
                # Handle streaming data
                queue_item = await self.stream_queue.get(timeout=1.0)
                if queue_item is None:
                    continue
                
                # Process streaming chunk
                chunk = queue_item.data
                logger.debug(f"Received stream chunk from subagent '{self.id}'", size=len(str(chunk)))
                
                await self.stream_queue.task_done(queue_item, success=True)
                
            except Exception as e:
                logger.error(f"Error in stream handler for subagent '{self.id}'", error=str(e))
                if 'queue_item' in locals():
                    await self.stream_queue.task_done(queue_item, success=False)
                await asyncio.sleep(1)
    
    async def _handle_response(self, response: Dict[str, Any]) -> None:
        """Handle a response from the subagent."""
        response_type = response.get("type")
        
        if response_type == "task_result":
            await self._handle_task_result(response)
        elif response_type == "message":
            await self._handle_message_response(response)
        elif response_type == "status":
            await self._handle_status_update(response)
        elif response_type == "error":
            await self._handle_error_response(response)
        else:
            logger.warning(f"Unknown response type from subagent '{self.id}': {response_type}")
    
    async def _handle_task_result(self, response: Dict[str, Any]) -> None:
        """Handle task completion response."""
        task_id = response.get("task_id")
        if not task_id or task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        
        # Create task result
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus(response.get("status", "completed")),
            result=response.get("result"),
            error=response.get("error"),
            execution_time=response.get("execution_time", 0.0)
        )
        
        # Update task
        task.status = result.status
        task.completed_at = datetime.utcnow()
        task.result = result.result
        task.error = result.error
        
        # Move to completed
        del self.active_tasks[task_id]
        self.completed_tasks.append(result)
        
        # Update metrics
        self.metrics.update_task_completion(
            result.execution_time,
            result.status == TaskStatus.COMPLETED
        )
        
        # Add to output queue for parent
        await self.output_queue.put(response)
        
        logger.info(f"Task completed by subagent '{self.id}'", 
                   task_id=task_id, status=result.status)
    
    async def _handle_message_response(self, response: Dict[str, Any]) -> None:
        """Handle message response."""
        self.metrics.messages_received += 1
        await self.output_queue.put(response)
    
    async def _handle_status_update(self, response: Dict[str, Any]) -> None:
        """Handle status update from subagent."""
        new_status = response.get("status")
        if new_status:
            self.status = SubagentStatus(new_status)
            logger.debug(f"Subagent '{self.id}' status updated to {new_status}")
    
    async def _handle_error_response(self, response: Dict[str, Any]) -> None:
        """Handle error response from subagent."""
        self.metrics.errors += 1
        error_msg = response.get("error", "Unknown error")
        logger.error(f"Error from subagent '{self.id}': {error_msg}")
        
        await self.output_queue.put(response)
    
    async def _send_command(self, command: str, data: Dict[str, Any]) -> None:
        """Send a command to the subagent."""
        message = {
            "type": "command",
            "command": command,
            "data": data
        }
        
        await self.input_queue.put(message, priority=10)  # High priority
    
    @property
    def runtime(self) -> Optional[float]:
        """Get runtime in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    @property
    def is_healthy(self) -> bool:
        """Check if subagent is healthy."""
        return (
            self.status in [SubagentStatus.READY, SubagentStatus.RUNNING] and
            self.metrics.is_healthy and
            (self.process is None or self.process.poll() is None)  # Process still running
        )


class SubagentManager:
    """
    Manager for multiple subagents with lifecycle coordination.
    
    Handles spawning, monitoring, and cleanup of subagent processes,
    as well as load balancing and communication coordination.
    """
    
    def __init__(self):
        self.subagents: Dict[str, SubagentProcess] = {}
        self.agent_types: Dict[SubagentType, List[str]] = {}  # Type -> agent IDs
        
        # Configuration
        self.max_subagents = settings.agent.max_subagents
        self.default_timeout = settings.agent.subagent_timeout
        
        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load balancing
        self._round_robin_counters: Dict[SubagentType, int] = {}
        
        logger.info("SubagentManager initialized", max_subagents=self.max_subagents)
    
    async def start(self) -> None:
        """Start the subagent manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("SubagentManager started")
    
    async def stop(self) -> None:
        """Stop the subagent manager and all subagents."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all subagents
        stop_tasks = [
            subagent.stop(graceful=True) 
            for subagent in self.subagents.values()
        ]
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.subagents.clear()
        self.agent_types.clear()
        
        logger.info("SubagentManager stopped")
    
    async def spawn_subagent(
        self,
        agent_type: SubagentType,
        config: Optional[SubagentConfig] = None,
        parent_session_id: str = "unknown"
    ) -> Optional[str]:
        """
        Spawn a new subagent.
        
        Returns:
            Subagent ID if successful, None if failed
        """
        if len(self.subagents) >= self.max_subagents:
            logger.warning("Maximum number of subagents reached")
            return None
        
        # Generate subagent ID
        subagent_id = f"subagent_{agent_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Use provided config or create default
        if config is None:
            config = SubagentConfig(
                agent_type=agent_type,
                name=subagent_id,
                description=f"Subagent of type {agent_type.value}"
            )
        
        # Create subagent process
        subagent = SubagentProcess(subagent_id, config, parent_session_id)
        
        # Start the subagent
        if await subagent.start():
            self.subagents[subagent_id] = subagent
            
            # Track by type
            if agent_type not in self.agent_types:
                self.agent_types[agent_type] = []
            self.agent_types[agent_type].append(subagent_id)
            
            logger.info(f"Spawned subagent '{subagent_id}'", agent_type=agent_type)
            return subagent_id
        else:
            logger.error(f"Failed to spawn subagent '{subagent_id}'")
            return None
    
    async def terminate_subagent(self, subagent_id: str, graceful: bool = True) -> bool:
        """Terminate a subagent."""
        if subagent_id not in self.subagents:
            return False
        
        subagent = self.subagents[subagent_id]
        agent_type = subagent.config.agent_type
        
        # Stop the subagent
        await subagent.stop(graceful=graceful)
        
        # Remove from tracking
        del self.subagents[subagent_id]
        
        if agent_type in self.agent_types and subagent_id in self.agent_types[agent_type]:
            self.agent_types[agent_type].remove(subagent_id)
            
            # Clean up empty type lists
            if not self.agent_types[agent_type]:
                del self.agent_types[agent_type]
        
        logger.info(f"Terminated subagent '{subagent_id}'")
        return True
    
    async def send_task_to_subagent(
        self,
        subagent_id: str,
        task: Task
    ) -> bool:
        """Send a task to a specific subagent."""
        if subagent_id not in self.subagents:
            return False
        
        subagent = self.subagents[subagent_id]
        return await subagent.send_task(task)
    
    async def send_task_to_type(
        self,
        agent_type: SubagentType,
        task: Task,
        load_balance: bool = True
    ) -> Optional[str]:
        """
        Send a task to any subagent of the specified type.
        
        Returns:
            Subagent ID that received the task, or None if failed
        """
        if agent_type not in self.agent_types or not self.agent_types[agent_type]:
            return None
        
        agent_ids = self.agent_types[agent_type]
        
        if load_balance:
            # Round-robin load balancing
            if agent_type not in self._round_robin_counters:
                self._round_robin_counters[agent_type] = 0
            
            index = self._round_robin_counters[agent_type] % len(agent_ids)
            selected_id = agent_ids[index]
            self._round_robin_counters[agent_type] += 1
        else:
            # Use first available
            selected_id = agent_ids[0]
        
        # Send task
        if await self.send_task_to_subagent(selected_id, task):
            return selected_id
        
        return None
    
    async def get_subagent_response(
        self,
        subagent_id: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Get a response from a subagent."""
        if subagent_id not in self.subagents:
            return None
        
        subagent = self.subagents[subagent_id]
        return await subagent.get_response(timeout)
    
    def get_subagent(self, subagent_id: str) -> Optional[SubagentProcess]:
        """Get a subagent by ID."""
        return self.subagents.get(subagent_id)
    
    def list_subagents(
        self,
        agent_type: Optional[SubagentType] = None,
        status: Optional[SubagentStatus] = None
    ) -> List[SubagentProcess]:
        """List subagents with optional filtering."""
        agents = list(self.subagents.values())
        
        if agent_type:
            agents = [a for a in agents if a.config.agent_type == agent_type]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    def get_available_types(self) -> List[SubagentType]:
        """Get list of available subagent types."""
        return list(self.agent_types.keys())
    
    async def _monitor_loop(self) -> None:
        """Monitor subagent health and cleanup."""
        while self._running:
            try:
                unhealthy_agents = []
                
                for subagent_id, subagent in self.subagents.items():
                    # Check health
                    if not subagent.is_healthy:
                        unhealthy_agents.append(subagent_id)
                    
                    # Check timeout
                    if subagent.runtime and subagent.runtime > self.default_timeout:
                        logger.warning(f"Subagent '{subagent_id}' exceeded timeout")
                        unhealthy_agents.append(subagent_id)
                
                # Clean up unhealthy agents
                for subagent_id in unhealthy_agents:
                    await self.terminate_subagent(subagent_id, graceful=False)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Error in subagent monitor loop", error=str(e))
                await asyncio.sleep(30)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get subagent manager statistics."""
        total_tasks = sum(len(s.active_tasks) + len(s.completed_tasks) for s in self.subagents.values())
        completed_tasks = sum(len(s.completed_tasks) for s in self.subagents.values())
        
        return {
            "total_subagents": len(self.subagents),
            "max_subagents": self.max_subagents,
            "active_subagents": len([s for s in self.subagents.values() if s.status == SubagentStatus.RUNNING]),
            "healthy_subagents": len([s for s in self.subagents.values() if s.is_healthy]),
            "types_available": len(self.agent_types),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "success_rate": (completed_tasks / max(1, total_tasks)),
            "subagents_by_type": {
                agent_type.value: len(agent_ids) 
                for agent_type, agent_ids in self.agent_types.items()
            }
        }
    
    @asynccontextmanager
    async def temporary_subagent(
        self,
        agent_type: SubagentType,
        config: Optional[SubagentConfig] = None,
        parent_session_id: str = "temp"
    ):
        """Context manager for temporary subagent that auto-cleans up."""
        subagent_id = await self.spawn_subagent(agent_type, config, parent_session_id)
        
        if subagent_id is None:
            raise RuntimeError(f"Failed to spawn temporary subagent of type {agent_type}")
        
        try:
            yield self.subagents[subagent_id]
        finally:
            await self.terminate_subagent(subagent_id, graceful=True)


# Global subagent manager instance  
subagent_manager = SubagentManager()