"""
Todo-list management system.

This module provides comprehensive task tracking, progress management,
and todo list functionality for the agent system.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable

import structlog

from ..config import settings
from ..core.async_queue import AsyncQueue


logger = structlog.get_logger(__name__)


class TodoStatus(str, Enum):
    """Todo item status."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class TodoPriority(int, Enum):
    """Todo priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TodoCategory(str, Enum):
    """Todo categories for organization."""
    
    GENERAL = "general"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    MAINTENANCE = "maintenance"


@dataclass
class TodoItem:
    """Individual todo item."""
    
    # Core properties
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    priority: TodoPriority = TodoPriority.NORMAL
    category: TodoCategory = TodoCategory.GENERAL
    
    # Relationships
    parent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    # Context
    session_id: Optional[str] = None
    assignee: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.status == TodoStatus.IN_PROGRESS and self.started_at is None:
            self.started_at = datetime.utcnow()
        elif self.status == TodoStatus.COMPLETED and self.completed_at is None:
            self.completed_at = datetime.utcnow()
    
    def update_status(self, new_status: TodoStatus) -> None:
        """Update status with automatic timestamp tracking."""
        if self.status == new_status:
            return
        
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        # Handle status-specific updates
        if new_status == TodoStatus.IN_PROGRESS and old_status == TodoStatus.PENDING:
            self.started_at = datetime.utcnow()
        elif new_status == TodoStatus.COMPLETED:
            self.completed_at = datetime.utcnow()
            self.progress = 1.0
        elif new_status == TodoStatus.CANCELLED:
            self.completed_at = datetime.utcnow()
    
    def update_progress(self, progress: float) -> None:
        """Update progress with validation."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        
        self.progress = progress
        self.updated_at = datetime.utcnow()
        
        # Auto-update status based on progress
        if progress == 1.0 and self.status != TodoStatus.COMPLETED:
            self.update_status(TodoStatus.COMPLETED)
        elif progress > 0.0 and self.status == TodoStatus.PENDING:
            self.update_status(TodoStatus.IN_PROGRESS)
    
    def add_note(self, note: str) -> None:
        """Add a note to the todo item."""
        timestamp = datetime.utcnow().isoformat()
        self.notes.append(f"[{timestamp}] {note}")
        self.updated_at = datetime.utcnow()
    
    def add_dependency(self, todo_id: str) -> None:
        """Add a dependency."""
        if todo_id not in self.dependencies:
            self.dependencies.append(todo_id)
            self.updated_at = datetime.utcnow()
    
    def remove_dependency(self, todo_id: str) -> bool:
        """Remove a dependency."""
        if todo_id in self.dependencies:
            self.dependencies.remove(todo_id)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def add_subtask(self, subtask_id: str) -> None:
        """Add a subtask."""
        if subtask_id not in self.subtasks:
            self.subtasks.append(subtask_id)
            self.updated_at = datetime.utcnow()
    
    def remove_subtask(self, subtask_id: str) -> bool:
        """Remove a subtask."""
        if subtask_id in self.subtasks:
            self.subtasks.remove(subtask_id)
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    @property
    def is_overdue(self) -> bool:
        """Check if the todo is overdue."""
        if self.due_date is None:
            return False
        return datetime.utcnow() > self.due_date and self.status not in [
            TodoStatus.COMPLETED, TodoStatus.CANCELLED
        ]
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of work on this todo."""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at
    
    @property
    def is_blocked_by_dependencies(self) -> bool:
        """Check if todo is blocked by incomplete dependencies."""
        # This would need access to the todo manager to check dependency status
        # For now, return False as a placeholder
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "category": self.category.value,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "progress": self.progress,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "session_id": self.session_id,
            "assignee": self.assignee,
            "tags": self.tags,
            "metadata": self.metadata,
            "notes": self.notes,
            "is_overdue": self.is_overdue
        }


@dataclass
class TodoList:
    """Collection of related todos."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Items
    items: Dict[str, TodoItem] = field(default_factory=dict)
    
    # Organization
    categories: Set[TodoCategory] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Context
    session_id: Optional[str] = None
    owner: Optional[str] = None
    shared: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_item(self, item: TodoItem) -> None:
        """Add a todo item to the list."""
        self.items[item.id] = item
        self.categories.add(item.category)
        self.tags.update(item.tags)
        self.updated_at = datetime.utcnow()
        
        # Set parent relationship if specified
        if item.parent_id and item.parent_id in self.items:
            parent = self.items[item.parent_id]
            parent.add_subtask(item.id)
    
    def remove_item(self, item_id: str) -> bool:
        """Remove a todo item from the list."""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # Remove from parent's subtasks if applicable
        if item.parent_id and item.parent_id in self.items:
            parent = self.items[item.parent_id]
            parent.remove_subtask(item_id)
        
        # Remove this item from any dependencies
        for other_item in self.items.values():
            other_item.remove_dependency(item_id)
        
        # Remove subtasks recursively or orphan them
        for subtask_id in item.subtasks.copy():
            if subtask_id in self.items:
                subtask = self.items[subtask_id]
                subtask.parent_id = None  # Orphan the subtask
        
        del self.items[item_id]
        self.updated_at = datetime.utcnow()
        
        # Update categories and tags
        self._update_collections()
        
        return True
    
    def get_item(self, item_id: str) -> Optional[TodoItem]:
        """Get a todo item by ID."""
        return self.items.get(item_id)
    
    def get_items_by_status(self, status: TodoStatus) -> List[TodoItem]:
        """Get all items with a specific status."""
        return [item for item in self.items.values() if item.status == status]
    
    def get_items_by_category(self, category: TodoCategory) -> List[TodoItem]:
        """Get all items in a specific category."""
        return [item for item in self.items.values() if item.category == category]
    
    def get_items_by_priority(self, priority: TodoPriority) -> List[TodoItem]:
        """Get all items with a specific priority."""
        return [item for item in self.items.values() if item.priority == priority]
    
    def get_overdue_items(self) -> List[TodoItem]:
        """Get all overdue items."""
        return [item for item in self.items.values() if item.is_overdue]
    
    def get_available_items(self) -> List[TodoItem]:
        """Get items that can be worked on (not blocked by dependencies)."""
        available = []
        
        for item in self.items.values():
            if item.status in [TodoStatus.PENDING, TodoStatus.IN_PROGRESS]:
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_id in item.dependencies:
                    if dep_id in self.items:
                        dep_item = self.items[dep_id]
                        if dep_item.status != TodoStatus.COMPLETED:
                            dependencies_met = False
                            break
                
                if dependencies_met:
                    available.append(item)
        
        return available
    
    def get_next_item(self, priority_order: bool = True) -> Optional[TodoItem]:
        """Get the next item to work on."""
        available = self.get_available_items()
        
        if not available:
            return None
        
        if priority_order:
            # Sort by priority (highest first), then by creation date
            available.sort(key=lambda x: (-x.priority.value, x.created_at))
        else:
            # Sort by creation date only
            available.sort(key=lambda x: x.created_at)
        
        return available[0]
    
    def _update_collections(self) -> None:
        """Update categories and tags collections."""
        self.categories = {item.category for item in self.items.values()}
        
        all_tags = set()
        for item in self.items.values():
            all_tags.update(item.tags)
        self.tags = all_tags
    
    @property
    def progress(self) -> float:
        """Calculate overall progress of the list."""
        if not self.items:
            return 0.0
        
        total_progress = sum(item.progress for item in self.items.values())
        return total_progress / len(self.items)
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate (completed items / total items)."""
        if not self.items:
            return 0.0
        
        completed = len(self.get_items_by_status(TodoStatus.COMPLETED))
        return completed / len(self.items)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics for the todo list."""
        status_counts = {}
        for status in TodoStatus:
            status_counts[status.value] = len(self.get_items_by_status(status))
        
        priority_counts = {}
        for priority in TodoPriority:
            priority_counts[priority.value] = len(self.get_items_by_priority(priority))
        
        category_counts = {}
        for category in TodoCategory:
            category_counts[category.value] = len(self.get_items_by_category(category))
        
        return {
            "total_items": len(self.items),
            "progress": self.progress,
            "completion_rate": self.completion_rate,
            "overdue_items": len(self.get_overdue_items()),
            "available_items": len(self.get_available_items()),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "category_counts": category_counts,
            "categories": list(self.categories),
            "tags": list(self.tags)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "session_id": self.session_id,
            "owner": self.owner,
            "shared": self.shared,
            "metadata": self.metadata,
            "stats": self.stats
        }


class TodoManager:
    """
    Comprehensive todo management system.
    
    Manages multiple todo lists, provides querying capabilities,
    handles notifications, and integrates with the agent system.
    """
    
    def __init__(self):
        self.lists: Dict[str, TodoList] = {}
        self.active_list_id: Optional[str] = None
        
        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._notification_queue = AsyncQueue("todo_notifications")
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Statistics
        self._stats = {
            "total_lists": 0,
            "total_items": 0,
            "completed_items": 0,
            "overdue_items": 0,
            "items_created_today": 0,
            "items_completed_today": 0
        }
        
        logger.info("TodoManager initialized")
    
    async def start(self) -> None:
        """Start the todo manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._background_tasks.add(
            asyncio.create_task(self._notification_processor())
        )
        self._background_tasks.add(
            asyncio.create_task(self._periodic_checks())
        )
        
        logger.info("TodoManager started")
    
    async def stop(self) -> None:
        """Stop the todo manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        
        # Close notification queue
        await self._notification_queue.close()
        
        logger.info("TodoManager stopped")
    
    # List management
    def create_list(
        self,
        name: str,
        description: str = "",
        session_id: Optional[str] = None,
        owner: Optional[str] = None
    ) -> str:
        """Create a new todo list."""
        todo_list = TodoList(
            name=name,
            description=description,
            session_id=session_id,
            owner=owner
        )
        
        self.lists[todo_list.id] = todo_list
        
        # Set as active if it's the first list
        if self.active_list_id is None:
            self.active_list_id = todo_list.id
        
        self._update_stats()
        self._emit_event("list_created", todo_list)
        
        logger.info(f"Created todo list '{name}'", list_id=todo_list.id)
        return todo_list.id
    
    def delete_list(self, list_id: str) -> bool:
        """Delete a todo list."""
        if list_id not in self.lists:
            return False
        
        todo_list = self.lists[list_id]
        del self.lists[list_id]
        
        # Update active list if needed
        if self.active_list_id == list_id:
            self.active_list_id = next(iter(self.lists.keys())) if self.lists else None
        
        self._update_stats()
        self._emit_event("list_deleted", todo_list)
        
        logger.info(f"Deleted todo list '{todo_list.name}'", list_id=list_id)
        return True
    
    def get_list(self, list_id: str) -> Optional[TodoList]:
        """Get a todo list by ID."""
        return self.lists.get(list_id)
    
    def get_active_list(self) -> Optional[TodoList]:
        """Get the currently active todo list."""
        if self.active_list_id:
            return self.lists.get(self.active_list_id)
        return None
    
    def set_active_list(self, list_id: str) -> bool:
        """Set the active todo list."""
        if list_id not in self.lists:
            return False
        
        self.active_list_id = list_id
        logger.info(f"Set active todo list", list_id=list_id)
        return True
    
    def list_all_lists(self) -> List[TodoList]:
        """Get all todo lists."""
        return list(self.lists.values())
    
    # Item management
    def add_item(
        self,
        title: str,
        description: str = "",
        priority: TodoPriority = TodoPriority.NORMAL,
        category: TodoCategory = TodoCategory.GENERAL,
        list_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Add a todo item to a list."""
        # Use active list if none specified
        target_list_id = list_id or self.active_list_id
        if target_list_id not in self.lists:
            return None
        
        todo_list = self.lists[target_list_id]
        
        # Create item
        item = TodoItem(
            title=title,
            description=description,
            priority=priority,
            category=category,
            **kwargs
        )
        
        # Add to list
        todo_list.add_item(item)
        
        self._update_stats()
        self._emit_event("item_created", item)
        
        logger.info(f"Added todo item '{title}'", item_id=item.id, list_id=target_list_id)
        return item.id
    
    def update_item_status(
        self,
        item_id: str,
        status: TodoStatus,
        list_id: Optional[str] = None
    ) -> bool:
        """Update the status of a todo item."""
        item = self._find_item(item_id, list_id)
        if not item:
            return False
        
        old_status = item.status
        item.update_status(status)
        
        self._update_stats()
        self._emit_event("item_status_changed", {
            "item": item,
            "old_status": old_status,
            "new_status": status
        })
        
        logger.info(f"Updated item status", 
                   item_id=item_id, old_status=old_status, new_status=status)
        return True
    
    def update_item_progress(
        self,
        item_id: str,
        progress: float,
        list_id: Optional[str] = None
    ) -> bool:
        """Update the progress of a todo item."""
        item = self._find_item(item_id, list_id)
        if not item:
            return False
        
        old_progress = item.progress
        item.update_progress(progress)
        
        self._emit_event("item_progress_changed", {
            "item": item,
            "old_progress": old_progress,
            "new_progress": progress
        })
        
        logger.debug(f"Updated item progress", 
                    item_id=item_id, progress=progress)
        return True
    
    def add_item_note(
        self,
        item_id: str,
        note: str,
        list_id: Optional[str] = None
    ) -> bool:
        """Add a note to a todo item."""
        item = self._find_item(item_id, list_id)
        if not item:
            return False
        
        item.add_note(note)
        
        self._emit_event("item_note_added", {"item": item, "note": note})
        
        logger.debug(f"Added note to item", item_id=item_id)
        return True
    
    def remove_item(self, item_id: str, list_id: Optional[str] = None) -> bool:
        """Remove a todo item."""
        if list_id:
            todo_list = self.lists.get(list_id)
            if todo_list:
                return todo_list.remove_item(item_id)
        else:
            # Search all lists
            for todo_list in self.lists.values():
                if todo_list.remove_item(item_id):
                    self._update_stats()
                    logger.info(f"Removed todo item", item_id=item_id)
                    return True
        
        return False
    
    def get_item(self, item_id: str, list_id: Optional[str] = None) -> Optional[TodoItem]:
        """Get a todo item."""
        return self._find_item(item_id, list_id)
    
    # Querying
    def get_items_by_status(
        self,
        status: TodoStatus,
        list_id: Optional[str] = None
    ) -> List[TodoItem]:
        """Get items by status across all or specific lists."""
        items = []
        
        if list_id:
            todo_list = self.lists.get(list_id)
            if todo_list:
                items.extend(todo_list.get_items_by_status(status))
        else:
            for todo_list in self.lists.values():
                items.extend(todo_list.get_items_by_status(status))
        
        return items
    
    def get_overdue_items(self, list_id: Optional[str] = None) -> List[TodoItem]:
        """Get all overdue items."""
        items = []
        
        if list_id:
            todo_list = self.lists.get(list_id)
            if todo_list:
                items.extend(todo_list.get_overdue_items())
        else:
            for todo_list in self.lists.values():
                items.extend(todo_list.get_overdue_items())
        
        return items
    
    def get_next_item(self, list_id: Optional[str] = None) -> Optional[TodoItem]:
        """Get the next item to work on."""
        target_list = None
        
        if list_id:
            target_list = self.lists.get(list_id)
        else:
            target_list = self.get_active_list()
        
        if target_list:
            return target_list.get_next_item()
        
        return None
    
    def search_items(
        self,
        query: str,
        list_id: Optional[str] = None,
        include_completed: bool = False
    ) -> List[TodoItem]:
        """Search for items by query string."""
        items = []
        query_lower = query.lower()
        
        lists_to_search = [self.lists[list_id]] if list_id else self.lists.values()
        
        for todo_list in lists_to_search:
            for item in todo_list.items.values():
                # Skip completed items if requested
                if not include_completed and item.status == TodoStatus.COMPLETED:
                    continue
                
                # Search in title, description, and notes
                if (query_lower in item.title.lower() or 
                    query_lower in item.description.lower() or
                    any(query_lower in note.lower() for note in item.notes)):
                    items.append(item)
        
        return items
    
    # Event system
    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _emit_event(self, event: str, data: Any = None) -> None:
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(data))
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler", event=event, error=str(e))
    
    # Background processing
    async def _notification_processor(self) -> None:
        """Process notifications."""
        while self._running:
            try:
                queue_item = await self._notification_queue.get(timeout=1.0)
                if queue_item is None:
                    continue
                
                notification = queue_item.data
                await self._process_notification(notification)
                
                await self._notification_queue.task_done(queue_item, success=True)
                
            except Exception as e:
                logger.error("Error processing notification", error=str(e))
                if 'queue_item' in locals():
                    await self._notification_queue.task_done(queue_item, success=False)
                await asyncio.sleep(1)
    
    async def _periodic_checks(self) -> None:
        """Perform periodic checks for overdue items, etc."""
        while self._running:
            try:
                # Check for overdue items
                overdue_items = self.get_overdue_items()
                if overdue_items:
                    await self._notification_queue.put({
                        "type": "overdue_items",
                        "items": [item.to_dict() for item in overdue_items],
                        "count": len(overdue_items)
                    })
                
                # Update statistics
                self._update_stats()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error in periodic checks", error=str(e))
                await asyncio.sleep(60)
    
    async def _process_notification(self, notification: Dict[str, Any]) -> None:
        """Process a notification."""
        notification_type = notification.get("type")
        
        if notification_type == "overdue_items":
            logger.warning(f"Found {notification['count']} overdue items")
            self._emit_event("overdue_items_found", notification["items"])
    
    # Utility methods
    def _find_item(self, item_id: str, list_id: Optional[str] = None) -> Optional[TodoItem]:
        """Find a todo item by ID."""
        if list_id:
            todo_list = self.lists.get(list_id)
            if todo_list:
                return todo_list.get_item(item_id)
        else:
            # Search all lists
            for todo_list in self.lists.values():
                item = todo_list.get_item(item_id)
                if item:
                    return item
        
        return None
    
    def _update_stats(self) -> None:
        """Update global statistics."""
        today = datetime.utcnow().date()
        
        self._stats["total_lists"] = len(self.lists)
        
        total_items = 0
        completed_items = 0
        overdue_items = 0
        items_created_today = 0
        items_completed_today = 0
        
        for todo_list in self.lists.values():
            total_items += len(todo_list.items)
            
            for item in todo_list.items.values():
                if item.status == TodoStatus.COMPLETED:
                    completed_items += 1
                    if item.completed_at and item.completed_at.date() == today:
                        items_completed_today += 1
                
                if item.is_overdue:
                    overdue_items += 1
                
                if item.created_at.date() == today:
                    items_created_today += 1
        
        self._stats.update({
            "total_items": total_items,
            "completed_items": completed_items,
            "overdue_items": overdue_items,
            "items_created_today": items_created_today,
            "items_completed_today": items_completed_today
        })
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get todo manager statistics."""
        self._update_stats()
        return dict(self._stats)
    
    def export_list(self, list_id: str, format: str = "json") -> Optional[str]:
        """Export a todo list in the specified format."""
        todo_list = self.lists.get(list_id)
        if not todo_list:
            return None
        
        if format.lower() == "json":
            return json.dumps(todo_list.to_dict(), indent=2)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return None
    
    def import_list(self, data: str, format: str = "json") -> Optional[str]:
        """Import a todo list from data."""
        try:
            if format.lower() == "json":
                list_data = json.loads(data)
                
                # Create list
                todo_list = TodoList(
                    name=list_data.get("name", "Imported List"),
                    description=list_data.get("description", "")
                )
                
                # Import items
                for item_data in list_data.get("items", {}).values():
                    item = TodoItem(
                        title=item_data["title"],
                        description=item_data["description"],
                        status=TodoStatus(item_data["status"]),
                        priority=TodoPriority(item_data["priority"]),
                        category=TodoCategory(item_data["category"]),
                        progress=item_data["progress"]
                    )
                    todo_list.add_item(item)
                
                self.lists[todo_list.id] = todo_list
                logger.info(f"Imported todo list '{todo_list.name}'", 
                           list_id=todo_list.id, items=len(todo_list.items))
                return todo_list.id
            
        except Exception as e:
            logger.error("Failed to import todo list", error=str(e))
        
        return None


# Global todo manager instance
todo_manager = TodoManager()