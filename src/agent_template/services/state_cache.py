"""
StateCache - Persistent state management and execution history.

This module provides comprehensive state persistence, caching, and
execution history tracking for the agent system.
"""

import asyncio
import json
import pickle
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, TypeVar, Generic
from weakref import WeakValueDictionary

import aiofiles
import structlog

from ..config import settings


logger = structlog.get_logger(__name__)

T = TypeVar('T')


class StorageBackend(str, Enum):
    """Available storage backends."""
    
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    REDIS = "redis"


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    SIZE = "size"        # Size-based eviction


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    
    key: str
    value: T
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size in bytes."""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Update access information."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "ttl": self.ttl,
            "tags": list(self.tags)
        }


class MemoryCache(Generic[T]):
    """In-memory cache with various eviction policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100 * 1024 * 1024,  # 100MB
        policy: CachePolicy = CachePolicy.LRU,
        default_ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.policy = policy
        self.default_ttl = default_ttl
        
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: List[str] = []  # For LRU
        self._lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory = 0
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access info
            entry.touch()
            
            # Update access order for LRU
            if self.policy == CachePolicy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            
            self.hits += 1
            return entry.value
    
    async def put(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Put value in cache."""
        async with self._lock:
            current_time = time.time()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                ttl=ttl or self.default_ttl,
                tags=tags or set()
            )
            
            # Check if we need to evict
            await self._ensure_capacity(entry.size_bytes)
            
            # Remove old entry if it exists
            if key in self._cache:
                await self._remove_entry(key)
            
            # Add new entry
            self._cache[key] = entry
            self.current_memory += entry.size_bytes
            
            # Update access order for LRU
            if self.policy == CachePolicy.LRU:
                self._access_order.append(key)
            
            return True
    
    async def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self.current_memory = 0
    
    async def clear_by_tag(self, tag: str) -> int:
        """Clear entries by tag."""
        async with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]
            
            for key in keys_to_remove:
                await self._remove_entry(key)
            
            return len(keys_to_remove)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage": self.current_memory,
            "max_memory": self.max_memory,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "policy": self.policy.value
        }
    
    async def _ensure_capacity(self, needed_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        while (len(self._cache) >= self.max_size or 
               self.current_memory + needed_bytes > self.max_memory):
            
            if not self._cache:
                break
            
            # Find entry to evict based on policy
            victim_key = await self._select_victim()
            if victim_key:
                await self._remove_entry(victim_key)
                self.evictions += 1
            else:
                break
    
    async def _select_victim(self) -> Optional[str]:
        """Select entry to evict based on policy."""
        if not self._cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            return self._access_order[0] if self._access_order else None
        
        elif self.policy == CachePolicy.LFU:
            # Find least frequently used
            min_count = float('inf')
            victim = None
            
            for key, entry in self._cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    victim = key
            
            return victim
        
        elif self.policy == CachePolicy.TTL:
            # Find oldest entry
            oldest_time = float('inf')
            victim = None
            
            for key, entry in self._cache.items():
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    victim = key
            
            return victim
        
        elif self.policy == CachePolicy.SIZE:
            # Find largest entry
            max_size = 0
            victim = None
            
            for key, entry in self._cache.items():
                if entry.size_bytes > max_size:
                    max_size = entry.size_bytes
                    victim = key
            
            return victim
        
        return None
    
    async def _remove_entry(self, key: str) -> None:
        """Remove entry and update bookkeeping."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self.current_memory -= entry.size_bytes
            
            if key in self._access_order:
                self._access_order.remove(key)


class FileStorage:
    """File-based storage backend."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        # Use hash to avoid filesystem issues with special characters
        import hashlib
        hash_key = hashlib.sha256(key.encode()).hexdigest()
        return self.storage_path / f"{hash_key}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file storage."""
        file_path = self._get_file_path(key)
        lock = self._get_lock(key)
        
        async with lock:
            try:
                if not file_path.exists():
                    return None
                
                async with aiofiles.open(file_path, 'rb') as f:
                    data = await f.read()
                
                entry_data = pickle.loads(data)
                entry = CacheEntry(**entry_data)
                
                # Check expiration
                if entry.is_expired():
                    await self.remove(key)
                    return None
                
                # Update access info and save back
                entry.touch()
                await self._save_entry(key, entry)
                
                return entry.value
                
            except Exception as e:
                logger.error(f"Error reading from file storage", key=key, error=str(e))
                return None
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Put value in file storage."""
        lock = self._get_lock(key)
        
        async with lock:
            try:
                current_time = time.time()
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=current_time,
                    accessed_at=current_time,
                    ttl=ttl,
                    tags=tags or set()
                )
                
                await self._save_entry(key, entry)
                return True
                
            except Exception as e:
                logger.error(f"Error writing to file storage", key=key, error=str(e))
                return False
    
    async def remove(self, key: str) -> bool:
        """Remove value from file storage."""
        file_path = self._get_file_path(key)
        lock = self._get_lock(key)
        
        async with lock:
            try:
                if file_path.exists():
                    file_path.unlink()
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Error removing from file storage", key=key, error=str(e))
                return False
    
    async def _save_entry(self, key: str, entry: CacheEntry) -> None:
        """Save entry to file."""
        file_path = self._get_file_path(key)
        
        # Convert entry to dict for serialization
        entry_dict = entry.to_dict()
        data = pickle.dumps(entry_dict)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)


class SQLiteStorage:
    """SQLite-based storage backend."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    ttl REAL,
                    tags TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)")
            conn.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from SQLite storage."""
        async with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT value, created_at, accessed_at, access_count, ttl, tags FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    value_blob, created_at, accessed_at, access_count, ttl, tags_str = row
                    
                    # Check expiration
                    if ttl and time.time() > (created_at + ttl):
                        await self.remove(key)
                        return None
                    
                    # Deserialize value
                    value = pickle.loads(value_blob)
                    
                    # Update access info
                    new_accessed_at = time.time()
                    new_access_count = access_count + 1
                    
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at = ?, access_count = ? WHERE key = ?",
                        (new_accessed_at, new_access_count, key)
                    )
                    conn.commit()
                    
                    return value
                    
            except Exception as e:
                logger.error(f"Error reading from SQLite storage", key=key, error=str(e))
                return None
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Put value in SQLite storage."""
        async with self._lock:
            try:
                current_time = time.time()
                value_blob = pickle.dumps(value)
                tags_str = json.dumps(list(tags)) if tags else "[]"
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, accessed_at, access_count, ttl, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (key, value_blob, current_time, current_time, 1, ttl, tags_str))
                    conn.commit()
                
                return True
                
            except Exception as e:
                logger.error(f"Error writing to SQLite storage", key=key, error=str(e))
                return False
    
    async def remove(self, key: str) -> bool:
        """Remove value from SQLite storage."""
        async with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0
                    
            except Exception as e:
                logger.error(f"Error removing from SQLite storage", key=key, error=str(e))
                return False
    
    async def clear_expired(self) -> int:
        """Clear expired entries."""
        async with self._lock:
            try:
                current_time = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE ttl IS NOT NULL AND ? > (created_at + ttl)",
                        (current_time,)
                    )
                    conn.commit()
                    return cursor.rowcount
                    
            except Exception as e:
                logger.error("Error clearing expired entries", error=str(e))
                return 0


@dataclass
class ExecutionRecord:
    """Record of a tool or task execution."""
    
    id: str
    type: str  # "task", "tool", "subagent", etc.
    name: str
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Input/Output
    input_data: Dict[str, Any] = None
    output_data: Dict[str, Any] = None
    
    # Status
    status: str = "running"  # running, completed, failed, cancelled
    error: Optional[str] = None
    
    # Context
    session_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Metrics
    memory_usage: Optional[int] = None  # bytes
    cpu_time: Optional[float] = None    # seconds
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.input_data is None:
            self.input_data = {}
        if self.output_data is None:
            self.output_data = {}
        if self.metadata is None:
            self.metadata = {}
    
    def complete(
        self,
        status: str = "completed",
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark execution as completed."""
        self.completed_at = datetime.utcnow()
        self.status = status
        
        if self.completed_at and self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        
        if output_data:
            self.output_data.update(output_data)
        
        if error:
            self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "error": self.error,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "memory_usage": self.memory_usage,
            "cpu_time": self.cpu_time,
            "metadata": self.metadata
        }


class StateCache:
    """
    Comprehensive state cache and execution history system.
    
    Provides persistent caching with multiple backends, execution tracking,
    and state management for the agent system.
    """
    
    def __init__(
        self,
        backend: StorageBackend = StorageBackend.MEMORY,
        storage_path: Optional[Path] = None,
        max_memory_items: int = 1000,
        default_ttl: Optional[float] = None
    ):
        self.backend = backend
        self.storage_path = storage_path or settings.data_dir / "cache"
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        # Initialize storage
        self._init_storage()
        
        # Execution history
        self._execution_history: List[ExecutionRecord] = []
        self._active_executions: Dict[str, ExecutionRecord] = {}
        self._max_history_size = 10000
        
        # Metrics and state tracking
        self._state_snapshots: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, Any] = {}
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"StateCache initialized with {backend.value} backend")
    
    def _init_storage(self) -> None:
        """Initialize storage backend."""
        if self.backend == StorageBackend.MEMORY:
            self.storage = MemoryCache(
                max_size=self.max_memory_items,
                default_ttl=self.default_ttl
            )
        elif self.backend == StorageBackend.FILE:
            self.storage = FileStorage(self.storage_path)
        elif self.backend == StorageBackend.SQLITE:
            db_path = self.storage_path / "state_cache.db"
            self.storage = SQLiteStorage(db_path)
        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")
    
    async def start(self) -> None:
        """Start the state cache."""
        if self._running:
            return
        
        self._running = True
        
        # Start cleanup task for backends that support it
        if hasattr(self.storage, 'clear_expired'):
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("StateCache started")
    
    async def stop(self) -> None:
        """Stop the state cache."""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("StateCache stopped")
    
    # Core caching interface
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return await self.storage.get(key)
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Put value in cache."""
        return await self.storage.put(key, value, ttl or self.default_ttl, tags)
    
    async def remove(self, key: str) -> bool:
        """Remove value from cache."""
        return await self.storage.remove(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        value = await self.get(key)
        return value is not None
    
    # Execution history
    def start_execution(
        self,
        execution_id: str,
        execution_type: str,
        name: str,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionRecord:
        """Start tracking an execution."""
        record = ExecutionRecord(
            id=execution_id,
            type=execution_type,
            name=name,
            started_at=datetime.utcnow(),
            session_id=session_id,
            parent_id=parent_id,
            input_data=input_data or {},
            metadata=metadata or {}
        )
        
        self._active_executions[execution_id] = record
        
        logger.debug(f"Started tracking execution", 
                    execution_id=execution_id, type=execution_type, name=name)
        return record
    
    def complete_execution(
        self,
        execution_id: str,
        status: str = "completed",
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[ExecutionRecord]:
        """Complete an execution."""
        if execution_id not in self._active_executions:
            logger.warning(f"Execution not found", execution_id=execution_id)
            return None
        
        record = self._active_executions[execution_id]
        record.complete(status, output_data, error)
        
        # Add metrics if provided
        if metrics:
            record.memory_usage = metrics.get('memory_usage')
            record.cpu_time = metrics.get('cpu_time')
            record.metadata.update(metrics.get('metadata', {}))
        
        # Move to history
        del self._active_executions[execution_id]
        self._execution_history.append(record)
        
        # Trim history if too large
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]
        
        logger.debug(f"Completed execution", 
                    execution_id=execution_id, status=status, duration=record.duration)
        return record
    
    def get_execution_history(
        self,
        session_id: Optional[str] = None,
        execution_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionRecord]:
        """Get execution history with optional filtering."""
        history = self._execution_history
        
        # Apply filters
        if session_id:
            history = [r for r in history if r.session_id == session_id]
        
        if execution_type:
            history = [r for r in history if r.type == execution_type]
        
        # Sort by start time (most recent first) and limit
        history.sort(key=lambda r: r.started_at, reverse=True)
        return history[:limit]
    
    def get_active_executions(self) -> List[ExecutionRecord]:
        """Get currently active executions."""
        return list(self._active_executions.values())
    
    # State snapshots
    async def save_state_snapshot(
        self,
        snapshot_id: str,
        state_data: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> bool:
        """Save a state snapshot."""
        snapshot = {
            "id": snapshot_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": state_data
        }
        
        cache_key = f"state_snapshot:{snapshot_id}"
        success = await self.put(cache_key, snapshot, ttl)
        
        if success:
            self._state_snapshots[snapshot_id] = snapshot
            logger.debug(f"Saved state snapshot", snapshot_id=snapshot_id)
        
        return success
    
    async def load_state_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load a state snapshot."""
        cache_key = f"state_snapshot:{snapshot_id}"
        snapshot = await self.get(cache_key)
        
        if snapshot:
            logger.debug(f"Loaded state snapshot", snapshot_id=snapshot_id)
            return snapshot.get("data")
        
        return None
    
    def list_state_snapshots(self) -> List[str]:
        """List available state snapshots."""
        return list(self._state_snapshots.keys())
    
    # Metrics and statistics
    async def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        metric_data = {
            "name": metric_name,
            "value": value,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "tags": tags or {}
        }
        
        cache_key = f"metric:{metric_name}:{int(time.time())}"
        await self.put(cache_key, metric_data, ttl=86400)  # Keep for 24 hours
        
        logger.debug(f"Recorded metric", name=metric_name, value=value)
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get metric values (limited functionality - would need time-series DB for full implementation)."""
        # This is a simplified implementation
        # In a real system, you'd use a proper time-series database
        logger.warning("get_metrics has limited functionality with current backends")
        return []
    
    # Cleanup and maintenance
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while self._running:
            try:
                if hasattr(self.storage, 'clear_expired'):
                    cleared = await self.storage.clear_expired()
                    if cleared > 0:
                        logger.debug(f"Cleared {cleared} expired cache entries")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error("Error in cache cleanup loop", error=str(e))
                await asyncio.sleep(60)
    
    async def clear_expired(self) -> int:
        """Manually clear expired entries."""
        if hasattr(self.storage, 'clear_expired'):
            return await self.storage.clear_expired()
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": self.backend.value,
            "execution_history_size": len(self._execution_history),
            "active_executions": len(self._active_executions),
            "state_snapshots": len(self._state_snapshots)
        }
        
        if hasattr(self.storage, 'get_stats'):
            cache_stats = await self.storage.get_stats()
            stats.update(cache_stats)
        
        return stats
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics (alias for get_cache_stats for API compatibility)."""
        return await self.get_cache_stats()
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        if hasattr(self.storage, 'clear'):
            await self.storage.clear()
        
        # Clear internal state
        self._execution_history.clear()
        self._active_executions.clear()
        self._state_snapshots.clear()
        
        logger.info("Cache cleared")
    
    # Context managers
    @asynccontextmanager
    async def execution_context(
        self,
        execution_id: str,
        execution_type: str,
        name: str,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """Context manager for execution tracking."""
        record = self.start_execution(
            execution_id, execution_type, name, session_id, input_data=input_data
        )
        
        try:
            yield record
            self.complete_execution(execution_id, "completed")
        except Exception as e:
            self.complete_execution(execution_id, "failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def cached_computation(
        self,
        cache_key: str,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ):
        """Context manager for cached computations."""
        # Check if result is cached
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            yield cached_result
            return
        
        # Compute and cache result
        class ResultWrapper:
            def __init__(self):
                self.result = None
                self.has_result = False
            
            def set_result(self, result):
                self.result = result
                self.has_result = True
        
        wrapper = ResultWrapper()
        
        try:
            yield wrapper
            
            # Cache the result if one was set
            if wrapper.has_result:
                await self.put(cache_key, wrapper.result, ttl, tags)
                
        except Exception:
            # Don't cache failed computations
            raise


# Global state cache instance
state_cache = StateCache(
    backend=StorageBackend.SQLITE,
    storage_path=settings.data_dir / "cache",
    default_ttl=3600  # 1 hour default TTL
)