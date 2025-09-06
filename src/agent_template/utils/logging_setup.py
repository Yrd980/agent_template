"""Logging setup and configuration utilities."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog.typing import FilteringBoundLogger

from ..config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> FilteringBoundLogger:
    """
    Set up structured logging with appropriate configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_file: Path to log file (optional)
    
    Returns:
        Configured structlog logger
    """
    # Use provided values or fall back to settings
    level = log_level or settings.logging.level
    format_type = log_format or settings.logging.format
    file_path = log_file or settings.logging.file
    
    # Configure standard library logging
    logging_level = getattr(logging, level.upper())
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging_level)
    handlers.append(console_handler)
    
    # File handler if specified
    if file_path:
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=_parse_size(settings.logging.max_size),
            backupCount=settings.logging.backup_count
        )
        file_handler.setLevel(logging_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging_level,
        handlers=handlers,
        format="%(message)s",  # Let structlog handle formatting
        force=True
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if format_type == "json":
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer()
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info("Logging configured", level=level, format=format_type, file=file_path)
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string (e.g., '100MB') into bytes."""
    size_str = size_str.upper().strip()
    
    multipliers = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            try:
                number = float(size_str[:-len(suffix)])
                return int(number * multiplier)
            except ValueError:
                break
    
    # Default to 100MB if parsing fails
    return 100 * 1024 * 1024


def get_logger(name: Optional[str] = None) -> FilteringBoundLogger:
    """Get a configured structlog logger."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def set_log_level(level: str) -> None:
    """Dynamically change the log level."""
    logging_level = getattr(logging, level.upper())
    logging.getLogger().setLevel(logging_level)
    
    # Update all handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging_level)
    
    logger = get_logger()
    logger.info("Log level changed", level=level)