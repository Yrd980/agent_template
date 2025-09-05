"""Configuration management for the agent template."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./agent_template.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")


class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")


class ModelConfig(BaseSettings):
    """AI model configuration."""
    
    default_provider: str = Field(default="openai", env="DEFAULT_MODEL_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # Local model settings
    local_model_url: Optional[str] = Field(default=None, env="LOCAL_MODEL_URL")
    local_model_name: Optional[str] = Field(default=None, env="LOCAL_MODEL_NAME")
    
    # Model parameters
    temperature: float = Field(default=0.7, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="MODEL_MAX_TOKENS")
    
    @field_validator('default_provider')
    @classmethod
    def validate_provider(cls, v):
        valid_providers = ['openai', 'anthropic', 'local']
        if v not in valid_providers:
            raise ValueError(f'Provider must be one of {valid_providers}')
        return v


class ServerConfig(BaseSettings):
    """Server configuration."""
    
    host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    port: int = Field(default=8000, env="SERVER_PORT")
    reload: bool = Field(default=False, env="SERVER_RELOAD")
    workers: int = Field(default=1, env="SERVER_WORKERS")
    
    # WebSocket settings
    websocket_ping_interval: int = Field(default=20, env="WS_PING_INTERVAL")
    websocket_ping_timeout: int = Field(default=10, env="WS_PING_TIMEOUT")


class AgentConfig(BaseSettings):
    """Agent-specific configuration."""
    
    # Task processing
    max_concurrent_tasks: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # seconds
    
    # Message handling
    max_context_length: int = Field(default=8000, env="MAX_CONTEXT_LENGTH")
    context_compression_threshold: float = Field(default=0.8, env="CONTEXT_COMPRESSION_THRESHOLD")
    
    # Stream settings
    stream_chunk_size: int = Field(default=1024, env="STREAM_CHUNK_SIZE")
    stream_timeout: int = Field(default=30, env="STREAM_TIMEOUT")
    
    # Subagent settings
    max_subagents: int = Field(default=5, env="MAX_SUBAGENTS")
    subagent_timeout: int = Field(default=600, env="SUBAGENT_TIMEOUT")
    
    # Tool settings
    enable_tool_discovery: bool = Field(default=True, env="ENABLE_TOOL_DISCOVERY")
    tool_timeout: int = Field(default=60, env="TOOL_TIMEOUT")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    file: Optional[str] = Field(default=None, env="LOG_FILE")
    max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class Settings(BaseSettings):
    """Main settings container."""
    
    # Environment
    env: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    models: ModelConfig = ModelConfig()
    server: ServerConfig = ServerConfig()
    agent: AgentConfig = AgentConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Paths
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    log_dir: Path = Field(default=Path("./logs"), env="LOG_DIR")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False
    }
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()