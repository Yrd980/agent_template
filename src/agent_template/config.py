"""Configuration management for the agent template."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    model_config = SettingsConfigDict(env_prefix='DATABASE_')
    
    url: str = "sqlite:///./agent_template.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class RedisConfig(BaseSettings):
    """Redis configuration."""
    model_config = SettingsConfigDict(env_prefix='REDIS_')
    
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10


class ModelConfig(BaseSettings):
    """AI model configuration."""
    
    default_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # Local model settings
    local_model_url: Optional[str] = None
    local_model_name: Optional[str] = None
    
    # DeepSeek settings
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    
    # Qwen settings  
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen-turbo"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    
    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @field_validator('default_provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ['openai', 'anthropic', 'local', 'deepseek', 'qwen', 'ollama']
        if v not in valid_providers:
            raise ValueError(f'Provider must be one of {valid_providers}')
        return v


class ServerConfig(BaseSettings):
    """Server configuration."""
    model_config = SettingsConfigDict(env_prefix='SERVER_')
    
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # WebSocket settings
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10


class AgentConfig(BaseSettings):
    """Agent-specific configuration."""
    
    # Task processing
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    
    # Message handling
    max_context_length: int = 8000
    context_compression_threshold: float = 0.8
    
    # Stream settings
    stream_chunk_size: int = 1024
    stream_timeout: int = 30
    
    # Subagent settings
    max_subagents: int = 5
    subagent_timeout: int = 600
    
    # Tool settings
    enable_tool_discovery: bool = True
    tool_timeout: int = 60


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    model_config = SettingsConfigDict(env_prefix='LOG_')
    
    level: str = "INFO"
    format: str = "json"  # json or text
    file: Optional[str] = None
    max_size: str = "100MB"
    backup_count: int = 5
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class Settings(BaseSettings):
    """Main settings container."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    env: str = "development"
    debug: bool = False
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Paths
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
        
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()