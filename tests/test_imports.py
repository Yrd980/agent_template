"""Test basic import functionality."""

import pytest


def test_main_import():
    """Test that the main package imports correctly."""
    import agent_template
    assert hasattr(agent_template, '__version__')


def test_core_imports():
    """Test that core modules import correctly."""
    from agent_template.core.agent_loop import AgentLoop
    from agent_template.core.async_queue import AsyncQueue
    from agent_template.core.stream_gen import StreamGenerator
    
    assert AgentLoop is not None
    assert AsyncQueue is not None
    assert StreamGenerator is not None


def test_model_imports():
    """Test that model classes import correctly."""
    from agent_template.models.messages import Message, MessageRole, MessageType, Session
    from agent_template.models.tasks import Task, TaskStatus, TaskType
    from agent_template.models.tools import ToolDefinition, ToolParameter
    
    assert Message is not None
    assert MessageRole is not None
    assert MessageType is not None
    assert Session is not None
    assert Task is not None
    assert TaskStatus is not None
    assert TaskType is not None
    assert ToolDefinition is not None
    assert ToolParameter is not None


def test_service_imports():
    """Test that service modules import correctly."""
    from agent_template.services.compressor import MessageCompressor
    from agent_template.services.state_cache import StateCache
    from agent_template.services.tool_manager import ToolManager
    from agent_template.services.session_manager import SessionManager
    from agent_template.services.model_provider import ModelProvider
    
    assert MessageCompressor is not None
    assert StateCache is not None
    assert ToolManager is not None
    assert SessionManager is not None
    assert ModelProvider is not None


def test_config_import():
    """Test that configuration imports correctly."""
    from agent_template.config import settings
    
    assert settings is not None
    assert hasattr(settings, 'database')
    assert hasattr(settings, 'models')
    assert hasattr(settings, 'server')