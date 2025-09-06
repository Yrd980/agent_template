"""
Advanced AI Agent Template with Terminal Frontend and Python Backend.

This package provides a comprehensive framework for building AI agents with:
- Real-time streaming responses
- Multi-model support (OpenAI, Anthropic, local models)
- Advanced message compression and context optimization
- Tool calling with MCP (Model Context Protocol) support
- Subagent process management
- Terminal-based rich interface
- Async communication pipeline
"""

__version__ = "0.1.0"
__author__ = "Agent Template"
__email__ = "agent@template.com"

from .core.agent_loop import AgentLoop
from .core.async_queue import AsyncQueue
from .core.stream_gen import StreamGenerator
from .models.messages import Message, Session, Context
from .services.compressor import MessageCompressor
from .services.state_cache import StateCache

__all__ = [
    "AgentLoop",
    "AsyncQueue", 
    "StreamGenerator",
    "Message",
    "Session",
    "Context",
    "MessageCompressor",
    "StateCache",
]