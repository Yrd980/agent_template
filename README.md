# Agent Template

Advanced AI Agent Template with Terminal Frontend and Python Backend.

## Overview

This is a comprehensive AI agent framework built with Python, providing a complete system for building AI agents with real-time streaming responses, multi-model support, advanced message compression, tool calling capabilities, and a rich terminal interface. The system features an event-driven architecture with async/await throughout, WebSocket integration, and persistent state management.

## Features

### Core Components
- **AgentLoop**: Central orchestration engine with priority-based task scheduling and event management
- **AsyncQueue**: Advanced async queue system with flow control strategies and backpressure handling
- **StreamGenerator**: Real-time streaming response generation with WebSocket integration
- **SessionManager**: Multi-user conversation context with persistent storage and cleanup
- **StateCache**: Comprehensive state caching with multiple backends (Memory, SQLite, Redis)

### AI & Model Support
- **Multi-Model Integration**: OpenAI, Anthropic, DeepSeek, Qwen, Ollama, and local model support
- **Model Provider Abstraction**: Unified interface across different AI providers
- **Automatic Failover**: Load balancing and failover between model providers
- **Message Compression**: AI-powered context compression when approaching token limits

### Tool & Agent Management
- **Tool Calling**: Full MCP (Model Context Protocol) support with dynamic tool discovery
- **Tool Registry**: Centralized registry for function-based and MCP tools
- **Subagent Management**: Spawn and manage specialized child agents
- **Execution Context**: Sandboxed tool execution with permissions and resource limits

### Communication & Interface
- **WebSocket Communication**: Real-time bidirectional communication
- **Rich Terminal Interface**: Advanced terminal UI with live updates and task monitoring
- **RESTful API**: Comprehensive FastAPI endpoints for all operations
- **Event System**: Comprehensive event-driven architecture with hooks

### Data & Persistence
- **Session Persistence**: Automatic session state persistence and recovery
- **Message History**: Compressed conversation history with context preservation
- **Metrics & Analytics**: Comprehensive performance metrics and monitoring
- **Todo Management**: Built-in task tracking and progress management

## Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────────┐
│  Terminal UI    │ ◄──────────────► │   FastAPI Server    │
│  (Rich/Textual) │                 │  + WebSocket Layer   │
└─────────────────┘                 └──────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │    Agent Loop        │
                                    │  - Task Scheduler    │
                                    │  - Event System      │
                                    │  - State Management  │
                                    └──────────────────────┘
                                               │
                     ┌─────────────────────────┼─────────────────────────┐
                     ▼                         ▼                         ▼
          ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
          │  AsyncQueue     │      │ StreamGenerator │      │ SessionManager  │
          │  - Flow Control │      │  - Real-time    │      │  - Sessions     │
          │  - Backpressure │      │  - WebSocket    │      │  - Context      │
          │  - Streaming    │      │  - Chunked      │      │  - Persistence  │
          └─────────────────┘      └─────────────────┘      └─────────────────┘
                     │                         │                         │
                     └─────────────────────────┼─────────────────────────┘
                                               ▼
                          ┌──────────────────────────────────────────┐
                          │              Services Layer              │
                          │  ┌─────────────┐  ┌─────────────────┐   │
                          │  │ ModelManager│  │  ToolManager    │   │
                          │  │ - OpenAI    │  │  - MCP Support  │   │
                          │  │ - Anthropic │  │  - Discovery    │   │
                          │  │ - DeepSeek  │  │  - Registry     │   │
                          │  │ - Qwen      │  │  - Execution    │   │
                          │  │ - Ollama    │  └─────────────────┘   │
                          │  └─────────────┘                      │
                          └──────────────────────────────────────────┘
                                               │
                                               ▼
                          ┌──────────────────────────────────────────┐
                          │           Storage & Cache Layer         │
                          │  ┌─────────────┐  ┌─────────────────┐   │
                          │  │ StateCache  │  │ MessageCompressor│  │
                          │  │ - Memory    │  │ - LZ4 Compression│  │
                          │  │ - SQLite    │  │ - AI Summarization│ │
                          │  │ - Redis     │  │ - Context Optim. │  │
                          │  └─────────────┘  └─────────────────┘   │
                          └──────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip

### Installation

#### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd agent_template

# Automated setup (creates venv, installs deps)
./scripts/setup.sh

# Or manual setup:
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
uv pip install -e ".[dev]"
```

#### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd agent_template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root:

```env
# Model Configuration - Add your API keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
QWEN_API_KEY=your_qwen_key_here

# Default provider
DEFAULT_MODEL_PROVIDER=openai

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./data/agent_template.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# Agent Configuration
MAX_CONCURRENT_TASKS=10
MAX_CONTEXT_LENGTH=8000
CONTEXT_COMPRESSION_THRESHOLD=0.8

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./data/logs/agent.log
```

### Usage

#### CLI Commands

```bash
# Start the FastAPI server
agent-server

# Start the terminal client
agent-client

# Show system status
agent-template status

# Test model connectivity
agent-template test-model --provider openai

# Initialize a new project (optional)
agent-template init

# Get help
agent-template --help
```

#### Server Startup

```bash
# Start with default configuration
agent-server

# Start with custom host/port
agent-server --host 0.0.0.0 --port 8080

# Start with debug mode
agent-server --debug

# View server help
agent-server --help
```

#### Python API Usage

```python
import asyncio
from agent_template import AgentLoop, StreamGenerator, SessionManager
from agent_template.config import settings
from agent_template.services.model_provider import ModelManager

async def main():
    # Initialize core components
    agent_loop = AgentLoop()
    session_manager = SessionManager()
    model_manager = ModelManager()
    stream_gen = StreamGenerator()
    
    # Create a session
    session = await session_manager.create_session(user_id="user_123")
    
    # Create and process a task
    task = await agent_loop.create_task({
        "type": "chat",
        "message": "Hello, what can you help me with?",
        "session_id": session.session_id
    })
    
    # Stream the response
    async for chunk in stream_gen.process_stream("stream_1", task):
        print(chunk.content, end="", flush=True)
    
    # Start the agent loop
    await agent_loop.run()

# Run the example
asyncio.run(main())
```

#### WebSocket Client Example

```python
import asyncio
import json
import websockets

async def websocket_client():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Send a message
        message = {
            "type": "chat",
            "content": "Hello, AI assistant!",
            "session_id": "demo_session"
        }
        await websocket.send(json.dumps(message))
        
        # Receive streaming response
        async for response in websocket:
            data = json.loads(response)
            if data["type"] == "stream_data":
                print(data["data"]["content"], end="", flush=True)
            elif data["type"] == "stream_complete":
                print("\n[Stream completed]")
                break

asyncio.run(websocket_client())
```

## Development

### Development Environment Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=agent_template --cov-report=html

# Type checking
mypy src/agent_template/

# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
```

### Development Scripts

```bash
# Run all tests
./scripts/test.sh

# Format and lint code
./scripts/lint.sh

# Build documentation
./scripts/docs.sh

# Clean temporary files
./scripts/clean.sh
```

### Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_imports.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=agent_template --cov-report=html
```

### Project Structure

```
src/agent_template/
├── __init__.py                 # Main package exports
├── config.py                   # Configuration management
├── cli.py                      # Command-line interface
├── server.py                   # FastAPI server application
├── core/                       # Core engine components
│   ├── agent_loop.py           # Central orchestration engine
│   ├── async_queue.py          # Async communication pipeline
│   └── stream_gen.py           # Real-time response generation
├── models/                     # Pydantic data models
│   ├── messages.py             # Message and session models
│   ├── tasks.py               # Task and state models
│   └── tools.py               # Tool and MCP models
├── services/                   # Business logic services
│   ├── compressor.py          # Message compression & optimization
│   ├── state_cache.py         # State persistence & caching
│   ├── model_provider.py      # Multi-model abstraction layer
│   ├── tool_manager.py        # Tool calling and MCP integration
│   ├── session_manager.py     # Session and context management
│   ├── subagent.py           # Subagent process management
│   ├── todo_manager.py       # Task tracking and todo management
│   └── messages.py           # Message processing services
├── api/                       # FastAPI routes and WebSocket
│   ├── __init__.py           # API package
│   ├── routes.py             # REST API endpoints
│   └── websocket.py          # WebSocket communication handlers
├── frontend/                  # Terminal interface components
│   ├── __init__.py           # Frontend package
│   ├── terminal_app.py       # Main terminal application
│   └── client.py             # WebSocket client implementation
└── utils/                    # Utility modules
    ├── __init__.py           # Utils package
    └── logging_setup.py      # Structured logging configuration

tests/                        # Test suite
├── __init__.py
├── test_imports.py          # Import verification tests
└── ...                      # Additional test modules

scripts/                     # Development and deployment scripts
├── setup.sh                 # Environment setup
├── test.sh                  # Test runner
├── lint.sh                  # Code formatting and linting
└── clean.sh                 # Cleanup temporary files

data/                        # Runtime data directory
├── logs/                    # Application logs
├── cache/                   # Temporary cache files
└── sessions/                # Session persistence
```

## API Documentation

### REST Endpoints

- `GET /health` - Health check endpoint
- `GET /api/v1/stats` - System statistics and metrics
- `POST /api/v1/sessions` - Create new session
- `GET /api/v1/sessions` - List sessions
- `GET /api/v1/sessions/{id}` - Get session details
- `DELETE /api/v1/sessions/{id}` - Delete session
- `POST /api/v1/messages` - Send message
- `POST /api/v1/messages/stream` - Stream message response
- `GET /api/v1/sessions/{id}/messages` - Get session messages
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks` - List tasks
- `GET /api/v1/tasks/{id}` - Get task details
- `DELETE /api/v1/tasks/{id}` - Cancel task
- `GET /api/v1/tools` - List available tools
- `POST /api/v1/tools/call` - Execute tool

### WebSocket Events

- `chat` - Send chat message
- `task_created` - Task creation event
- `task_updated` - Task progress update
- `task_completed` - Task completion event
- `stream_data` - Streaming response chunk
- `stream_complete` - Stream completion event
- `session_created` - New session created
- `session_updated` - Session state updated

## Configuration Options

### Model Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo models
- **Anthropic**: Claude-3 Sonnet, Haiku, Opus models
- **DeepSeek**: DeepSeek-Chat with OpenAI-compatible API
- **Qwen**: Qwen-Turbo via Alibaba Cloud
- **Ollama**: Local model serving
- **Local Models**: Custom model integration

### Storage Backends

- **Memory**: In-memory cache (development)
- **SQLite**: Local file-based persistence
- **Redis**: Distributed cache with clustering support

### Compression Strategies

- **LZ4**: Fast lossless compression
- **AI Summarization**: Intelligent content summarization
- **Context Optimization**: Smart context window management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: See inline code documentation and type hints
- **Issues**: Report bugs and feature requests via GitHub issues
- **Development**: Follow the development guidelines in this README

---

**Built with**: Python 3.11+, FastAPI, WebSockets, Rich/Textual, Pydantic v2, SQLAlchemy, Redis, and modern async/await patterns.
