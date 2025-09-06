# Agent Template Project

## Overview

This is a comprehensive AI agent template system with terminal frontend and Python backend. The system provides a complete framework for building AI agents with multi-model support, tool calling capabilities, and real-time communication.

## Architecture

### Core Components

1. **AgentLoop** (`src/agent_template/core/agent_loop.py`)
   - Central orchestration engine with priority-based task scheduling
   - Event-driven state management with hooks
   - Async task processing with graceful shutdown

2. **AsyncQueue** (`src/agent_template/core/async_queue.py`)
   - Advanced async queue system with flow control strategies
   - Backpressure handling and streaming support
   - Multiple queue modes (FIFO, LIFO, Priority)

3. **StreamGen** (`src/agent_template/core/stream_gen.py`)
   - Real-time streaming response generation
   - WebSocket integration for live updates
   - Chunked response handling

4. **Multi-Model Support** (`src/agent_template/services/model_provider.py`)
   - Abstract provider interface supporting multiple AI models
   - Providers: OpenAI, Anthropic, DeepSeek, Qwen, Ollama, Local models
   - Automatic failover and load balancing

5. **Tool Calling & MCP** (`src/agent_template/services/tool_manager.py`)
   - Model Context Protocol (MCP) integration
   - Dynamic tool discovery and registration
   - Secure tool execution with sandboxing

6. **Message System** (`src/agent_template/core/messages.py`)
   - Session management with context preservation
   - Message compression with multiple strategies (LZ4, AI summarization)
   - Temporary caching and history management

7. **State Management** (`src/agent_template/utils/state_cache.py`)
   - Persistent state caching with multiple backends (Memory, Redis, SQLite)
   - Tool state tracking and execution history
   - Performance metrics and analytics

### Frontend

- **Terminal Interface** (`src/agent_template/frontend/terminal_app.py`)
  - Rich/Textual-based terminal UI
  - Real-time task monitoring and todo list display
  - WebSocket communication with backend

### API Layer

- **FastAPI Server** (`src/agent_template/server.py`)
  - RESTful API endpoints for all agent operations
  - WebSocket support for real-time communication
  - Comprehensive health checking and monitoring

- **WebSocket Manager** (`src/agent_template/api/websocket.py`)
  - Connection management with automatic cleanup
  - Event broadcasting and session subscriptions
  - Heartbeat and reconnection handling

## Key Features

### 1. Multi-Model AI Support
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3 Sonnet, Haiku, Opus
- **DeepSeek**: DeepSeek-Chat (OpenAI-compatible API)
- **Qwen**: Qwen-Turbo (Alibaba Cloud)
- **Ollama**: Local model serving
- **Local Models**: Custom model integration

### 2. Advanced Task Management
- Priority-based task scheduling
- Concurrent task execution with limits
- Task dependencies and chaining
- Progress tracking and cancellation

### 3. Real-time Communication
- WebSocket-based live updates
- Event-driven architecture
- Session-based message routing
- Streaming responses

### 4. Tool Integration
- Model Context Protocol (MCP) support
- Dynamic tool discovery
- Secure execution environment
- Tool state persistence

### 5. Message Compression & Optimization
- Automatic context compression when limits approached
- Multiple compression strategies:
  - LZ4 lossless compression
  - AI-based summarization
  - Key information extraction
- Context window optimization

### 6. State Persistence
- Multiple storage backends
- Tool execution history
- Performance metrics
- Session state recovery

## Configuration

The system uses environment-based configuration with `.env` file support:

```env
# Model Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
QWEN_API_KEY=your_qwen_key_here

# Default provider
DEFAULT_MODEL_PROVIDER=openai

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Database
DATABASE_URL=sqlite:///./data/agent_template.db
REDIS_URL=redis://localhost:6379/0

# Agent Configuration
MAX_CONCURRENT_TASKS=10
MAX_CONTEXT_LENGTH=8000
CONTEXT_COMPRESSION_THRESHOLD=0.8
```

## Usage

### Setup with UV
```bash
# Create virtual environment and install dependencies
./scripts/setup.sh

# Or manually:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running the System

1. **Start the Server**:
   ```bash
   uv run agent-server
   # or
   python -m agent_template.server
   ```

2. **Start the Terminal Client**:
   ```bash
   uv run agent-client
   # or
   python -m agent_template.frontend.terminal_app
   ```

3. **CLI Commands**:
   ```bash
   # Initialize project
   agent-template init

   # Test model connectivity
   agent-template test-model --provider openai

   # Check system status
   agent-template status
   ```

### Development Commands

```bash
# Run tests
uv run pytest

# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy src/

# Start development server with reload
uv run uvicorn agent_template.server:get_server --reload
```

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /api/v1/stats` - System statistics
- `WS /ws` - WebSocket connection

### Session Management
- `POST /api/v1/sessions` - Create session
- `GET /api/v1/sessions` - List sessions
- `GET /api/v1/sessions/{id}` - Get session
- `DELETE /api/v1/sessions/{id}` - Delete session

### Messaging
- `POST /api/v1/messages` - Send message
- `POST /api/v1/messages/stream` - Stream message
- `GET /api/v1/sessions/{id}/messages` - Get messages

### Task Management
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks` - List tasks
- `GET /api/v1/tasks/{id}` - Get task
- `DELETE /api/v1/tasks/{id}` - Cancel task

### Tools
- `GET /api/v1/tools` - List available tools
- `POST /api/v1/tools/call` - Call a tool

## Project Structure

```
src/agent_template/
├── api/                    # FastAPI routes and WebSocket handling
├── core/                   # Core engine components
├── frontend/               # Terminal interface
├── models/                 # Pydantic data models
├── services/               # Business logic services  
├── utils/                  # Utilities and helpers
├── config.py              # Configuration management
├── cli.py                 # Command line interface
└── server.py              # Main server application

scripts/                   # Development and setup scripts
tests/                     # Test suite
data/                      # Runtime data (logs, cache, sessions)
```

## Development Guidelines

### Code Style
- Python 3.11+ with type hints
- Async/await throughout
- Pydantic v2 for data validation
- Structured logging with contextual information

### Architecture Patterns
- Event-driven design with hooks
- Dependency injection
- Abstract base classes for extensibility
- Comprehensive error handling

### Testing
```bash
pytest --cov=agent_template --cov-report=html
```

### Contributing
1. Follow existing code patterns
2. Add comprehensive type hints
3. Include tests for new functionality
4. Update documentation

## Troubleshooting

### Common Issues

1. **Pydantic Import Errors**
   - Ensure using Pydantic v2
   - Check imports from `pydantic_settings`

2. **WebSocket Connection Issues**
   - Verify server is running on correct port
   - Check firewall settings

3. **Model API Errors**
   - Validate API keys in `.env`
   - Check model availability and quotas

### Debugging

Enable debug mode:
```bash
DEBUG=true LOG_LEVEL=DEBUG agent-template server
```

Check logs:
```bash
tail -f data/logs/agent.log
```

## License

MIT License - see LICENSE file for details.
