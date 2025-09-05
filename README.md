# Agent Template

Advanced AI Agent Template with Terminal Frontend and Python Backend.

## Features

- **AgentLoop**: Task scheduling and state management
- **AsyncQueue**: Async communication pipeline with stream handling
- **StreamGen**: Real-time response generation and streaming
- **Message System**: Session management, context queue, and caching
- **Multi-model Support**: OpenAI, Anthropic, and local model integration
- **Message Compression**: Automatic compression and context optimization
- **Tool Calling**: MCP (Model Context Protocol) support
- **Subagent Management**: Spawn and manage specialized child agents
- **Todo-list**: Built-in task tracking and progress management
- **StateCache**: Persistent state and execution history
- **Rich Terminal**: Advanced terminal interface with real-time updates

## Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────────┐
│  Terminal UI    │ ◄──────────────► │   FastAPI Server    │
│  (Rich/Textual) │                 │                      │
└─────────────────┘                 └──────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │    Agent Loop        │
                                    │  - Task Scheduler    │
                                    │  - State Management  │
                                    └──────────────────────┘
                                               │
                     ┌─────────────────────────┼─────────────────────────┐
                     ▼                         ▼                         ▼
          ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
          │  AsyncQueue     │      │   StreamGen     │      │  Messages       │
          │  - Flow Control │      │  - Real-time    │      │  - Sessions     │
          │  - Backpressure │      │  - Streaming    │      │  - Context      │
          └─────────────────┘      └─────────────────┘      └─────────────────┘
                     │                         │                         │
                     └─────────────────────────┼─────────────────────────┘
                                               ▼
                                    ┌──────────────────────┐
                                    │   Multi-Model        │
                                    │  - OpenAI API        │
                                    │  - Anthropic API     │
                                    │  - Local Models      │
                                    └──────────────────────┘
```

## Quick Start

### Installation

#### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd agent_template

# Automated setup (creates venv, installs deps, creates config)
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

# Install development dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root:

```env
# Model Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_MODEL_PROVIDER=openai

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Database
DATABASE_URL=sqlite:///./agent_template.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Usage

#### Start the Server

```bash
# Start the FastAPI server
agent-server

# Or with custom configuration
uvicorn agent_template.api.main:app --host 0.0.0.0 --port 8000
```

#### Launch Terminal Client

```bash
# Start the terminal interface
agent-client

# Or connect to remote server
agent-client --host remote-server.com --port 8000
```

#### Python API

```python
from agent_template import AgentLoop, StreamGen
from agent_template.config import settings

# Initialize components
agent_loop = AgentLoop()
stream_gen = StreamGen()

# Process a task
async def main():
    task = await agent_loop.create_task({
        "type": "chat",
        "message": "Hello, what can you help me with?",
        "session_id": "user_123"
    })
    
    async for chunk in stream_gen.generate_stream(task):
        print(chunk, end="", flush=True)

# Run the agent loop
agent_loop.run()
```

## Development

### Setup Development Environment

#### Using uv and dev scripts (Recommended)

```bash
# Install pre-commit hooks
pre-commit install

# Run tests
./scripts/dev.sh test
# or: uv run pytest

# Run linting
./scripts/dev.sh lint
# or: uv run flake8 src/ tests/ && uv run mypy src/agent_template

# Format code
./scripts/dev.sh format
# or: uv run black src/ tests/ && uv run isort src/ tests/

# Install/update dependencies
./scripts/dev.sh install

# Clean up temp files
./scripts/dev.sh clean
```

#### Traditional approach

```bash
# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/agent_template
```

### Project Structure

```
src/agent_template/
├── __init__.py
├── config.py              # Configuration management
├── core/                  # Core components
│   ├── agent_loop.py      # Main orchestration engine
│   ├── async_queue.py     # Async communication pipeline
│   └── stream_gen.py      # Real-time response generation
├── models/                # Data models
│   ├── messages.py        # Message and session models
│   ├── tasks.py          # Task and state models
│   └── tools.py          # Tool and MCP models
├── services/             # Business logic services
│   ├── compressor.py     # Message compression
│   ├── state_cache.py    # State persistence
│   ├── model_provider.py # Multi-model abstraction
│   ├── tool_manager.py   # Tool calling and MCP
│   └── subagent.py       # Subagent management
├── api/                  # FastAPI routes and WebSocket
│   ├── main.py           # FastAPI application
│   ├── routes/           # API endpoints
│   └── websocket.py      # WebSocket handlers
├── frontend/             # Terminal interface
│   ├── client.py         # Terminal client
│   ├── components/       # UI components
│   └── widgets/          # Custom widgets
├── utils/                # Utilities
│   ├── logging.py        # Structured logging
│   ├── compression.py    # Compression utilities
│   └── validation.py     # Data validation
└── cli.py               # Command-line interface
```

## License

MIT License - see LICENSE file for details.
