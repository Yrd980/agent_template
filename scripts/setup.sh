#!/bin/bash
# Setup script for agent-template using uv

set -e

echo "Setting up Agent Template environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.11

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
uv pip install -e ".[dev]"

# Create data directories
echo "Creating data directories..."
mkdir -p data/cache
mkdir -p data/logs
mkdir -p data/sessions

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Model Configuration - Add your API keys here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Qwen Configuration  
QWEN_API_KEY=your_qwen_key_here
QWEN_MODEL=qwen-turbo
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Ollama Configuration (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Default provider (openai, anthropic, deepseek, qwen, ollama, local)
DEFAULT_MODEL_PROVIDER=openai

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Database
DATABASE_URL=sqlite:///./data/agent_template.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./data/logs/agent.log

# Agent Configuration
MAX_CONCURRENT_TASKS=10
MAX_CONTEXT_LENGTH=8000
CONTEXT_COMPRESSION_THRESHOLD=0.8
EOF
    echo "Created .env file. Please edit it with your API keys and settings."
fi

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the server, run:"
echo "  uv run agent-server"
echo ""
echo "To start the terminal client, run:"
echo "  uv run agent-client"