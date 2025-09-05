#!/bin/bash
# Development helper script for agent-template

set -e

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Initial project setup"
    echo "  server    - Start the agent server"
    echo "  client    - Start the terminal client"
    echo "  test      - Run tests"
    echo "  lint      - Run linting checks"
    echo "  format    - Format code"
    echo "  clean     - Clean up temporary files"
    echo "  install   - Install/update dependencies"
    echo ""
    exit 1
}

# Ensure we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from the project root directory"
    exit 1
fi

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found. Run './scripts/setup.sh' first."
    exit 1
fi

# Activate if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

case "$1" in
    setup)
        ./scripts/setup.sh
        ;;
    server)
        echo "Starting agent server..."
        uv run agent-server "${@:2}"
        ;;
    client)
        echo "Starting terminal client..."
        uv run agent-client "${@:2}"
        ;;
    test)
        echo "Running tests..."
        uv run pytest "${@:2}"
        ;;
    lint)
        echo "Running linting checks..."
        uv run flake8 src/ tests/
        uv run mypy src/agent_template
        ;;
    format)
        echo "Formatting code..."
        uv run black src/ tests/
        uv run isort src/ tests/
        ;;
    clean)
        echo "Cleaning up temporary files..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        find . -type f -name "*.pyo" -delete 2>/dev/null || true
        find . -type f -name "*.coverage" -delete 2>/dev/null || true
        rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/ 2>/dev/null || true
        echo "Cleanup complete."
        ;;
    install)
        echo "Installing/updating dependencies..."
        uv pip install -e .
        uv pip install -e ".[dev]"
        ;;
    *)
        usage
        ;;
esac