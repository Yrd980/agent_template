"""Command line interface for the agent template."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import settings
from .frontend.terminal_app import TerminalClient
from .server import run_server
from .utils.logging_setup import setup_logging


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--log-level", default="INFO", help="Set log level")
def cli(debug: bool, log_level: str):
    """Agent Template CLI - AI Agent with Multi-Model Support."""
    # Update settings
    settings.debug = debug
    settings.logging.level = log_level
    
    # Setup logging
    setup_logging()


@cli.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
def server(host: Optional[str], port: Optional[int]):
    """Start the agent server."""
    click.echo("Starting Agent Template Server...")
    
    try:
        asyncio.run(run_server(host=host, port=port))
    except KeyboardInterrupt:
        click.echo("\nServer stopped by user")
    except Exception as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)


@cli.command()
def client():
    """Start the terminal client."""
    click.echo("Starting Agent Template Terminal Client...")
    
    try:
        client = TerminalClient()
        client.run_sync()
    except KeyboardInterrupt:
        click.echo("\nClient stopped by user")
    except Exception as e:
        click.echo(f"Client error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--provider", help="Model provider to test")
def test_model(provider: Optional[str]):
    """Test model connectivity."""
    from .services.model_provider import ModelManager
    
    async def run_test():
        manager = ModelManager()
        await manager.initialize()
        
        if provider:
            # Test specific provider
            if provider not in manager.providers:
                click.echo(f"Provider '{provider}' not available")
                return
            
            provider_obj = manager.providers[provider]
            try:
                response = await provider_obj.generate(
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    model=provider_obj.default_model
                )
                click.echo(f"✓ {provider}: {response.content}")
            except Exception as e:
                click.echo(f"✗ {provider}: {e}")
        else:
            # Test all providers
            for name, provider_obj in manager.providers.items():
                try:
                    response = await provider_obj.generate(
                        messages=[{"role": "user", "content": "Hello, world!"}],
                        model=provider_obj.default_model
                    )
                    click.echo(f"✓ {name}: {response.content[:50]}...")
                except Exception as e:
                    click.echo(f"✗ {name}: {e}")
    
    click.echo("Testing model providers...")
    asyncio.run(run_test())


@cli.command()
def init():
    """Initialize a new agent template project."""
    click.echo("Initializing Agent Template...")
    
    # Create data directories
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "cache").mkdir(exist_ok=True)
    (data_dir / "logs").mkdir(exist_ok=True)
    (data_dir / "sessions").mkdir(exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Model Configuration - Add your API keys here
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
"""
        env_file.write_text(env_content)
        click.echo("Created .env file with default configuration")
    
    click.echo("✓ Agent Template initialized successfully")
    click.echo("\nNext steps:")
    click.echo("1. Edit .env file with your API keys")
    click.echo("2. Run 'agent-template server' to start the server")
    click.echo("3. Run 'agent-template client' to start the terminal client")


@cli.command()
def status():
    """Show system status."""
    import aiohttp
    import asyncio
    
    async def check_status():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{settings.server.host}:{settings.server.port}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        click.echo("Server Status: ✓ Running")
                        components = data.get("components", {})
                        for name, status in components.items():
                            status_icon = "✓" if status else "✗"
                            click.echo(f"  {name}: {status_icon}")
                    else:
                        click.echo("Server Status: ✗ Error")
        except Exception as e:
            click.echo(f"Server Status: ✗ Not running ({e})")
    
    click.echo("Checking system status...")
    asyncio.run(check_status())


if __name__ == "__main__":
    cli()
