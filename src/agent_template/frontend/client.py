"""
Terminal client with Rich/Textual interface.

This module provides the main terminal interface for interacting with
the agent system, featuring real-time updates, rich formatting, and
interactive components.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Log, 
    ProgressBar, Static, TextArea, Tree as TextualTree
)
from textual.screen import Screen
from textual.binding import Binding
import websockets
import structlog

from ..config import settings
from ..models.messages import Message, MessageRole
from ..models.tasks import Task, TaskStatus
from .components import ChatPanel, TaskPanel, StatusPanel, MetricsPanel
from .widgets import StreamingOutput, TodoWidget, ToolCallWidget


logger = structlog.get_logger(__name__)


class MainScreen(Screen):
    """Main application screen."""
    
    CSS = """
    #header {
        dock: top;
        height: 3;
        background: $primary;
    }
    
    #footer {
        dock: bottom;
        height: 1;
        background: $primary;
    }
    
    #sidebar {
        dock: left;
        width: 30;
        background: $surface;
    }
    
    #main-content {
        background: $background;
    }
    
    #chat-panel {
        height: 70%;
    }
    
    #input-panel {
        dock: bottom;
        height: 10;
        background: $surface;
    }
    
    .panel {
        border: round $primary;
        margin: 1;
        padding: 1;
    }
    
    .highlight {
        background: $primary 20%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("ctrl+t", "toggle_sidebar", "Toggle Sidebar"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("f1", "show_tasks", "Tasks"),
        Binding("f2", "show_tools", "Tools"),
        Binding("f3", "show_metrics", "Metrics"),
    ]
    
    def __init__(self, client: "TerminalClient"):
        super().__init__()
        self.client = client
        self.sidebar_visible = True
        
    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        yield Header(show_clock=True, name="Agent Terminal")
        
        with Container():
            with Horizontal():
                # Sidebar
                with Vertical(id="sidebar"):
                    yield StatusPanel(self.client)
                    yield TaskPanel(self.client)
                    yield TodoWidget(self.client)
                
                # Main content area
                with Vertical(id="main-content"):
                    yield ChatPanel(self.client, id="chat-panel")
                    
                    # Input area
                    with Container(id="input-panel"):
                        yield Input(
                            placeholder="Type your message here...",
                            id="message-input"
                        )
                        with Horizontal():
                            yield Button("Send", variant="primary", id="send-button")
                            yield Button("Clear", variant="default", id="clear-button")
                            yield Button("Stop", variant="error", id="stop-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle screen mount."""
        self.query_one("#message-input", Input).focus()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "message-input":
            message = event.value.strip()
            if message:
                await self.client.send_message(message)
                event.input.value = ""
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "send-button":
            input_widget = self.query_one("#message-input", Input)
            message = input_widget.value.strip()
            if message:
                await self.client.send_message(message)
                input_widget.value = ""
        
        elif event.button.id == "clear-button":
            chat_panel = self.query_one("#chat-panel", ChatPanel)
            chat_panel.clear()
        
        elif event.button.id == "stop-button":
            await self.client.stop_current_task()
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.client.exit()
    
    def action_new_chat(self) -> None:
        """Start a new chat session."""
        self.client.new_session()
    
    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        sidebar = self.query_one("#sidebar")
        if self.sidebar_visible:
            sidebar.add_class("hidden")
            self.sidebar_visible = False
        else:
            sidebar.remove_class("hidden")
            self.sidebar_visible = True
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self.client.show_help()
    
    def action_show_tasks(self) -> None:
        """Show tasks screen."""
        self.client.show_tasks()
    
    def action_show_tools(self) -> None:
        """Show tools screen."""
        self.client.show_tools()
    
    def action_show_metrics(self) -> None:
        """Show metrics screen."""
        self.client.show_metrics()


class TasksScreen(Screen):
    """Screen for managing tasks."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+n", "new_task", "New Task"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]
    
    def __init__(self, client: "TerminalClient"):
        super().__init__()
        self.client = client
    
    def compose(self) -> ComposeResult:
        yield Header(name="Tasks")
        
        with Container():
            yield DataTable(id="tasks-table")
            
            with Horizontal():
                yield Button("New Task", variant="primary", id="new-task-btn")
                yield Button("Cancel Task", variant="error", id="cancel-task-btn")
                yield Button("Refresh", variant="default", id="refresh-btn")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Setup the tasks table."""
        table = self.query_one("#tasks-table", DataTable)
        table.add_columns("ID", "Type", "Status", "Created", "Progress")
        self.refresh_tasks()
    
    def refresh_tasks(self) -> None:
        """Refresh the tasks display."""
        # This would fetch tasks from the agent system
        # For now, show placeholder data
        table = self.query_one("#tasks-table", DataTable)
        table.clear()
        
        # Add sample tasks
        sample_tasks = [
            ("task_001", "chat", "completed", "10:30 AM", "100%"),
            ("task_002", "analysis", "running", "10:35 AM", "45%"),
            ("task_003", "research", "pending", "10:40 AM", "0%"),
        ]
        
        for task_data in sample_tasks:
            table.add_row(*task_data)
    
    def action_back(self) -> None:
        """Return to main screen."""
        self.app.pop_screen()
    
    def action_new_task(self) -> None:
        """Create a new task."""
        # Would show task creation dialog
        pass
    
    def action_refresh(self) -> None:
        """Refresh tasks."""
        self.refresh_tasks()


class MetricsScreen(Screen):
    """Screen for viewing metrics and statistics."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]
    
    def __init__(self, client: "TerminalClient"):
        super().__init__()
        self.client = client
    
    def compose(self) -> ComposeResult:
        yield Header(name="Metrics & Statistics")
        
        with Container():
            with Horizontal():
                with Vertical():
                    yield Static("System Metrics", id="system-metrics-title")
                    yield MetricsPanel(self.client, id="system-metrics")
                
                with Vertical():
                    yield Static("Performance Stats", id="performance-title") 
                    yield Static("", id="performance-stats")
        
        yield Footer()
    
    def action_back(self) -> None:
        """Return to main screen."""
        self.app.pop_screen()
    
    def action_refresh(self) -> None:
        """Refresh metrics."""
        metrics_panel = self.query_one("#system-metrics", MetricsPanel)
        metrics_panel.refresh()


class TerminalClient(App):
    """
    Main terminal client application.
    
    Provides a rich terminal interface for interacting with the agent system,
    including real-time chat, task management, metrics viewing, and more.
    """
    
    CSS_PATH = "styles.css"
    TITLE = "Agent Template Terminal"
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.server_host = server_host
        self.server_port = server_port
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
        # State
        self.session_id: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.connected = False
        
        # Message history
        self.messages: List[Message] = []
        self.tasks: List[Task] = []
        
        # Event callbacks
        self._message_handlers: List[Callable] = []
        self._task_handlers: List[Callable] = []
        self._status_handlers: List[Callable] = []
        
        logger.info("TerminalClient initialized", 
                   server_host=server_host, server_port=server_port)
    
    def compose(self) -> ComposeResult:
        """Compose the main application."""
        yield MainScreen(self)
    
    async def on_mount(self) -> None:
        """Handle application startup."""
        await self.connect_to_server()
    
    async def on_unmount(self) -> None:
        """Handle application shutdown."""
        await self.disconnect_from_server()
    
    # Connection management
    async def connect_to_server(self) -> bool:
        """Connect to the agent server."""
        try:
            ws_url = f"ws://{self.server_host}:{self.server_port}/ws"
            self.websocket = await websockets.connect(ws_url)
            self.connected = True
            
            # Start message handling
            self.create_task(self._message_loop())
            
            # Request session info
            await self._send_command("create_session", {})
            
            logger.info("Connected to agent server", url=ws_url)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to server", error=str(e))
            self.notify("Failed to connect to server", severity="error")
            return False
    
    async def disconnect_from_server(self) -> None:
        """Disconnect from the agent server."""
        if self.websocket and self.connected:
            try:
                await self.websocket.close()
                self.connected = False
                logger.info("Disconnected from agent server")
            except Exception as e:
                logger.error("Error disconnecting from server", error=str(e))
    
    async def _message_loop(self) -> None:
        """Handle incoming messages from the server."""
        while self.connected and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self._handle_server_message(data)
                
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                self.notify("Connection to server lost", severity="error")
                break
            except Exception as e:
                logger.error("Error in message loop", error=str(e))
    
    async def _handle_server_message(self, data: Dict[str, Any]) -> None:
        """Handle a message from the server."""
        message_type = data.get("type")
        
        if message_type == "message":
            await self._handle_chat_message(data)
        elif message_type == "task_update":
            await self._handle_task_update(data)
        elif message_type == "stream_data":
            await self._handle_stream_data(data)
        elif message_type == "status_update":
            await self._handle_status_update(data)
        elif message_type == "error":
            await self._handle_error_message(data)
        else:
            logger.warning("Unknown message type from server", type=message_type)
    
    async def _handle_chat_message(self, data: Dict[str, Any]) -> None:
        """Handle chat message from server."""
        message_data = data.get("message", {})
        message = Message(
            role=MessageRole(message_data.get("role", "assistant")),
            content=message_data.get("content", ""),
            session_id=self.session_id or "unknown"
        )
        
        self.messages.append(message)
        
        # Update chat panel
        if hasattr(self.screen, 'query_one'):
            try:
                chat_panel = self.screen.query_one("#chat-panel", ChatPanel)
                chat_panel.add_message(message)
            except:
                pass  # Panel might not be available
        
        # Notify handlers
        for handler in self._message_handlers:
            try:
                await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
            except Exception as e:
                logger.error("Error in message handler", error=str(e))
    
    async def _handle_task_update(self, data: Dict[str, Any]) -> None:
        """Handle task update from server."""
        task_data = data.get("task", {})
        
        # Update or create task
        task_id = task_data.get("id")
        existing_task = next((t for t in self.tasks if t.id == task_id), None)
        
        if existing_task:
            existing_task.status = TaskStatus(task_data.get("status", "pending"))
            existing_task.progress = task_data.get("progress", 0.0)
        else:
            # Create new task (simplified)
            pass
        
        # Notify handlers
        for handler in self._task_handlers:
            try:
                await handler(task_data) if asyncio.iscoroutinefunction(handler) else handler(task_data)
            except Exception as e:
                logger.error("Error in task handler", error=str(e))
    
    async def _handle_stream_data(self, data: Dict[str, Any]) -> None:
        """Handle streaming data from server."""
        stream_data = data.get("data", {})
        content = stream_data.get("content", "")
        
        if content:
            # Update streaming output widget
            try:
                if hasattr(self.screen, 'query_one'):
                    chat_panel = self.screen.query_one("#chat-panel", ChatPanel)
                    chat_panel.append_streaming_content(content)
            except:
                pass
    
    async def _handle_status_update(self, data: Dict[str, Any]) -> None:
        """Handle status update from server."""
        status_data = data.get("status", {})
        
        # Update status panel
        try:
            if hasattr(self.screen, 'query_one'):
                status_panel = self.screen.query_one(StatusPanel)
                status_panel.update_status(status_data)
        except:
            pass
        
        # Notify handlers
        for handler in self._status_handlers:
            try:
                await handler(status_data) if asyncio.iscoroutinefunction(handler) else handler(status_data)
            except Exception as e:
                logger.error("Error in status handler", error=str(e))
    
    async def _handle_error_message(self, data: Dict[str, Any]) -> None:
        """Handle error message from server."""
        error_msg = data.get("error", "Unknown error")
        self.notify(f"Server error: {error_msg}", severity="error")
        logger.error("Server error", error=error_msg)
    
    # User actions
    async def send_message(self, content: str) -> None:
        """Send a chat message to the server."""
        if not self.connected or not self.websocket:
            self.notify("Not connected to server", severity="error")
            return
        
        # Create user message
        user_message = Message(
            role=MessageRole.USER,
            content=content,
            session_id=self.session_id or "unknown"
        )
        
        self.messages.append(user_message)
        
        # Update chat panel immediately
        try:
            chat_panel = self.screen.query_one("#chat-panel", ChatPanel)
            chat_panel.add_message(user_message)
        except:
            pass
        
        # Send to server
        await self._send_command("send_message", {
            "content": content,
            "session_id": self.session_id
        })
        
        logger.debug("Sent message to server", content=content[:50])
    
    async def stop_current_task(self) -> None:
        """Stop the current running task."""
        if self.current_task_id:
            await self._send_command("cancel_task", {
                "task_id": self.current_task_id
            })
            self.notify("Task cancellation requested", severity="info")
    
    def new_session(self) -> None:
        """Start a new chat session."""
        self.session_id = None
        self.messages.clear()
        self.tasks.clear()
        
        # Clear chat panel
        try:
            chat_panel = self.screen.query_one("#chat-panel", ChatPanel)
            chat_panel.clear()
        except:
            pass
        
        # Request new session
        if self.connected:
            self.create_task(self._send_command("create_session", {}))
        
        self.notify("Started new session", severity="info")
    
    async def _send_command(self, command: str, data: Dict[str, Any]) -> None:
        """Send a command to the server."""
        if not self.websocket or not self.connected:
            return
        
        message = {
            "type": "command",
            "command": command,
            "data": data,
            "timestamp": time.time()
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send command", command=command, error=str(e))
    
    # Screen management
    def show_help(self) -> None:
        """Show help screen."""
        help_text = """
# Agent Terminal Help

## Keyboard Shortcuts
- Ctrl+C: Quit application
- Ctrl+N: New chat session
- Ctrl+T: Toggle sidebar
- Ctrl+H: Show this help
- F1: Show tasks
- F2: Show tools
- F3: Show metrics

## Chat Commands
- Type normally to chat with the agent
- Use /help for agent-specific help
- Use /clear to clear chat history

## Features
- Real-time streaming responses
- Task tracking and management
- Todo list integration
- Metrics and performance monitoring
- Tool calling and subagent support
        """
        
        self.push_screen(Screen(content=Static(help_text, id="help-content")))
    
    def show_tasks(self) -> None:
        """Show tasks screen."""
        self.push_screen(TasksScreen(self))
    
    def show_tools(self) -> None:
        """Show tools screen."""
        # Placeholder - would show available tools
        self.notify("Tools screen not implemented yet", severity="info")
    
    def show_metrics(self) -> None:
        """Show metrics screen."""
        self.push_screen(MetricsScreen(self))
    
    # Event registration
    def on_message(self, handler: Callable) -> None:
        """Register a message handler."""
        self._message_handlers.append(handler)
    
    def on_task_update(self, handler: Callable) -> None:
        """Register a task update handler."""
        self._task_handlers.append(handler)
    
    def on_status_update(self, handler: Callable) -> None:
        """Register a status update handler."""
        self._status_handlers.append(handler)
    
    # Utility methods
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        return {
            "session_id": self.session_id,
            "connected": self.connected,
            "messages": len(self.messages),
            "tasks": len(self.tasks),
            "current_task": self.current_task_id
        }


# Main entry point for CLI
async def main(
    host: str = "localhost",
    port: int = 8000,
    debug: bool = False
) -> None:
    """Run the terminal client."""
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    client = TerminalClient(server_host=host, server_port=port)
    
    try:
        await client.run_async()
    except KeyboardInterrupt:
        logger.info("Terminal client interrupted by user")
    except Exception as e:
        logger.error("Terminal client error", error=str(e))
        raise
    finally:
        logger.info("Terminal client stopped")


if __name__ == "__main__":
    import sys
    
    # Simple argument parsing
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    debug = "--debug" in sys.argv
    
    asyncio.run(main(host, port, debug))