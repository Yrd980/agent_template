"""Terminal application using Rich and Textual."""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button, Footer, Header, Input, Log, RichLog, Static, TextArea
)

from ..models.messages import Message, MessageType, MessageRole, Session
from ..models.tasks import Task, TaskStatus, TaskType


class TaskTable(Static):
    """Widget to display current tasks."""
    
    def __init__(self) -> None:
        super().__init__()
        self.tasks: List[Task] = []
    
    def update_tasks(self, tasks: List[Task]) -> None:
        """Update the tasks list."""
        self.tasks = tasks
        self.refresh()
    
    def render(self) -> Table:
        """Render the tasks table."""
        table = Table(title="Active Tasks", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Type", style="green", width=12)
        table.add_column("Status", style="yellow", width=12)
        table.add_column("Priority", style="red", width=8)
        table.add_column("Progress", style="blue", width=10)
        table.add_column("Created", style="dim", width=16)
        
        for task in self.tasks:
            progress_text = f"{task.progress:.1%}" if task.progress > 0 else "-"
            created_text = task.created_at.strftime("%H:%M:%S")
            
            table.add_row(
                task.id[:8],
                task.type.value,
                task.status.value,
                str(task.priority.value),
                progress_text,
                created_text
            )
        
        return table


class ModelStatus(Static):
    """Widget to display model provider status."""
    
    def __init__(self) -> None:
        super().__init__()
        self.current_provider = "openai"
        self.model_stats: Dict[str, Any] = {}
    
    def update_status(self, provider: str, stats: Dict[str, Any]) -> None:
        """Update model status."""
        self.current_provider = provider
        self.model_stats = stats
        self.refresh()
    
    def render(self) -> Panel:
        """Render model status panel."""
        content = f"[bold green]Current Provider:[/bold green] {self.current_provider}\n"
        
        if self.model_stats:
            content += f"[bold blue]Tokens Used:[/bold blue] {self.model_stats.get('tokens_used', 0)}\n"
            content += f"[bold yellow]Requests:[/bold yellow] {self.model_stats.get('requests', 0)}\n"
            content += f"[bold red]Errors:[/bold red] {self.model_stats.get('errors', 0)}"
        
        return Panel(content, title="Model Status", border_style="blue")


class StreamOutput(RichLog):
    """Widget for streaming output display."""
    
    def __init__(self) -> None:
        super().__init__(auto_scroll=True, markup=True)
        self.max_lines = 1000
    
    def add_message(self, message: Message) -> None:
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message.type == MessageType.USER:
            self.write(f"[dim]{timestamp}[/dim] [bold blue]User:[/bold blue] {message.content}")
        elif message.type == MessageType.ASSISTANT:
            self.write(f"[dim]{timestamp}[/dim] [bold green]Agent:[/bold green] {message.content}")
        elif message.type == MessageType.SYSTEM:
            self.write(f"[dim]{timestamp}[/dim] [bold yellow]System:[/bold yellow] {message.content}")
        elif message.type == MessageType.TOOL_CALL:
            self.write(f"[dim]{timestamp}[/dim] [bold magenta]Tool:[/bold magenta] {message.content}")
        
        # Keep only recent lines
        if len(self.lines) > self.max_lines:
            self.clear()


class TodoList(Static):
    """Widget to display todo list."""
    
    def __init__(self) -> None:
        super().__init__()
        self.todos: List[Dict[str, Any]] = []
    
    def update_todos(self, todos: List[Dict[str, Any]]) -> None:
        """Update the todos list."""
        self.todos = todos
        self.refresh()
    
    def render(self) -> Panel:
        """Render todo list panel."""
        if not self.todos:
            content = "[dim]No todos[/dim]"
        else:
            lines = []
            for i, todo in enumerate(self.todos, 1):
                status = todo.get("status", "pending")
                content_text = todo.get("content", "")
                
                if status == "completed":
                    lines.append(f"[green]✓[/green] [dim]{content_text}[/dim]")
                elif status == "in_progress":
                    lines.append(f"[yellow]⚠[/yellow] [bold]{content_text}[/bold]")
                else:
                    lines.append(f"[red]○[/red] {content_text}")
            
            content = "\n".join(lines)
        
        return Panel(content, title="Todo List", border_style="green")


class AgentTerminalApp(App):
    """Main terminal application for the agent."""
    
    CSS = """
    #input-container {
        dock: bottom;
        height: 5;
    }
    
    #main-container {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    
    #chat-log {
        height: 1fr;
        border: solid $primary;
    }
    
    #tasks-panel {
        height: 1fr;
        border: solid $secondary;
    }
    
    #model-status {
        height: 1fr;
        border: solid $accent;
    }
    
    #todos-panel {
        height: 1fr;
        border: solid $success;
    }
    
    .input-field {
        dock: left;
        width: 1fr;
    }
    
    .send-button {
        dock: right;
        width: 10;
    }
    """
    
    # Reactive attributes
    current_session: reactive[Optional[str]] = reactive(None)
    connection_status: reactive[str] = reactive("disconnected")
    
    def __init__(self):
        super().__init__()
        self.console = Console()
        self.websocket = None
        self.session_id: Optional[str] = None
        
        # Components
        self.stream_output = StreamOutput()
        self.task_table = TaskTable()
        self.model_status = ModelStatus()
        self.todo_list = TodoList()
    
    def compose(self) -> ComposeResult:
        """Create the app layout."""
        yield Header(show_clock=True)
        
        with Container(id="main-container"):
            with Vertical():
                yield self.stream_output
            
            with Vertical():
                yield self.task_table
            
            with Vertical():
                yield self.model_status
            
            with Vertical():
                yield self.todo_list
        
        with Horizontal(id="input-container"):
            yield Input(placeholder="Enter your message...", id="message-input", classes="input-field")
            yield Button("Send", id="send-button", variant="primary", classes="send-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the app."""
        self.title = "Agent Template Terminal"
        self.sub_title = "AI Agent Interface"
        
        # Focus the input field
        self.query_one("#message-input").focus()
        
        # Start connection to backend
        self.run_worker(self.connect_to_backend(), exclusive=True)
    
    @on(Button.Pressed, "#send-button")
    async def send_message(self, event: Button.Pressed) -> None:
        """Handle send button click."""
        await self._send_current_message()
    
    @on(Input.Submitted, "#message-input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        await self._send_current_message()
    
    async def _send_current_message(self) -> None:
        """Send the current message."""
        input_widget = self.query_one("#message-input")
        message_text = input_widget.value.strip()
        
        if not message_text:
            return
        
        # Clear input
        input_widget.value = ""
        
        # Create message
        message = Message(
            type=MessageType.USER,
            content=message_text,
            session_id=self.session_id or "default"
        )
        
        # Display in chat log
        self.stream_output.add_message(message)
        
        # Send to backend
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": "message",
                    "data": message.model_dump()
                }))
            except Exception as e:
                error_msg = Message(
                    type=MessageType.SYSTEM,
                    content=f"Failed to send message: {e}",
                    session_id=self.session_id or "default"
                )
                self.stream_output.add_message(error_msg)
    
    async def connect_to_backend(self) -> None:
        """Connect to the backend WebSocket."""
        import websockets
        
        try:
            # Connect to backend
            self.websocket = await websockets.connect("ws://localhost:8000/ws")
            self.connection_status = "connected"
            
            # Listen for messages
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_backend_message(data)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.log(f"Error handling message: {e}")
        
        except Exception as e:
            self.connection_status = "error"
            error_msg = Message(
                type=MessageType.SYSTEM,
                content=f"Connection failed: {e}",
                session_id="system"
            )
            self.stream_output.add_message(error_msg)
    
    async def handle_backend_message(self, data: Dict[str, Any]) -> None:
        """Handle messages from the backend."""
        msg_type = data.get("type")
        
        if msg_type == "message":
            # Regular message
            msg_data = data.get("data", {})
            message = Message(**msg_data)
            self.stream_output.add_message(message)
        
        elif msg_type == "tasks_update":
            # Task list update
            tasks_data = data.get("data", [])
            tasks = [Task(**task_data) for task_data in tasks_data]
            self.task_table.update_tasks(tasks)
        
        elif msg_type == "model_status":
            # Model status update
            provider = data.get("provider", "unknown")
            stats = data.get("stats", {})
            self.model_status.update_status(provider, stats)
        
        elif msg_type == "todos_update":
            # Todo list update
            todos = data.get("data", [])
            self.todo_list.update_todos(todos)
        
        elif msg_type == "session_created":
            # Session created
            self.session_id = data.get("session_id")
            self.current_session = self.session_id
        
        elif msg_type == "stream_chunk":
            # Streaming response chunk
            chunk = data.get("data", "")
            # Add chunk to current message or create new one
            # This would need more sophisticated handling for real streaming
            
        elif msg_type == "error":
            # Error message
            error_msg = Message(
                type=MessageType.SYSTEM,
                content=f"Error: {data.get('message', 'Unknown error')}",
                session_id=self.session_id or "system"
            )
            self.stream_output.add_message(error_msg)
    
    def action_quit(self) -> None:
        """Quit the application."""
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        self.exit()
    
    def watch_connection_status(self, status: str) -> None:
        """Update UI based on connection status."""
        if status == "connected":
            self.sub_title = "Connected to Agent Backend"
        elif status == "disconnected":
            self.sub_title = "Disconnected"
        elif status == "error":
            self.sub_title = "Connection Error"


class TerminalClient:
    """Terminal client for the agent system."""
    
    def __init__(self):
        self.app = AgentTerminalApp()
    
    async def run(self) -> None:
        """Run the terminal client."""
        await self.app.run_async()
    
    def run_sync(self) -> None:
        """Run the terminal client synchronously."""
        self.app.run()


def main():
    """Main entry point for the terminal client."""
    client = TerminalClient()
    client.run_sync()


if __name__ == "__main__":
    main()
