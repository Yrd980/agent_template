"""Main server application with FastAPI and WebSocket support."""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

from .api.routes import APIDependencies, router as api_router, set_dependencies
from .api.websocket import WebSocketManager
from .config import settings
from .core.agent_loop import AgentLoop
from .core.async_queue import AsyncQueue
from .services.model_provider import ModelManager
from .services.session_manager import SessionManager
from .services.tool_manager import ToolManager
from .utils.logging_setup import setup_logging
from .services.state_cache import StateCache


logger = logging.getLogger(__name__)


class AgentServer:
    """Main agent server with FastAPI and WebSocket support."""
    
    def __init__(self):
        self.app: Optional[FastAPI] = None
        self.agent_loop: Optional[AgentLoop] = None
        self.session_manager: Optional[SessionManager] = None
        self.tool_manager: Optional[ToolManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.state_cache: Optional[StateCache] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self.async_queue: Optional[AsyncQueue] = None
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing agent server...")
        
        # Initialize core components
        self.state_cache = StateCache()
        
        self.async_queue = AsyncQueue()
        
        self.session_manager = SessionManager(state_cache=self.state_cache)
        
        self.model_manager = ModelManager()
        
        self.tool_manager = ToolManager()
        
        self.agent_loop = AgentLoop()
        
        # Initialize WebSocket manager
        self.websocket_manager = WebSocketManager(
            agent_loop=self.agent_loop,
            session_manager=self.session_manager
        )
        
        logger.info("Agent server initialized successfully")
    
    async def create_app(self) -> FastAPI:
        """Create and configure FastAPI app."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """FastAPI lifespan manager."""
            # Startup
            await self.initialize()
            
            # Set dependencies for API routes
            deps = APIDependencies(
                agent_loop=self.agent_loop,
                session_manager=self.session_manager,
                tool_manager=self.tool_manager,
                state_cache=self.state_cache
            )
            set_dependencies(deps)
            
            # Start background tasks
            background_tasks = [
                asyncio.create_task(self.agent_loop.run()),
                asyncio.create_task(self._cleanup_task()),
                asyncio.create_task(self._heartbeat_task())
            ]
            
            logger.info("Agent server started")
            
            try:
                yield
            finally:
                # Shutdown
                logger.info("Shutting down agent server...")
                self.shutdown_event.set()
                
                # Cancel background tasks
                for task in background_tasks:
                    task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*background_tasks, return_exceptions=True)
                
                # Cleanup components
                if self.agent_loop:
                    await self.agent_loop.shutdown()
                if self.session_manager:
                    await self.session_manager.shutdown()
                if self.tool_manager:
                    await self.tool_manager.stop()
                if self.model_manager:
                    await self.model_manager.shutdown()
                if self.state_cache:
                    await self.state_cache.stop()
                
                logger.info("Agent server shutdown complete")
        
        # Create FastAPI app
        app = FastAPI(
            title="Agent Template Server",
            description="AI Agent Template with Multi-Model Support",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include API routes
        app.include_router(api_router)
        
        # Static files serving for React frontend
        frontend_dist_path = Path(__file__).parent.parent.parent / "frontend_web" / "agent-web" / "dist"
        if frontend_dist_path.exists():
            # Serve static files
            app.mount("/static", StaticFiles(directory=str(frontend_dist_path / "assets")), name="static")
            
            # Serve React app for all non-API routes
            @app.get("/{full_path:path}")
            async def serve_react_app(full_path: str):
                # Don't serve React app for API routes or WebSocket
                if full_path.startswith(("api/", "ws", "health")):
                    return {"error": "Not found"}
                
                # Serve index.html for all other routes (React Router will handle routing)
                index_path = frontend_dist_path / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
                return {"error": "Frontend not built"}
        else:
            logger.warning("React frontend not found. Run 'npm run build' in frontend_web/agent-web/")
        
        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            connection_id = await self.websocket_manager.connect(websocket)
            
            try:
                while True:
                    # Receive message
                    message = await websocket.receive_text()
                    await self.websocket_manager.handle_message(connection_id, message)
            
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await self.websocket_manager.disconnect(connection_id)
        
        # Health check endpoints
        @app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Agent Template Server",
                "version": "1.0.0",
                "status": "running"
            }
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            try:
                agent_loop_status = self.agent_loop.is_running if self.agent_loop else False
            except AttributeError as e:
                logger.error("AttributeError accessing agent_loop.is_running", error=str(e))
                agent_loop_status = False

            return {
                "status": "healthy",
                "components": {
                    "agent_loop": agent_loop_status,
                    "session_manager": bool(self.session_manager),
                    "tool_manager": bool(self.tool_manager),
                    "model_manager": bool(self.model_manager),
                    "websocket_manager": bool(self.websocket_manager),
                }
            }
        
        self.app = app
        return app
    
    async def _cleanup_task(self):
        """Background task for cleanup operations."""
        while not self.shutdown_event.is_set():
            try:
                # Cleanup dead WebSocket connections
                if self.websocket_manager:
                    await self.websocket_manager.cleanup_dead_connections()
                
                # Cleanup expired sessions
                if self.session_manager:
                    await self.session_manager.cleanup_expired_sessions()
                
                # Cleanup completed tasks
                if self.agent_loop:
                    await self.agent_loop.cleanup_completed_tasks()
                
                # Wait before next cleanup
                await asyncio.sleep(60)  # Run every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_task(self):
        """Background task for sending heartbeats."""
        while not self.shutdown_event.is_set():
            try:
                # Send WebSocket heartbeat
                if self.websocket_manager:
                    await self.websocket_manager.send_heartbeat()
                
                # Wait before next heartbeat
                await asyncio.sleep(30)  # Send every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)
    
    async def run(self, host: str = None, port: int = None):
        """Run the server."""
        import uvicorn
        
        # Use settings defaults if not provided
        host = host or settings.server.host
        port = port or settings.server.port
        
        # Create app
        app = await self.create_app()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=settings.server.reload,
            log_level=settings.logging.level.lower(),
            access_log=True,
        )
        
        # Create and run server
        server = uvicorn.Server(config)
        
        # Handle shutdown signals using asyncio
        loop = asyncio.get_running_loop()

        def signal_handler(signum):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()
            server.should_exit = True

        loop.add_signal_handler(signal.SIGTERM, lambda: signal_handler(signal.SIGTERM))
        loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(signal.SIGINT))

        await server.serve()


# Global server instance
_server: Optional[AgentServer] = None


def get_server() -> AgentServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = AgentServer()
    return _server


async def run_server(host: str = None, port: int = None):
    """Run the agent server."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting Agent Template Server...")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Model Provider: {settings.models.default_provider}")
    
    # Create and run server
    server = get_server()
    await server.run(host=host, port=port)


def main():
    """Main entry point for the server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
