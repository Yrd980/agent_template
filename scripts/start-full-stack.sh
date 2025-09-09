#!/bin/bash
# Start full stack (backend + built frontend)

set -e  # Exit on any error

echo "🚀 Starting Agent Template Full Stack..."

# Build frontend if dist doesn't exist
if [ ! -d "frontend_web/agent-web/dist" ]; then
    echo "Frontend not built. Building now..."
    ./scripts/build-frontend.sh
fi

# Start the Python server (which will serve both API and React app)
echo "Starting Python backend server..."
echo ""
echo "🌐 Web interface will be available at: http://127.0.0.1:8000"
echo "📡 WebSocket endpoint: ws://127.0.0.1:8000/ws"
echo "🔧 API endpoints: http://127.0.0.1:8000/api/v1/*"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

agent-server