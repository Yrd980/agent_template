#!/bin/bash
# Build React frontend script

set -e  # Exit on any error

echo "Building React frontend..."
cd frontend_web/agent-web

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Build for production
echo "Building React app for production..."
npm run build

echo "✅ Frontend build complete!"
echo "Built files are in frontend_web/agent-web/dist/"
echo ""
echo "You can now start the Python server with: agent-server"
echo "The web interface will be available at http://127.0.0.1:8000"