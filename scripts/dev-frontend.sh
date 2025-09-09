#!/bin/bash
# Development script for React frontend

set -e  # Exit on any error

echo "Starting React frontend development server..."
cd frontend_web/agent-web

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Start development server
echo "Starting Vite dev server..."
echo "Frontend will be available at http://localhost:5173"
echo "Make sure the Python backend is running on http://127.0.0.1:8000"
echo ""
npm run dev