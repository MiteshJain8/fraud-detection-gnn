#!/bin/bash

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

# Activate virtual environment
# Use source for Linux/macOS, .\venv\Scripts\Activate for Windows Git Bash/PowerShell
# This script assumes a Unix-like environment (Linux, macOS, Git Bash on Windows)
source .venv/bin/activate || . .venv/Scripts/activate || echo "Failed to activate virtual environment. Please activate manually."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Run the FastAPI server
echo "Starting FastAPI server..."
# Ensure the path to the app is correct relative to the backend directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
