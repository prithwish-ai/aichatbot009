#!/bin/bash

echo "==================================="
echo "ITI Chatbot - Installation Script"
echo "==================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed!"
    echo "Please install Python 3.9 or higher."
    echo
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment!"
    echo "Make sure venv module is available."
    echo
    exit 1
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "==================================="
echo "Installation Complete!"
echo
echo "To run the application:"
echo "1. Open a terminal in this directory"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python run_iti_app.py"
echo "==================================="
echo

# Make the script executable
chmod +x run_iti_app.py 