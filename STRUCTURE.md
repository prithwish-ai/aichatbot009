# ITI Chatbot Package Structure

This document provides an overview of the ITI Chatbot package structure and organization.

## Directory Structure

```
iti-chatbot-package/
├── core/                     # Core functionality
│   ├── __init__.py           # Package initialization
│   ├── progress_tracker.py   # Progress tracking classes
│   └── progress_bar.py       # Progress visualization classes
│
├── examples/                 # Example scripts
│   ├── __init__.py           # Package initialization
│   ├── application_integration.py  # Example of integrating progress tracking
│   ├── chatbot_demo.py       # Simple chatbot demo
│   └── progress_demo.py      # Progress tracking demo
│
├── tests/                    # Test suite
│   ├── __init__.py           # Package initialization
│   └── test_progress_tracker.py  # Unit tests for progress tracking
│
├── .env.example              # Example environment variables
├── install.bat               # Windows installation script
├── install.sh                # Linux/macOS installation script
├── LICENSE                   # MIT License
├── README.md                 # Main documentation
├── requirements.txt          # Package dependencies
├── run_iti_app.py            # Application runner
├── setup.py                  # Package setup script
├── STRUCTURE.md              # This file
└── iti_app.py                # Main application file
```

## Core Components

### Progress Tracking System

The progress tracking system consists of several components:

- **ProgressTracker**: Tracks progress for a single task, calculating completion percentage, elapsed time, and estimated time to completion.
- **MultiProgressTracker**: Manages multiple related tasks, providing aggregated progress information.
- **ProgressBar**: Visualizes a single progress tracker as a console-based progress bar.
- **MultiProgressBarManager**: Manages multiple progress bars, ensuring they display correctly in the console.

### Main Application

The `iti_app.py` file contains the main application code, including:

- **ChatBot**: The main chatbot class that handles user interactions, AI responses, and features.
- **DocumentProcessingManager**: Handles document processing with progress tracking.
- **Various utility functions**: For handling voice, language translation, web search, etc.

## Usage Examples

The `examples/` directory contains several scripts demonstrating how to use the ITI Chatbot components:

- **application_integration.py**: Shows how to integrate progress tracking into a document processing application.
- **chatbot_demo.py**: Demonstrates a simplified version of the chatbot without requiring API keys.
- **progress_demo.py**: Shows various ways to use the progress tracking system.

## Installation and Setup

The package includes installation scripts for different platforms:

- **install.bat**: Windows installation script
- **install.sh**: Linux/macOS installation script

These scripts create a virtual environment, install dependencies, and set up the application.

## Configuration

The `.env.example` file shows the environment variables that can be configured:

- API keys for various services
- Application settings
- Feature toggles
- Path configurations

Copy this file to `.env` and update the values to configure the application. 