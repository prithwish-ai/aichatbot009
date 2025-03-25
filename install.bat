@echo off
echo ===================================
echo ITI Chatbot - Installation Script
echo ===================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.9 or higher from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment!
    echo Make sure venv module is available.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================
echo Installation Complete!
echo.
echo To run the application:
echo 1. Open a command prompt in this directory
echo 2. Run: venv\Scripts\activate.bat
echo 3. Run: python run_iti_app.py
echo ===================================
echo.

pause 