#!/usr/bin/env python3
"""
ITI Chatbot Application Runner

This script initializes and runs the ITI Chatbot application.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

def setup_environment():
    """Set up the environment by loading .env file and creating directories."""
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories
    dirs = [
        os.getenv('DATA_DIR', './data'),
        os.getenv('LOG_DIR', './logs'),
        os.getenv('CACHE_DIR', './cache')
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    # Check for API keys
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not found in environment or .env file.")
        print("The application will not be able to use AI features without this key.")
        print("Please edit the .env file and add your API key.")
        choice = input("Would you like to continue anyway? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ITI Chatbot Application')
    parser.add_argument('--no-voice', action='store_true', help='Disable voice features')
    parser.add_argument('--language', type=str, default=None, help='Set default language (e.g., en, hi)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str, default=None, help='Specify AI model to use')
    return parser.parse_args()

def main():
    """Main function to run the ITI Chatbot application."""
    # Set up environment
    setup_environment()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set environment variables from arguments
    if args.no_voice:
        os.environ['ENABLE_VOICE'] = 'false'
    if args.language:
        os.environ['DEFAULT_LANGUAGE'] = args.language
    if args.debug:
        os.environ['DEBUG_MODE'] = 'true'
    if args.model:
        os.environ['AI_MODEL'] = args.model
    
    try:
        # Import after environment setup to ensure settings are applied
        from iti_app import main as app_main
        app_main()
    except ImportError:
        print("Error: Could not import the ITI Chatbot application.")
        print("Please make sure the application is installed correctly.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running application: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 