#!/usr/bin/env python3
"""
Test Voice Integration

This script tests the enhanced voice integration to ensure it works correctly.
"""

import os
import sys
import time
from voice_integration import VoiceInteractionManager

def main():
    """Test the voice integration."""
    print("Testing Enhanced Voice Integration")
    print("==================================")
    
    # Initialize the voice manager
    voice_manager = VoiceInteractionManager()
    
    if not voice_manager.is_available:
        print("Voice interaction is not available. Exiting.")
        return
    
    # Test speaking
    print("\nTesting text-to-speech...")
    voice_manager.speak("Hello! This is a test of the enhanced voice integration. I will now listen for your response.")
    
    # Test listening
    print("\nTesting speech recognition...")
    print("Please say something when prompted.")
    time.sleep(1)
    
    user_input = voice_manager.listen(timeout=5)
    
    if user_input:
        print(f"You said: {user_input}")
        voice_manager.speak(f"I heard you say: {user_input}. The voice integration is working correctly.")
    else:
        print("No speech detected or recognition failed.")
        voice_manager.speak("I couldn't understand what you said, but the voice integration is still working.")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 