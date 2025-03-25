"""
Voice Integration Module

This module provides an enhanced voice interaction manager for the ITI application.
It imports and extends the EnhancedVoiceManager from the core module.
"""

# Import the EnhancedVoiceManager from the core module
from core.voice_integration import EnhancedVoiceManager
from colorama import Fore, Style
import os
import json

class VoiceInteractionManager(EnhancedVoiceManager):
    """
    VoiceInteractionManager acts as a bridge between the application and the
    EnhancedVoiceManager, providing a simplified interface for voice interactions.
    """
    
    def __init__(self):
        """Initialize the voice interaction manager."""
        super().__init__()
        
        # Initialize settings
        self.settings_path = "voice_settings.json"
        self.settings = {
            "wake_word": None,
            "volume": 0.8,
            "rate": 150,
            "voice_id": "default",
            "continuous_listening": False
        }
        
        # Load settings if available
        self._load_settings()
        
        # Set wake word to the desired value
        self.set_wake_word("utho iti")
        
        # Any additional initialization can be added here
        # This extends the EnhancedVoiceManager with app-specific functionality
        
        # Set up default event handlers
        self._setup_event_handlers()
    
    def _load_settings(self):
        """Load voice settings from file."""
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    # Update settings with saved values
                    for key, value in saved_settings.items():
                        if key in self.settings:
                            self.settings[key] = value
                print(f"{Fore.GREEN}[VOICE] ✓ Loaded voice settings{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}[VOICE] Could not load voice settings: {e}{Style.RESET_ALL}")
    
    def _save_settings(self):
        """Save voice settings to file."""
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"{Fore.YELLOW}[VOICE] Could not save voice settings: {e}{Style.RESET_ALL}")
    
    def set_wake_word(self, wake_word=None):
        """Set wake word for voice activation.
        
        Args:
            wake_word: Wake word or phrase (None to disable)
            
        Returns:
            True if successful, False otherwise
        """
        self.settings["wake_word"] = wake_word
        print(f"{Fore.CYAN}[VOICE] Wake word set to: '{wake_word}'{Style.RESET_ALL}")
        self._save_settings()
        return True
    
    def listen_for_wake_word(self, timeout=3):
        """Listen specifically for the wake word.
        
        Args:
            timeout: Maximum time to wait for wake word
            
        Returns:
            True if wake word detected, False otherwise
        """
        if not self.is_available or not self.settings["wake_word"]:
            return False
        
        print(f"{Fore.GREEN}[VOICE] Listening for wake word: '{self.settings['wake_word']}'...{Style.RESET_ALL}")
        
        text = self.listen(timeout=timeout)
        if not text:
            return False
        
        # Check if wake word is in the recognized text
        wake_word_detected = self.settings["wake_word"].lower() in text.lower()
        
        if wake_word_detected:
            print(f"{Fore.GREEN}[VOICE] ✓ Wake word detected!{Style.RESET_ALL}")
        
        return wake_word_detected
    
    def _setup_event_handlers(self):
        """Set up default event handlers for voice events."""
        # Set up event handlers with debug messages
        self.on("speech_started", self._on_speech_started)
        self.on("speech_completed", self._on_speech_completed)
        self.on("recognition_started", self._on_recognition_started)
        self.on("recognition_completed", self._on_recognition_completed)
        self.on("error_occurred", self._on_error_occurred)
    
    def _on_speech_started(self, data):
        """Handle speech started event."""
        print(f"{Fore.CYAN}[VOICE DEBUG] 🔊 Speaking started{Style.RESET_ALL}")
    
    def _on_speech_completed(self, data):
        """Handle speech completed event."""
        print(f"{Fore.CYAN}[VOICE DEBUG] ✓ Speaking completed{Style.RESET_ALL}")
    
    def _on_recognition_started(self, data):
        """Handle recognition started event."""
        print(f"{Fore.GREEN}[VOICE DEBUG] 🎤 Listening... (Speak now){Style.RESET_ALL}")
    
    def _on_recognition_completed(self, data):
        """Handle recognition completed event."""
        if data and "text" in data and data["text"]:
            print(f"{Fore.GREEN}[VOICE DEBUG] ✓ Voice recognized: \"{data['text']}\"{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[VOICE DEBUG] ⚠ No speech detected{Style.RESET_ALL}")
    
    def _on_error_occurred(self, data):
        """Handle error events."""
        if data and "error" in data:
            error_msg = data["error"]
            source = data.get("source", "unknown")
            print(f"{Fore.RED}[VOICE DEBUG] ❌ Error in {source}: {error_msg}{Style.RESET_ALL}")
    
    # Override listen method to add debug messages
    def listen(self, language="en-US", timeout=5):
        """Listen for speech input with debug messages.
        
        Args:
            language: Language code for speech recognition
            timeout: Maximum time to wait for speech input in seconds
            
        Returns:
            Recognized text or None if recognition failed
        """
        print(f"{Fore.GREEN}[VOICE DEBUG] 🎤 Starting to listen in {language} (timeout: {timeout}s)...{Style.RESET_ALL}")
        
        # Call the parent class listen method
        try:
            # Set a timeout for listening to ensure typing is still possible
            result = super().listen(language=language, timeout=timeout)
            
            if not result:
                print(f"{Fore.YELLOW}[VOICE DEBUG] ⚠ No speech recognized{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}[VOICE DEBUG] ✓ Successfully recognized speech{Style.RESET_ALL}")
                
            return result
        except Exception as e:
            print(f"{Fore.RED}[VOICE DEBUG] ❌ Error during speech recognition: {str(e)}{Style.RESET_ALL}")
            return None
    
    # Override speak method to add debug messages
    def speak(self, text, language="en"):
        """Speak text with debug messages.
        
        Args:
            text: Text to speak
            language: Language code
            
        Returns:
            Success flag
        """
        if not text:
            print(f"{Fore.YELLOW}[VOICE DEBUG] ⚠ Empty text, nothing to speak{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}[VOICE DEBUG] 🔊 Speaking in {language}... ({len(text)} chars){Style.RESET_ALL}")
        
        # For long text, break it into manageable chunks
        if len(text) > 500:
            print(f"{Fore.CYAN}[VOICE DEBUG] Long text detected, breaking into chunks for reliable playback{Style.RESET_ALL}")
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            success = True
            
            # Group paragraphs into reasonable chunks (around 500 chars)
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < 500:
                    current_chunk += para + "\n\n"
                else:
                    # Speak the current chunk
                    if current_chunk:
                        try:
                            print(f"{Fore.CYAN}[VOICE DEBUG] Speaking chunk ({len(current_chunk)} chars)...{Style.RESET_ALL}")
                            super().speak(current_chunk, language=language)
                        except Exception as e:
                            print(f"{Fore.RED}[VOICE DEBUG] ❌ Error during speech chunk: {str(e)}{Style.RESET_ALL}")
                            success = False
                    
                    # Start a new chunk
                    current_chunk = para + "\n\n"
            
            # Speak the final chunk
            if current_chunk:
                try:
                    print(f"{Fore.CYAN}[VOICE DEBUG] Speaking final chunk ({len(current_chunk)} chars)...{Style.RESET_ALL}")
                    super().speak(current_chunk, language=language)
                except Exception as e:
                    print(f"{Fore.RED}[VOICE DEBUG] ❌ Error during speech final chunk: {str(e)}{Style.RESET_ALL}")
                    success = False
                    
            return success
        
        # For shorter text, use the original method
        try:
            result = super().speak(text, language=language)
            return result
        except Exception as e:
            print(f"{Fore.RED}[VOICE DEBUG] ❌ Error during speech: {str(e)}{Style.RESET_ALL}")
            return False
    
    # Add debug for start_continuous_listening
    def start_continuous_listening(self, language="en-US"):
        """Start continuous listening with debug messages.
        
        Args:
            language: Language code for speech recognition
            
        Returns:
            VoiceSession instance or None if failed
        """
        print(f"{Fore.GREEN}[VOICE DEBUG] 🎤 Starting continuous listening mode in {language}...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[VOICE DEBUG] Speak at any time. Session will continue until stopped.{Style.RESET_ALL}")
        
        return super().start_continuous_listening(language=language)
    
    # Add debug for stop_continuous_listening
    def stop_continuous_listening(self):
        """Stop continuous listening with debug messages."""
        print(f"{Fore.YELLOW}[VOICE DEBUG] 🛑 Stopping continuous listening mode{Style.RESET_ALL}")
        
        return super().stop_continuous_listening() 