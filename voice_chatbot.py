"""
Interactive Voice Chatbot

This script implements a voice chatbot that:
1. Converts text to speech output using gTTS
2. Speaks audibly before listening for input
3. Provides clear feedback about its state
"""

import time
import os
import sys
import tempfile
import subprocess
import uuid
from gtts import gTTS
import speech_recognition as sr
import pygame  # Use pygame for cross-platform audio playback

class SimpleVoiceChatbot:
    def __init__(self):
        """Initialize the voice chatbot with basic settings"""
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # Get available microphones
        self.mic_index = 0  # Default microphone
        try:
            mics = sr.Microphone.list_microphone_names()
            print(f"Available microphones:")
            for i, mic in enumerate(mics):
                print(f"{i}: {mic}")
        except Exception as e:
            print(f"Error listing microphones: {e}")
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.quit()  # Ensure mixer is closed before initializing
            pygame.mixer.init()
            print("Audio system initialized successfully")
        except Exception as e:
            print(f"Error initializing audio: {e}")
        
        # Tracking state
        self.is_speaking = False
        self.is_listening = False
        
        # Speech settings
        self.speech_speed = 2.0  # 2x normal speed
        print(f"Speech speed set to {self.speech_speed}x")
        
        # Create temp directory for audio files
        self.temp_dir = os.path.join(tempfile.gettempdir(), "voice_chatbot")
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Using temp directory: {self.temp_dir}")
    
    def speak(self, text, language="en"):
        """Speak text and wait until completely finished before returning"""
        if not text:
            return False
        
        print(f"\n💬 TEXT TO SPEECH: \"{text}\"")
        self.is_speaking = True
        
        # Create a unique filename in the temp directory
        unique_id = str(uuid.uuid4())[:8]
        filename = os.path.join(self.temp_dir, f"tts_output_{unique_id}.mp3")
        
        try:
            # Generate speech with gTTS
            print("⚙️ Generating speech...")
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filename)
            print(f"✓ Speech file saved to: {filename}")
            
            # Play the audio file
            print("🔊 Playing speech...")
            success = self._play_audio(filename)
            
            if success:
                print("✓ Audio playback completed")
            else:
                print("❌ Audio playback failed")
            
            # Add a short pause after speaking finishes
            time.sleep(0.5)
            
            return success
        except Exception as e:
            print(f"❌ Speech error: {e}")
            return False
        finally:
            self.is_speaking = False
            # Clean up the file if it exists
            try:
                # Make sure pygame is done with the file
                pygame.mixer.music.unload()
                time.sleep(0.2)  # Small delay to ensure file is released
                
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"✓ Removed temporary file: {filename}")
            except Exception as e:
                print(f"❌ Error removing file: {e}")
            print("👂 Now ready to listen")
    
    def _play_audio(self, audio_file):
        """Play audio using pygame"""
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
            
        try:
            # Make sure nothing is playing
            pygame.mixer.music.stop()
            
            # Load and play the file
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            return True
        except Exception as e:
            print(f"❌ Playback error: {e}")
            
            # Try alternative method if pygame fails
            try:
                print("⚠️ Trying alternative playback method...")
                if sys.platform == 'win32':
                    os.startfile(audio_file)
                    # Approximate wait time for audio to play
                    time.sleep(len(text) / 10)  # Rough estimate: 10 chars per second
                else:
                    subprocess.run(['xdg-open', audio_file], check=True)
                    time.sleep(len(text) / 10)
                return True
            except Exception as alt_e:
                print(f"❌ Alternative playback also failed: {alt_e}")
            
            return False
    
    def listen(self, timeout=10, phrase_time_limit=10):
        """Listen for speech input but only after speaking has completely finished"""
        # Double-check we're not speaking
        if self.is_speaking:
            print("Cannot listen while speaking")
            return None
            
        self.is_listening = True
        print("\n🎤 Listening for input...")
        
        try:
            # Use specified microphone
            with sr.Microphone(device_index=self.mic_index) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                try:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                    
                    # Recognize speech
                    print("🔍 Processing speech...")
                    text = self.recognizer.recognize_google(audio)
                    
                    if text and text.strip():
                        print(f"🗣️ You said: \"{text}\"")
                        return text.strip()
                    else:
                        print("❓ Couldn't understand audio")
                        return None
                except sr.WaitTimeoutError:
                    print("⏰ No speech detected (timeout)")
                    return None
                except sr.UnknownValueError:
                    print("❓ Speech not understood")
                    return None
                except Exception as e:
                    print(f"❌ Recognition error: {e}")
                    return None
                
        except Exception as e:
            print(f"❌ Listening error: {e}")
            return None
        finally:
            self.is_listening = False
    
    def set_microphone(self, index):
        """Set the microphone device index"""
        try:
            self.mic_index = int(index)
            # Test the microphone
            with sr.Microphone(device_index=self.mic_index) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            print(f"✓ Microphone set to index {index}")
            return True
        except Exception as e:
            print(f"❌ Error setting microphone {index}: {e}")
            return False
    
    def run(self):
        """Run the chatbot in a conversation loop"""
        print("\n==================================================")
        print("🤖 INTERACTIVE VOICE CHATBOT STARTED")
        print("==================================================")
        print("✓ Text-to-Speech: Enabled")
        print("✓ Speech Recognition: Enabled")
        print("✓ Speech Speed: 2x normal speed")
        print("✓ Interactive Mode: The chatbot will speak first, then listen")
        print("==================================================\n")
        
        # Initial greeting
        success = self.speak("Voice chatbot is now active. I'll speak to you and then listen for your response.")
        
        if not success:
            print("\n❌ CRITICAL ERROR: Text-to-speech failed on startup.")
            print("Please check that:")
            print("1. You have internet connection (gTTS requires internet)")
            print("2. You have audio output device connected and working")
            print("3. Your system has a media player installed")
            print("\nTrying one more time with a shorter message...")
            
            if not self.speak("Testing voice output."):
                print("\n❌ CRITICAL ERROR: Voice output is not working.")
                print("The chatbot will continue in text-only mode.")
        
        while True:
            try:
                # Get user input after speaking has finished
                user_input = self.listen()
                
                if not user_input:
                    print("Waiting for input...")
                    continue
                
                # Simple responses
                if "hello" in user_input.lower():
                    self.speak("Hello there! How can I help you today?")
                
                elif "bye" in user_input.lower() or "exit" in user_input.lower() or "quit" in user_input.lower():
                    self.speak("Goodbye! Have a great day.")
                    break
                    
                elif "who are you" in user_input.lower():
                    self.speak("I'm a voice chatbot with interactive text to speech capabilities. I'll always speak to you and then wait for your response.")
                    
                else:
                    self.speak(f"You said: {user_input}. I'm responding with my voice at twice the normal speed.")
                
            except KeyboardInterrupt:
                print("\n🛑 Exiting voice chatbot...")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                
        # Clean up on exit
        try:
            pygame.mixer.quit()
        except:
            pass

# Run the chatbot if executed directly
if __name__ == "__main__":
    print("Starting Interactive Voice Chatbot")
    print("Press Ctrl+C to exit")
    
    # Initialize and run the chatbot
    chatbot = SimpleVoiceChatbot()
    chatbot.run()