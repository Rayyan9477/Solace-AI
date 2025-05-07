"""
Voice-enabled chat interface for the Mental Health Chatbot.
Provides a command-line interface for interacting with the chatbot using voice input.
"""

import os
import sys
import argparse
import logging
import asyncio
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.utils.voice_input_manager import VoiceInputManager
from src.agents.agent_orchestrator import AgentOrchestrator
from src.agents.chat_agent import ChatAgent
from src.agents.emotion_agent import EmotionAgent
from src.agents.safety_agent import SafetyAgent
from src.models.llm import AgnoLLM
from src.utils.voice_ai import VoiceManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceChat:
    """
    Voice-enabled chat interface for the Mental Health Chatbot.
    Provides a command-line interface for interacting with the chatbot.
    """
    
    def __init__(self, 
                 model_name: str = "turbo", 
                 voice_only: bool = False,
                 tts_enabled: bool = True,
                 voice_style: str = "warm"):
        """
        Initialize voice chat
        
        Args:
            model_name: Whisper model to use for speech recognition
            voice_only: Whether to use only voice input (no text input)
            tts_enabled: Whether to enable text-to-speech output
            voice_style: Voice style to use for TTS
        """
        self.model_name = model_name
        self.voice_only = voice_only
        self.tts_enabled = tts_enabled
        self.voice_style = voice_style
        self.voice_manager = None
        self.orchestrator = None
        self.is_running = False
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize chat components"""
        try:
            print("🤖 Initializing Mental Health Support Bot...")
            
            # Initialize LLM
            llm = AgnoLLM()
            
            # Initialize agents
            chat_agent = ChatAgent(model=llm)
            emotion_agent = EmotionAgent(model=llm)
            safety_agent = SafetyAgent(model=llm)
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(
                chat=chat_agent,
                emotion=emotion_agent,
                safety=safety_agent
            )
            
            # Initialize voice manager for both STT and TTS
            print("🎤 Initializing voice capabilities...")
            self.voice_manager = VoiceManager()
            voice_init_result = await self.voice_manager.initialize()
            
            if not voice_init_result["success"]:
                logger.warning(f"Voice initialization failed: {voice_init_result.get('error', 'Unknown error')}")
                print("⚠️ Voice capabilities could not be initialized. Falling back to text-only mode.")
                self.voice_only = False
                self.tts_enabled = False
            else:
                logger.info("Voice capabilities initialized successfully")
                print("✅ Voice capabilities initialized successfully.")
                
                # Initialize voice input manager if needed
                if self.voice_only or self.model_name:
                    print(f"🎤 Setting up voice recognition (Whisper {self.model_name})...")
                    self.voice_input_manager = VoiceInputManager(model_name=self.model_name)
                
                # Set voice style for TTS
                if self.voice_style and hasattr(self.voice_manager.dia_tts, "set_style"):
                    self.voice_manager.dia_tts.set_style(self.voice_style)
                    print(f"🔊 Voice style set to: {self.voice_style}")
            
            print("✅ Initialization complete!")
            print("\n💬 Mental Health Support Bot is ready to chat!")
            print("Type 'exit', 'quit', or 'bye' to end the conversation.")
            
            if self.voice_only:
                print("🎤 Voice-only mode enabled. Speak to interact.")
            elif self.voice_input_manager:
                print("🎤 Voice input available. Type 'voice' to use voice input, or 'text' to switch back to text input.")
            
            if self.tts_enabled:
                print("🔊 Text-to-speech output enabled. Type 'tts off' to disable, or 'tts on' to enable.")
                style_name = self.voice_style or "warm"
                print(f"   Voice style: {style_name} (Type 'voice-style [style]' to change)")
                if hasattr(self.voice_manager, "dia_tts") and self.voice_manager.dia_tts:
                    available_styles = list(self.voice_manager.dia_tts.voice_styles.keys())
                    print(f"   Available styles: {', '.join(available_styles)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chat: {str(e)}", exc_info=True)
            print(f"❌ Error initializing chat: {str(e)}")
            return False
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message
        
        Args:
            message: The user's message
            
        Returns:
            Response dictionary
        """
        try:
            # First, analyze emotions in the message
            emotion_result = await self.orchestrator.emotion.generate_response(message)
            logger.debug(f"Emotion analysis: {emotion_result}")
            
            # Then, run safety check
            safety_result = await self.orchestrator.safety.generate_response(
                message, 
                {"emotion": emotion_result}
            )
            logger.debug(f"Safety check: {safety_result}")
            
            # Build context from previous results
            context = {
                "emotion": emotion_result,
                "safety": safety_result
            }
            
            # Add personality context if available from history
            if self.conversation_history and len(self.conversation_history) > 2:
                # Extract any personality data from previous interactions
                for entry in reversed(self.conversation_history):
                    if "metadata" in entry and "personality" in entry["metadata"]:
                        context["personality"] = entry["metadata"]["personality"]
                        break
            
            # Add cultural context if we have detected it
            # This would come from previous interactions
            for entry in reversed(self.conversation_history):
                if "metadata" in entry and "cultural_context" in entry["metadata"]:
                    context["culture"] = entry["metadata"]["cultural_context"]
                    break
            
            # Generate chat response with all context
            response_result = await self.orchestrator.chat.generate_response(message, context)
            
            # Add emotion analysis to response metadata
            response_result["emotion_analysis"] = emotion_result
            response_result["safety_check"] = safety_result
            
            return response_result
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I'm having trouble responding right now. Could you try again?",
                "error": str(e)
            }
    
    async def handle_voice_input(self):
        """Handle voice input for one utterance"""
        print("\n🎤 Listening... (speak now)")
        result = self.voice_manager.transcribe_once()
        
        if result["success"] and result["text"]:
            user_input = result["text"].strip()
            print(f"🗣️ You said: {user_input}")
            return user_input
        else:
            print(f"❌ {result.get('error', 'Could not understand audio')}")
            return None
    
    async def run(self):
        """Run the chat interface"""
        if not await self.initialize():
            print("Failed to initialize chat. Exiting...")
            return
        
        self.is_running = True
        
        # Initial welcome message
        welcome_message = "Welcome! I'm here to listen and support you. How are you feeling today?"
        print(f"\n🧠 Assistant: {welcome_message}")
        
        # Speak welcome message if TTS is enabled
        if self.tts_enabled and self.voice_manager:
            try:
                await self.voice_manager.speak_text(welcome_message, self.voice_style)
            except Exception as e:
                logger.error(f"Error in TTS for welcome message: {str(e)}")
                print("⚠️ Text-to-speech failed for welcome message")
        
        # Input mode (text or voice)
        input_mode = "voice" if self.voice_only else "text"
        
        # Main chat loop
        while self.is_running:
            user_input = None
            
            # Get user input based on current mode
            if input_mode == "voice":
                user_input = await self.handle_voice_input()
                if not user_input:
                    continue
            else:  # text mode
                try:
                    user_input = input("\n💬 You: ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                goodbye_message = "Goodbye! Take care."
                print(goodbye_message)
                
                # Speak goodbye if TTS is enabled
                if self.tts_enabled and self.voice_manager:
                    await self.voice_manager.speak_text(goodbye_message, self.voice_style)
                    
                break
            
            # Check for mode switch commands
            if user_input.lower() == "voice" and not self.voice_only and self.voice_manager:
                input_mode = "voice"
                print("🎤 Switching to voice input mode. Speak to interact.")
                continue
            elif user_input.lower() == "text" and input_mode == "voice":
                input_mode = "text"
                print("⌨️ Switching to text input mode.")
                continue
            
            # Check for TTS commands
            if user_input.lower() == "tts off":
                self.tts_enabled = False
                print("🔇 Text-to-speech disabled.")
                continue
            elif user_input.lower() == "tts on":
                if self.voice_manager:
                    self.tts_enabled = True
                    print("🔊 Text-to-speech enabled.")
                else:
                    print("⚠️ Voice capabilities not available. Cannot enable TTS.")
                continue
            
            # Check for voice style commands
            if user_input.lower().startswith("voice-style "):
                style_name = user_input.lower().replace("voice-style ", "").strip()
                if self.voice_manager and self.voice_manager.dia_tts:
                    if self.voice_manager.dia_tts.set_style(style_name):
                        self.voice_style = style_name
                        print(f"🔊 Voice style changed to: {style_name}")
                    else:
                        available_styles = list(self.voice_manager.dia_tts.voice_styles.keys())
                        print(f"⚠️ Unknown voice style: {style_name}")
                        print(f"Available styles: {', '.join(available_styles)}")
                else:
                    print("⚠️ Advanced voice capabilities not available.")
                continue
            
            # Process message
            print("🤔 Processing...")
            start_time = time.time()
            result = await self.process_message(user_input)
            end_time = time.time()
            
            # Extract response
            response = result.get("response", "I'm not sure how to respond to that.")
            
            # Get emotion analysis if available
            emotion_info = ""
            emotion = "neutral"
            if "emotion_analysis" in result and result["emotion_analysis"]:
                emotion = result["emotion_analysis"].get("primary_emotion", "neutral")
                intensity = result["emotion_analysis"].get("intensity", 5)
                emotion_info = f" [Detected emotion: {emotion}, intensity: {intensity}/10]"
            
            # Display response
            print(f"\n🧠 Assistant: {response}{emotion_info}")
            
            # Speak response if TTS is enabled
            if self.tts_enabled and self.voice_manager:
                # Adapt voice style to match emotional context
                tts_style = self.voice_style
                if emotion in ["sad", "anxious", "depressed"] and "sad" in self.voice_manager.dia_tts.voice_styles:
                    tts_style = "sad"
                elif emotion in ["happy", "excited", "grateful"] and "excited" in self.voice_manager.dia_tts.voice_styles:
                    tts_style = "excited"
                elif emotion in ["angry", "frustrated"] and "professional" in self.voice_manager.dia_tts.voice_styles:
                    tts_style = "professional"
                
                try:
                    tts_result = await self.voice_manager.speak_text(response, tts_style)
                    if not tts_result:
                        logger.warning("Text-to-speech failed silently")
                except Exception as e:
                    logger.error(f"Error in text-to-speech: {str(e)}")
                    print("⚠️ Text-to-speech failed")
            
            # Add to history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "metadata": result
            })
            
            # Show processing time
            processing_time = end_time - start_time
            print(f"(Processed in {processing_time:.2f}s)")
        
        self.is_running = False

def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description="Voice-enabled Mental Health Chatbot")
    parser.add_argument(
        "--model", 
        default="turbo", 
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use for speech recognition"
    )
    parser.add_argument(
        "--voice-only", 
        action="store_true",
        help="Use only voice input (no text input)"
    )
    parser.add_argument(
        "--text-only", 
        action="store_true",
        help="Use only text input (no voice input)"
    )
    parser.add_argument(
        "--tts", 
        action="store_true",
        default=True,
        help="Enable text-to-speech for bot responses"
    )
    parser.add_argument(
        "--no-tts", 
        action="store_true",
        help="Disable text-to-speech for bot responses"
    )
    parser.add_argument(
        "--voice-style", 
        default="warm",
        choices=["default", "warm", "calm", "excited", "sad", "professional"],
        help="Voice style to use for text-to-speech"
    )
    
    args = parser.parse_args()
    
    # Check for incompatible options
    if args.voice_only and args.text_only:
        print("Error: Cannot specify both --voice-only and --text-only")
        return
    
    # Handle TTS options
    tts_enabled = args.tts and not args.no_tts
    
    # Select model based on args
    model = args.model if not args.text_only else None
    
    # Create and run chat
    chat = VoiceChat(
        model_name=model, 
        voice_only=args.voice_only,
        tts_enabled=tts_enabled,
        voice_style=args.voice_style
    )
    
    try:
        # Run the chat interface
        asyncio.run(chat.run())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()