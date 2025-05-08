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
# Import new personality and memory modules
from src.personality.chatbot_personality import ChatbotPersonality
from src.utils.conversation_memory import ConversationMemory
from src.diagnosis.comprehensive_diagnosis import ComprehensiveDiagnosis

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
                 voice_style: str = "warm",
                 user_id: str = "default_user",
                 personality_type: str = "supportive_counselor"):
        """
        Initialize voice chat
        
        Args:
            model_name: Whisper model to use for speech recognition
            voice_only: Whether to use only voice input (no text input)
            tts_enabled: Whether to enable text-to-speech output
            voice_style: Voice style to use for TTS
            user_id: Unique identifier for the user
            personality_type: Type of personality for the chatbot
        """
        self.model_name = model_name
        self.voice_only = voice_only
        self.tts_enabled = tts_enabled
        self.voice_style = voice_style
        self.user_id = user_id
        self.personality_type = personality_type
        
        self.voice_manager = None
        self.orchestrator = None
        self.voice_input_manager = None
        self.is_running = False
        self.conversation_history = []
        self.user_profile = {}
        self.last_emotion_detected = "neutral"
        self.last_interaction_time = None
        
        # Initialize new personality and memory components
        self.personality = ChatbotPersonality(personality_name=personality_type)
        self.memory = ConversationMemory(user_id=user_id)
        self.diagnosis = None  # Will be initialized later

    async def initialize(self) -> bool:
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
            
            # Initialize diagnosis module
            self.diagnosis = ComprehensiveDiagnosis(model=llm)
            
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
                
                # Initialize voice input manager with Whisper V3 Turbo
                if self.voice_only or self.model_name:
                    print(f"🎙️ Setting up enhanced speech recognition (Whisper {self.model_name})...")
                    self.voice_input_manager = VoiceInputManager(
                        model_name=self.model_name,
                        analyze_emotions=True,  # Enable emotion analysis in speech
                        use_audeering=True      # Use audeering for better emotion detection
                    )
                
                # Set voice style for TTS
                if self.voice_style and hasattr(self.voice_manager.dia_tts, "set_style"):
                    self.voice_manager.dia_tts.set_style(self.voice_style)
                    print(f"🔊 Voice style set to: {self.voice_style}")
                    
            # Load user profile from memory
            user_profile = self.memory.get_user_profile()
            if user_profile:
                self.user_profile = user_profile
                print("📋 Loaded user profile from memory.")
            
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
        Process a user message with emotional intelligence and memory
        
        Args:
            message: The user's message
            
        Returns:
            Response dictionary with emotional context
        """
        try:
            # Update last interaction time
            self.last_interaction_time = time.time()
            
            # First, analyze emotions in the message and voice (if available)
            emotion_result = await self.orchestrator.emotion.generate_response(message)
            logger.debug(f"Emotion analysis: {emotion_result}")
            
            # Store the detected emotion for context
            if emotion_result and "primary_emotion" in emotion_result:
                self.last_emotion_detected = emotion_result["primary_emotion"]
                
                # Adjust personality based on detected emotion
                personality_adjustments = self.personality.adjust_for_emotion(self.last_emotion_detected)
                
                # Update voice style if needed
                if "voice_style" in personality_adjustments and self.tts_enabled:
                    voice_style = personality_adjustments["voice_style"].get("tone", self.voice_style)
                    if voice_style != self.voice_style and hasattr(self.voice_manager.dia_tts, "set_style"):
                        self.voice_manager.dia_tts.set_style(voice_style)
            
            # Then, run safety check
            safety_result = await self.orchestrator.safety.generate_response(
                message, 
                {"emotion": emotion_result}
            )
            logger.debug(f"Safety check: {safety_result}")
            
            # Get relevant conversation history from memory
            conversation_context = self.memory.format_context_for_prompt(query=message)
            
            # Add personality information to context
            personality_context = self.personality.format_for_prompt()
            
            # Build context from previous results
            context = {
                "emotion": emotion_result,
                "safety": safety_result,
                "conversation_history": conversation_context,
                "personality": personality_context
            }
            
            # Add user profile information to context
            if self.user_profile:
                context["user_profile"] = self.user_profile
            
            # Run diagnostic analysis if we have enough context
            if len(self.conversation_history) > 2:
                try:
                    # Build context for diagnosis from recent conversations
                    diagnosis_input = "\n".join([
                        f"User: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}"
                        for entry in self.conversation_history[-5:]
                    ]) + f"\nUser: {message}"
                    
                    diagnosis_result = await self.diagnosis.analyze(
                        diagnosis_input, 
                        context={"emotion": emotion_result}
                    )
                    
                    if diagnosis_result:
                        context["diagnosis"] = diagnosis_result
                        
                        # Extract topics for memory
                        if "topics" in diagnosis_result:
                            topics = diagnosis_result["topics"]
                            logger.info(f"Detected topics: {topics}")
                except Exception as e:
                    logger.error(f"Error in diagnostic analysis: {str(e)}")
            
            # Generate chat response with all context
            response_result = await self.orchestrator.chat.generate_response(message, context)
            
            # Add emotion analysis to response metadata
            response_result["emotion_analysis"] = emotion_result
            response_result["safety_check"] = safety_result
            
            # Store conversation in memory
            self.memory.add_conversation(
                user_message=message,
                assistant_response=response_result.get("response", ""),
                metadata={
                    "emotion_analysis": emotion_result,
                    "safety_check": safety_result,
                    "diagnosis": context.get("diagnosis", {})
                }
            )
            
            return response_result
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I'm having trouble processing right now. Could you try again or rephrase that?",
                "error": str(e)
            }

    async def handle_voice_input(self) -> Optional[str]:
        """Handle voice input for one utterance using Whisper V3 Turbo"""
        print("\n🎤 Listening... (speak now)")
        
        try:
            # Use voice input manager for better emotion detection if available
            if self.voice_input_manager:
                result = await self.voice_input_manager.transcribe_once_with_emotion()
                
                # Extract speech emotion data if available
                if result.get("success") and "emotion" in result and result["emotion"].get("success"):
                    emotion_data = result["emotion"]
                    self.last_emotion_detected = emotion_data.get("primary_emotion", "neutral")
                    confidence = emotion_data.get("confidence", 0)
                    
                    # Adjust personality based on detected voice emotion
                    if confidence > 0.4:
                        personality_adjustments = self.personality.adjust_for_emotion(self.last_emotion_detected)
                        print(f"🔍 Detected voice emotion: {self.last_emotion_detected} ({confidence:.2f} confidence)")
            else:
                # Fall back to regular voice manager
                result = self.voice_manager.transcribe_once()
            
            if result.get("success") and result.get("text"):
                user_input = result["text"].strip()
                print(f"🗣️ You said: {user_input}")
                return user_input
            else:
                print(f"❌ {result.get('error', 'Could not understand audio')}")
                return None
                
        except Exception as e:
            logger.error(f"Error handling voice input: {str(e)}")
            print("❌ Error processing voice input. Please try again.")
            return None

    async def run(self):
        """Run the voice-enabled chat interface with improved interaction"""
        if not await self.initialize():
            print("Failed to initialize chat. Exiting...")
            return
        
        self.is_running = True
        
        # Get personalized introduction message
        welcome_message = self.personality.get_introduction()
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
        
        # Conversation idle timer
        last_active_time = time.time()
        
        # Main chat loop
        while self.is_running:
            user_input = None
            
            # Check for conversation idle timeout (2 minutes)
            current_time = time.time()
            if current_time - last_active_time > 120 and len(self.conversation_history) > 0:
                # Provide a gentle prompt if user has been quiet
                print("\n🧠 Assistant: Are you still there? I'm here to continue our conversation whenever you're ready.")
                if self.tts_enabled and self.voice_manager:
                    await self.voice_manager.speak_text("Are you still there? I'm here to continue our conversation whenever you're ready.", "gentle")
                last_active_time = current_time
            
            # Get user input based on current mode
            if input_mode == "voice":
                user_input = await self.handle_voice_input()
                if not user_input:
                    # If voice input failed, wait a moment before trying again
                    await asyncio.sleep(1)
                    continue
            else:  # text mode
                try:
                    user_input = input("\n💬 You: ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
            
            # Reset idle timer on new input
            last_active_time = time.time()
            
            # Check for exit commands
            if user_input and user_input.lower() in ["exit", "quit", "bye"]:
                goodbye_message = "Goodbye! Take care of yourself. Remember I'm here whenever you need to talk."
                print(f"\n🧠 Assistant: {goodbye_message}")
                
                # Speak goodbye if TTS is enabled
                if self.tts_enabled and self.voice_manager:
                    await self.voice_manager.speak_text(goodbye_message, "gentle")
                    
                break
            
            # Check for mode switch commands
            if user_input and user_input.lower() == "voice" and not self.voice_only and self.voice_manager:
                input_mode = "voice"
                print("🎤 Switching to voice input mode. Speak to interact.")
                continue
            elif user_input and user_input.lower() == "text" and input_mode == "voice":
                input_mode = "text"
                print("⌨️ Switching to text input mode.")
                continue
            
            # Check for TTS commands
            if user_input and user_input.lower() == "tts off":
                self.tts_enabled = False
                print("🔇 Text-to-speech disabled.")
                continue
            elif user_input and user_input.lower() == "tts on":
                if self.voice_manager:
                    self.tts_enabled = True
                    print("🔊 Text-to-speech enabled.")
                else:
                    print("⚠️ Voice capabilities not available. Cannot enable TTS.")
                continue
            
            # Check for voice style commands
            if user_input and user_input.lower().startswith("voice-style "):
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
            
            # Skip empty inputs
            if not user_input:
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
            
            # Speak response if TTS is enabled with emotion-appropriate voice
            if self.tts_enabled and self.voice_manager:
                # Get appropriate voice style based on user's emotion
                tts_style = self.personality.get_voice_style_for_emotion(emotion)
                
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
                "metadata": result,
                "timestamp": time.time()
            })
            
            # Update user profile with any new insights
            self._update_user_profile(result)
            
            # Show processing time if not too slow
            processing_time = end_time - start_time
            if processing_time < 10:  # Only show if reasonably fast to avoid drawing attention to slowness
                print(f"(Processed in {processing_time:.2f}s)")
        
        self.is_running = False
        
        # Save user profile and conversation context before exiting
        try:
            self.memory.update_user_profile(self.user_profile)
            print("📋 Saved user profile and conversation history.")
        except Exception as e:
            logger.error(f"Error saving user profile: {str(e)}")
    
    def _update_user_profile(self, result: Dict[str, Any]) -> None:
        """Update user profile with insights from the conversation"""
        # Extract personality insights if available
        if "personality_insights" in result:
            self.user_profile.update(result["personality_insights"])
        
        # Extract preferences if mentioned
        if "preferences" in result:
            if "preferences" not in self.user_profile:
                self.user_profile["preferences"] = {}
            self.user_profile["preferences"].update(result["preferences"])
            
            # Also update the personality based on user preferences
            if hasattr(self.personality, "adjust_to_user_preferences"):
                self.personality.adjust_to_user_preferences(result["preferences"])
        
        # Update emotional patterns
        if "emotion_analysis" in result and result["emotion_analysis"]:
            if "emotional_patterns" not in self.user_profile:
                self.user_profile["emotional_patterns"] = {}
            
            emotion = result["emotion_analysis"].get("primary_emotion")
            if emotion:
                # Count occurrences of emotions
                self.user_profile["emotional_patterns"][emotion] = self.user_profile["emotional_patterns"].get(emotion, 0) + 1
                
        # Extract diagnostic insights
        if "diagnosis" in result:
            diagnosis_data = result["diagnosis"]
            
            # Update user profile with diagnostic insights
            if "diagnosis_results" not in self.user_profile:
                self.user_profile["diagnosis_results"] = {}
                
            # Store latest diagnosis
            self.user_profile["diagnosis_results"]["latest"] = diagnosis_data
            
            # Update topics
            if "topics" in diagnosis_data:
                if "topics" not in self.user_profile:
                    self.user_profile["topics"] = {}
                
                for topic in diagnosis_data["topics"]:
                    self.user_profile["topics"][topic] = self.user_profile["topics"].get(topic, 0) + 1

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
    parser.add_argument(
        "--user-id",
        default="default_user",
        help="Unique identifier for the user (for conversation memory)"
    )
    parser.add_argument(
        "--personality-type",
        default="supportive_counselor",
        choices=["supportive_counselor", "empathetic_listener", "analytical_advisor"],
        help="Type of personality for the chatbot"
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
        voice_style=args.voice_style,
        user_id=args.user_id,
        personality_type=args.personality_type
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