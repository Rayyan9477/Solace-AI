import time
import random
import logging
from typing import Dict, Any, List, Optional, Callable

# Import personality assessment modules using relative imports
from ..personality.big_five import BigFiveAssessment
from ..personality.mbti import MBTIAssessment

logger = logging.getLogger(__name__)

class DynamicPersonalityAssessment:
    """
    Component for conducting dynamic, conversational personality assessments
    with voice and emotion integration - refactored for API/CLI use
    """
    
    def __init__(
        self, 
        personality_agent,
        voice_component=None,
        emotion_agent=None,
        on_complete=None
    ):
        """
        Initialize the dynamic personality assessment component
        
        Args:
            personality_agent: PersonalityAgent instance for assessment logic
            voice_component: Optional VoiceComponent instance
            emotion_agent: Optional EmotionAgent for emotional analysis
            on_complete: Callback function to call when assessment is complete
        """
        self.personality_agent = personality_agent
        self.voice_component = voice_component
        self.emotion_agent = emotion_agent
        self.on_complete_callback = on_complete
        
        # Set up assessment state
        self.state = {
            "assessment_type": None,  # 'big_five' or 'mbti'
            "conversation_mode": False,  # True for conversational, False for traditional
            "current_question_index": 0,
            "responses": {},
            "conversation_history": [],
            "emotion_data": {},
            "adaptations": {},
            "voice_mode": False,
            "auto_speak": False,
            "voice_style": "warm",
            "current_traits_focus": [],  # Traits currently being assessed
            "assessment_progress": 0.0,  # Progress from 0.0 to 1.0
            "user_context": {},  # Additional context about the user
            "last_emotion": "neutral"
        }
        
        # Initialize the assessment managers
        self.big_five_assessment = BigFiveAssessment()
        self.mbti_assessment = MBTIAssessment()
        
        # Set up adapters for conversational flow
        self.trait_conversation_starters = {
            # Big Five conversation starters
            "extraversion": [
                "Tell me about the last time you were at a large social gathering. How did you feel?",
                "Do you find yourself energized after spending time with others, or do you need time alone to recharge?",
                "When faced with a group activity, do you prefer to take the lead or observe first?"
            ],
            "agreeableness": [
                "How do you typically handle disagreements with friends or colleagues?",
                "When someone asks for your help with something inconvenient, what's your usual response?",
                "What's your approach to giving feedback when someone has made a mistake?"
            ],
            "conscientiousness": [
                "How do you approach deadlines and planning for important tasks?",
                "Describe your ideal workspace or environment for getting things done.",
                "When starting a new project, what's your usual process?"
            ],
            "neuroticism": [
                "How do you typically respond to unexpected changes or disruptions to your plans?",
                "Tell me about how you handle stressful situations.",
                "What helps you feel calm when you're worried about something?"
            ],
            "openness": [
                "What's your approach to trying new experiences or unfamiliar activities?",
                "How important is it for you to have variety in your day-to-day life?",
                "When you encounter ideas that challenge your worldview, how do you typically respond?"
            ],
            
            # MBTI conversation starters
            "E/I": [
                "How do you prefer to spend your free time - with others or by yourself?",
                "When solving a problem, do you think better by talking it through with someone or reflecting on your own?",
                "What feels more natural to you - having many shorter conversations or a few deeper ones?"
            ],
            "S/N": [
                "When learning something new, do you prefer practical examples or understanding the theory behind it?",
                "When planning a trip, what aspects do you focus on the most?",
                "Do you find yourself more interested in what is actually happening now or what could happen in the future?"
            ],
            "T/F": [
                "How do you typically make important decisions in your life?",
                "When a friend comes to you with a problem, what's your first instinct - to offer solutions or emotional support?",
                "In a debate, what matters more to you - the logical consistency of your argument or maintaining harmony in the group?"
            ],
            "J/P": [
                "How far in advance do you usually plan your activities?",
                "How do you feel about last-minute changes to your plans?",
                "Do you prefer having a structured routine or keeping your options open?"
            ]
        }

    def render(self):
        """Render the dynamic personality assessment UI"""
        import streamlit as st
        import asyncio
        
        # Initialize state in session if needed
        if "dynamic_assessment_state" not in st.session_state:
            st.session_state["dynamic_assessment_state"] = self.state.copy()
        
        state = st.session_state["dynamic_assessment_state"]
        
        # Step 1: Choose assessment type if not already chosen
        if state["assessment_type"] is None:
            st.header("Personalized Assessment Experience")
            
            # Create layout with columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Big Five (OCEAN) Assessment
                
                Measures 5 key personality dimensions:
                - **Openness** to experience
                - **Conscientiousness**
                - **Extraversion**
                - **Agreeableness**
                - **Neuroticism**
                """)
                
                if st.button("Take Big Five Assessment"):
                    state["assessment_type"] = "big_five"
                    state["current_traits_focus"] = ["extraversion", "openness"]  # Start with these traits
                    st.rerun()
            
            with col2:
                st.markdown("""
                ### Myers-Briggs Type Indicator (MBTI)
                
                Identifies your personality type across 4 dimensions:
                - **Extraversion (E)** vs. **Introversion (I)**
                - **Sensing (S)** vs. **Intuition (N)**
                - **Thinking (T)** vs. **Feeling (F)**
                - **Judging (J)** vs. **Perceiving (P)**
                """)
                
                if st.button("Take MBTI Assessment"):
                    state["assessment_type"] = "mbti"
                    state["current_traits_focus"] = ["E/I", "S/N"]  # Start with these dimensions
                    st.rerun()
            
            # Choose assessment mode
            st.markdown("### Assessment Style")
            
            mode_col1, mode_col2 = st.columns(2)
            
            with mode_col1:
                st.markdown("""
                #### Traditional Questionnaire
                
                Answer structured questions with clear options.
                Faster and more straightforward.
                """)
                
                if st.button("Choose Traditional Mode"):
                    state["conversation_mode"] = False
                    # Will be set once assessment type is chosen
            
            with mode_col2:
                st.markdown("""
                #### Conversational Assessment
                
                Have a natural dialogue that feels more like a conversation 
                than a formal assessment. More engaging but takes longer.
                """)
                
                if st.button("Choose Conversational Mode"):
                    state["conversation_mode"] = True
                    # Will be set once assessment type is chosen
            
            # Voice options (if available)
            if self.voice_ai or self.whisper_voice_manager:
                st.markdown("### Voice Interaction")
                
                voice_col1, voice_col2 = st.columns(2)
                
                with voice_col1:
                    voice_enabled = st.toggle("Enable Voice Interaction", value=False)
                    state["voice_mode"] = voice_enabled
                
                with voice_col2:
                    if voice_enabled:
                        auto_speak = st.toggle("Automatically speak questions", value=True)
                        state["auto_speak"] = auto_speak
                        
                        # Voice style selector
                        if self.voice_ai:
                            voice_styles = ["warm", "neutral", "professional", "empathetic", "energetic"]
                            selected_style = st.selectbox("Voice Style", voice_styles, index=0)
                            state["voice_style"] = selected_style
            
            return
        
        # Assessment is in progress
        if state["assessment_type"] and not state.get("assessment_complete", False):
            # Display progress
            st.progress(state["assessment_progress"])
            
            # Voice controls if voice is enabled
            if state["voice_mode"] and (self.voice_ai or self.whisper_voice_manager):
                with st.expander("Voice Controls", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.toggle("Enable Voice Input", value=True, key="voice_input_enabled")
                        
                        # Voice input button for Whisper
                        if self.whisper_voice_manager and st.session_state.get("voice_input_enabled", True):
                            if st.button("🎙️ Speak Your Answer"):
                                with st.spinner("Listening..."):
                                    result = self.whisper_voice_manager.transcribe_once()
                                    if result["success"] and result["text"]:
                                        st.session_state["voice_response"] = result["text"]
                                        st.rerun()
                    
                    with col2:
                        st.toggle("Enable Voice Output", value=state["auto_speak"], key="voice_output_enabled")
                        
                        if st.session_state.get("voice_output_enabled", True):
                            voice_styles = ["warm", "neutral", "professional", "empathetic", "energetic"]
                            state["voice_style"] = st.selectbox(
                                "Voice Style", 
                                voice_styles, 
                                index=voice_styles.index(state["voice_style"]) if state["voice_style"] in voice_styles else 0
                            )
            
            if state["conversation_mode"]:
                # Render conversational assessment
                self._render_conversational_assessment()
            else:
                # Render traditional assessment
                self._render_traditional_assessment()
            
            return
        
        # Assessment is complete
        if state.get("assessment_complete", False):
            st.success("Assessment Complete!")
            
            # Call completion callback if provided
            if self.on_complete_callback:
                self.on_complete_callback(state.get("results", {}))
            
            # Show option to retake
            if st.button("Start a New Assessment"):
                # Reset state
                st.session_state["dynamic_assessment_state"] = self.state.copy()
                st.rerun()
    
    def _render_traditional_assessment(self):
        """Render the traditional questionnaire assessment"""
        import streamlit as st
        import asyncio
        
        state = st.session_state["dynamic_assessment_state"]
        
        # Get the appropriate assessment
        assessment = self.big_five_assessment if state["assessment_type"] == "big_five" else self.mbti_assessment
        
        # Get current questions if not already in state
        if "current_questions" not in state:
            if state["assessment_type"] == "big_five":
                # For Big Five, get 50 questions (10 per trait)
                state["current_questions"] = assessment.get_questions(50)
                state["total_questions"] = len(state["current_questions"])
            else:
                # For MBTI, get 36 questions (9 per dimension)
                state["current_questions"] = assessment.get_questions(36)
                state["total_questions"] = len(state["current_questions"])
        
        # Calculate progress
        state["assessment_progress"] = min(1.0, state["current_question_index"] / max(1, state["total_questions"]))
        
        # Display current question
        if state["current_question_index"] < len(state["current_questions"]):
            current_q = state["current_questions"][state["current_question_index"]]
            
            # Check if we should adapt the question
            if state["current_question_index"] in state["adaptations"]:
                # Use the adapted question text
                question_text = state["adaptations"][state["current_question_index"]]
            else:
                question_text = current_q["text"]
            
            st.subheader(f"Question {state['current_question_index'] + 1} of {state['total_questions']}")
            
            # If voice is enabled, speak the question
            if state["voice_mode"] and state["auto_speak"] and self.voice_ai:
                # Use a key to avoid re-speaking on reruns
                question_key = f"spoken_q_{state['current_question_index']}"
                if question_key not in st.session_state:
                    self.voice_ai.text_to_speech(
                        question_text,
                        voice_style=state["voice_style"]
                    )
                    st.session_state[question_key] = True
            
            # Display the question
            st.markdown(f"**{question_text}**")
            
            # Display options based on assessment type
            if state["assessment_type"] == "big_five":
                # Big Five uses 1-5 scale
                response = st.radio(
                    "Your response:",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: [
                        "Very Inaccurate", 
                        "Moderately Inaccurate", 
                        "Neither Accurate Nor Inaccurate", 
                        "Moderately Accurate", 
                        "Very Accurate"
                    ][x-1],
                    horizontal=True,
                    key=f"response_{state['current_question_index']}"
                )
                
                # Store the response
                if st.button("Next Question"):
                    q_id = str(current_q["id"])
                    state["responses"][q_id] = response
                    
                    # Capture emotion if available
                    if self.emotion_agent:
                        # TODO: Emotion capture logic
                        pass
                    
                    # Handle question adaptation for future questions
                    self._adapt_upcoming_questions()
                    
                    # Move to the next question
                    state["current_question_index"] += 1
                    st.rerun()
            else:
                # MBTI uses A/B options
                options = current_q.get("options", [])
                if options:
                    response = st.radio(
                        "Choose the option that better describes you:",
                        options=options,
                        format_func=lambda x: f"{x['key']}: {x['text']}",
                        key=f"response_{state['current_question_index']}"
                    )
                    
                    # Store the response
                    if st.button("Next Question"):
                        q_id = str(current_q["id"])
                        state["responses"][q_id] = response["key"]
                        
                        # Capture emotion if available
                        if self.emotion_agent:
                            # TODO: Emotion capture logic
                            pass
                        
                        # Handle question adaptation for future questions
                        self._adapt_upcoming_questions()
                        
                        # Move to the next question
                        state["current_question_index"] += 1
                        st.rerun()
                else:
                    st.error("Question options not found")
        else:
            # All questions answered, calculate results
            st.success("All questions completed!")
            
            with st.spinner("Computing your personality profile..."):
                # Use asyncio to call the async method
                results = asyncio.run(self._compute_results())
                
                # Store results and mark as complete
                state["results"] = results
                state["assessment_complete"] = True
                
                st.rerun()
    
    def _render_conversational_assessment(self):
        """Render the conversational assessment experience"""
        import streamlit as st
        import asyncio
        import time
        import random
        
        state = st.session_state["dynamic_assessment_state"]
        
        # Display conversation history
        st.subheader("Your Personality Conversation")
        
        # Create a chat-like interface
        for entry in state["conversation_history"]:
            if entry["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(entry["content"])
            else:
                with st.chat_message("user"):
                    st.write(entry["content"])
        
        # Determine what question to ask next if we don't have one queued
        if "next_question" not in state or not state["next_question"]:
            if not state["conversation_history"]:
                # First question - introduction
                if state["assessment_type"] == "big_five":
                    state["next_question"] = (
                        "Hi there! I'd like to learn more about your personality through a friendly conversation. "
                        "To start, could you tell me a bit about how you typically approach social situations? "
                        "For example, do you prefer being in large groups or one-on-one interactions?"
                    )
                else:  # MBTI
                    state["next_question"] = (
                        "Hi there! I'd like to learn about your personality preferences through our conversation. "
                        "To start, I'm curious about how you recharge your energy. Do you prefer spending time with others "
                        "or having time to yourself? Could you tell me a bit about that?"
                    )
            else:
                # Generate next question based on current focus traits
                trait = random.choice(state["current_traits_focus"])
                question_options = self.trait_conversation_starters.get(trait, [])
                
                if question_options:
                    state["next_question"] = random.choice(question_options)
                else:
                    # Fallback question
                    state["next_question"] = "Could you tell me more about how you typically handle challenges in your life?"
        
        # Display next question from assistant
        if "next_question" in state and state["next_question"]:
            with st.chat_message("assistant"):
                st.write(state["next_question"])
                
                # Speak the question if voice enabled
                if state["voice_mode"] and state["auto_speak"] and self.voice_ai:
                    # Use a key to avoid re-speaking on reruns
                    question_key = f"spoken_conv_{len(state['conversation_history'])}"
                    if question_key not in st.session_state:
                        self.voice_ai.text_to_speech(
                            state["next_question"],
                            voice_style=state["voice_style"]
                        )
                        st.session_state[question_key] = True
            
            # Add to conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []
                
            state["conversation_history"].append({
                "role": "assistant",
                "content": state["next_question"],
                "timestamp": time.time()
            })
            
            # Clear the next question to avoid duplication
            state["next_question"] = None
            
        # Get user input
        user_input = None
        
        # Check for voice input first
        if "voice_response" in st.session_state:
            user_input = st.session_state["voice_response"]
            del st.session_state["voice_response"]
        
        # Text input as backup
        if not user_input:
            user_input = st.chat_input("Type your response here...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add to conversation history
            state["conversation_history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Process the response
            with st.spinner("Analyzing your response..."):
                # Extract personality insights from the response
                # This would be an async call in a real implementation
                trait_scores = asyncio.run(self._analyze_conversation_response(user_input))
                
                # Update state with new scores
                if "trait_scores" not in state:
                    state["trait_scores"] = {}
                
                # Update trait scores
                for trait, score in trait_scores.items():
                    if trait not in state["trait_scores"]:
                        state["trait_scores"][trait] = []
                    state["trait_scores"][trait].append(score)
                
                # Update progress - increment by a fixed amount per response
                increment = 0.1  # 10% progress per response
                state["assessment_progress"] = min(0.9, state["assessment_progress"] + increment)
                
                # Generate follow-up question based on the response
                state["next_question"] = asyncio.run(self._generate_follow_up_question(user_input))
                
                # Switch focus to other traits after a few exchanges
                if len(state["conversation_history"]) % 4 == 0:
                    self._rotate_trait_focus()
                
                # Check if we have enough data to complete the assessment
                if len(state["conversation_history"]) >= 10 and state["assessment_progress"] >= 0.9:
                    # Finalize the assessment
                    st.success("I've gathered enough information about your personality!")
                    
                    with st.spinner("Computing your personality profile..."):
                        # Use asyncio to call the async method
                        results = asyncio.run(self._finalize_conversational_assessment())
                        
                        # Store results and mark as complete
                        state["results"] = results
                        state["assessment_complete"] = True
                        
                        st.rerun()
                
                st.rerun()
    
    async def _analyze_conversation_response(self, response: str):
        """
        Analyze a conversational response to extract personality trait indicators
        
        Args:
            response: The user's text response
            
        Returns:
            Dictionary mapping traits to scores
        """
        import asyncio
        
        state = st.session_state["dynamic_assessment_state"]
        
        trait_scores = {}
        
        # Get context for prompt
        assessment_type = state["assessment_type"]
        conversation_history = state["conversation_history"][-5:]  # Last 5 exchanges
        
        # Format the context for the prompt
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])
        
        # Get traits to analyze based on assessment type
        traits_to_analyze = []
        if assessment_type == "big_five":
            traits_to_analyze = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
        else:  # MBTI
            traits_to_analyze = ["E/I", "S/N", "T/F", "J/P"]
        
        try:
            # Create a prompt for the LLM
            system_prompt = f"""
            You are an expert personality psychologist analyzing conversational responses.
            Extract personality signals from this response related to {assessment_type} traits.
            For each trait, provide a score from 0.0 to 1.0 indicating the degree to which the response suggests this trait.
            
            The traits to analyze are: {', '.join(traits_to_analyze)}
            
            In your analysis, consider:
            1. Explicit statements about preferences or behaviors
            2. Linguistic patterns and word choice
            3. Content themes and topics discussed
            4. Emotional tone and expressions
            5. Contextual signals from the conversation history
            
            Format your response as a JSON object mapping trait names to scores.
            Example: {{"extraversion": 0.7, "agreeableness": 0.8}}
            """
            
            # Call the LLM to analyze the response
            # In a real implementation, this would use the actual LLM
            responses = await self.personality_agent.llm.agenerate_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context of conversation:\n{context}\n\nUser's most recent response: {response}"}
            ])
            
            response_text = responses.generations[0][0].text
            
            # Try to parse JSON from the response
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    trait_scores = json.loads(json_str)
                except json.JSONDecodeError:
                    # Fallback to default scores if parsing fails
                    trait_scores = {trait: 0.5 for trait in traits_to_analyze}
            else:
                # Fallback to default scores if no JSON is found
                trait_scores = {trait: 0.5 for trait in traits_to_analyze}
            
            # If emotion agent is available, integrate emotional analysis
            if self.emotion_agent:
                try:
                    emotion_result = await self.emotion_agent.analyze_emotion(response)
                    primary_emotion = emotion_result.get("primary_emotion", "neutral")
                    
                    # Update last emotion
                    state["last_emotion"] = primary_emotion
                    
                    # Add emotional data to state
                    if "emotional_data" not in state:
                        state["emotional_data"] = []
                    
                    state["emotional_data"].append({
                        "text": response,
                        "emotion": emotion_result,
                        "timestamp": time.time()
                    })
                    
                    # Adjust trait scores based on emotional analysis
                    if assessment_type == "big_five":
                        # Example: high anxiety might suggest higher neuroticism
                        if primary_emotion in ["anxiety", "fear", "stress"]:
                            trait_scores["neuroticism"] = max(0.6, trait_scores.get("neuroticism", 0.5))
                        
                        # Example: enthusiasm might suggest higher extraversion
                        if primary_emotion in ["joy", "excitement", "enthusiasm"]:
                            trait_scores["extraversion"] = max(0.6, trait_scores.get("extraversion", 0.5))
                except Exception as e:
                    # If emotion analysis fails, continue with the extracted trait scores
                    pass
            
            return trait_scores
            
        except Exception as e:
            # Return default scores if analysis fails
            return {trait: 0.5 for trait in traits_to_analyze}
    
    async def _generate_follow_up_question(self, user_response: str):
        """Generate a follow-up question based on the user's response"""
        import asyncio
        import random
        
        state = st.session_state["dynamic_assessment_state"]
        
        # Get context
        assessment_type = state["assessment_type"]
        traits_to_focus = state["current_traits_focus"]
        conversation_history = state["conversation_history"]
        last_emotion = state.get("last_emotion", "neutral")
        
        # Format conversation history
        conversation_context = "\n".join([
            f"{entry['role']}: {entry['content']}" 
            for entry in conversation_history[-5:]  # Last 5 exchanges
        ])
        
        try:
            # Create a prompt for the LLM to generate a follow-up question
            prompt = f"""
            You are conducting a conversational personality assessment using the {assessment_type} model.
            
            Recent conversation:
            {conversation_context}
            
            The user's emotional state appears to be: {last_emotion}
            
            You are currently focusing on assessing these traits: {', '.join(traits_to_focus)}
            
            Generate a natural, conversational follow-up question that:
            1. Feels like a natural continuation of the conversation
            2. Helps assess one of the focus traits: {', '.join(traits_to_focus)}
            3. Takes into account the user's apparent emotional state
            4. Is open-ended and encourages elaboration
            5. Avoids repetitive questioning
            6. Doesn't sound like a formal assessment item
            
            Your response should ONLY include the follow-up question, nothing else.
            """
            
            # Call the LLM to generate the follow-up question
            responses = await self.personality_agent.llm.agenerate_messages([
                {"role": "system", "content": prompt}
            ])
            
            follow_up_question = responses.generations[0][0].text.strip()
            
            # Fallback to template questions if needed
            if not follow_up_question or len(follow_up_question) < 10:
                # Pick a template question for one of the focus traits
                trait = random.choice(traits_to_focus)
                questions = self.trait_conversation_starters.get(trait, [])
                
                if questions:
                    follow_up_question = random.choice(questions)
                else:
                    # Generic fallback question
                    follow_up_question = "Could you tell me more about how you typically respond to challenges in your life?"
            
            return follow_up_question
            
        except Exception as e:
            # Fallback to a generic follow-up question
            return "That's interesting. Could you tell me more about why you feel that way?"
    
    def _rotate_trait_focus(self):
        """Rotate the traits being assessed to ensure comprehensive coverage"""
        import random
        
        state = st.session_state["dynamic_assessment_state"]
        
        # Define all traits based on assessment type
        all_traits = []
        if state["assessment_type"] == "big_five":
            all_traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
        else:  # MBTI
            all_traits = ["E/I", "S/N", "T/F", "J/P"]
        
        # Get current focus traits
        current_focus = state["current_traits_focus"]
        
        # Find traits not currently in focus
        other_traits = [t for t in all_traits if t not in current_focus]
        
        if other_traits:
            # Add a new trait to focus on
            new_trait = random.choice(other_traits)
            
            # Optionally remove one current trait if we have more than 2
            if len(current_focus) > 2:
                current_focus.pop(0)  # Remove oldest trait
            
            # Add new trait
            current_focus.append(new_trait)
            
            # Update state
            state["current_traits_focus"] = current_focus
    
    async def _finalize_conversational_assessment(self):
        """Finalize the conversational assessment and generate results"""
        import asyncio
        
        state = st.session_state["dynamic_assessment_state"]
        
        assessment_type = state["assessment_type"]
        trait_scores = state.get("trait_scores", {})
        
        # Generate the assessment results based on assessment type
        if assessment_type == "big_five":
            # Average the scores for each trait
            avg_scores = {}
            for trait, scores in trait_scores.items():
                if scores:
                    avg_scores[trait] = sum(scores) / len(scores)
                else:
                    avg_scores[trait] = 0.5  # Default for traits without scores
            
            # Ensure all traits have scores
            for trait in ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]:
                if trait not in avg_scores:
                    avg_scores[trait] = 0.5
            
            # Format for the Big Five assessment
            responses = {}
            for i, trait in enumerate(avg_scores.keys()):
                # Create synthetic responses that would yield these scores
                score = avg_scores[trait]
                # Convert 0-1 scale to 1-5 scale for the Big Five assessment
                synthetic_score = int(1 + score * 4)
                responses[str(i+1)] = synthetic_score
            
            # Compute results using the BigFiveAssessment
            results = self.big_five_assessment.compute_results(responses)
            
            # Add conversational data
            results["conversation_based"] = True
            results["conversation_length"] = len(state["conversation_history"])
            results["trait_scores"] = avg_scores
            
            return results
            
        else:  # MBTI
            # Process scores for each dimension
            dimension_scores = {
                "E": 0, "I": 0,
                "S": 0, "N": 0,
                "T": 0, "F": 0,
                "J": 0, "P": 0
            }
            
            # Map trait scores to dimension scores
            if "E/I" in trait_scores:
                e_i_scores = trait_scores["E/I"]
                # Lower scores favor I, higher scores favor E
                avg_e_i = sum(e_i_scores) / len(e_i_scores) if e_i_scores else 0.5
                dimension_scores["E"] = avg_e_i
                dimension_scores["I"] = 1 - avg_e_i
            
            if "S/N" in trait_scores:
                s_n_scores = trait_scores["S/N"]
                # Lower scores favor S, higher scores favor N
                avg_s_n = sum(s_n_scores) / len(s_n_scores) if s_n_scores else 0.5
                dimension_scores["S"] = 1 - avg_s_n
                dimension_scores["N"] = avg_s_n
            
            if "T/F" in trait_scores:
                t_f_scores = trait_scores["T/F"]
                # Lower scores favor T, higher scores favor F
                avg_t_f = sum(t_f_scores) / len(t_f_scores) if t_f_scores else 0.5
                dimension_scores["T"] = 1 - avg_t_f
                dimension_scores["F"] = avg_t_f
            
            if "J/P" in trait_scores:
                j_p_scores = trait_scores["J/P"]
                # Lower scores favor J, higher scores favor P
                avg_j_p = sum(j_p_scores) / len(j_p_scores) if j_p_scores else 0.5
                dimension_scores["J"] = 1 - avg_j_p
                dimension_scores["P"] = avg_j_p
            
            # Create synthetic responses
            responses = {}
            
            # MBTI type determination
            mbti_type = ""
            mbti_type += "E" if dimension_scores["E"] > dimension_scores["I"] else "I"
            mbti_type += "S" if dimension_scores["S"] > dimension_scores["N"] else "N"
            mbti_type += "T" if dimension_scores["T"] > dimension_scores["F"] else "F"
            mbti_type += "J" if dimension_scores["J"] > dimension_scores["P"] else "P"
            
            # Get type description from MBTIAssessment
            type_descriptions = self.mbti_assessment.type_descriptions
            type_info = type_descriptions.get(mbti_type, {})
            
            # Create a results object
            results = {
                "type": mbti_type,
                "type_name": type_info.get("name", f"Type {mbti_type}"),
                "description": type_info.get("description", ""),
                "strengths": type_info.get("strengths", []),
                "weaknesses": type_info.get("weaknesses", []),
                "conversation_based": True,
                "conversation_length": len(state["conversation_history"]),
                "dimension_scores": dimension_scores,
                "dimensions": {
                    "E/I": {
                        "dominant": "E" if dimension_scores["E"] > dimension_scores["I"] else "I",
                        "scores": {
                            "E": dimension_scores["E"],
                            "I": dimension_scores["I"]
                        },
                        "percentages": {
                            "E": dimension_scores["E"] * 100,
                            "I": dimension_scores["I"] * 100
                        }
                    },
                    "S/N": {
                        "dominant": "S" if dimension_scores["S"] > dimension_scores["N"] else "N",
                        "scores": {
                            "S": dimension_scores["S"],
                            "N": dimension_scores["N"]
                        },
                        "percentages": {
                            "S": dimension_scores["S"] * 100,
                            "N": dimension_scores["N"] * 100
                        }
                    },
                    "T/F": {
                        "dominant": "T" if dimension_scores["T"] > dimension_scores["F"] else "F",
                        "scores": {
                            "T": dimension_scores["T"],
                            "F": dimension_scores["F"]
                        },
                        "percentages": {
                            "T": dimension_scores["T"] * 100,
                            "F": dimension_scores["F"] * 100
                        }
                    },
                    "J/P": {
                        "dominant": "J" if dimension_scores["J"] > dimension_scores["P"] else "P",
                        "scores": {
                            "J": dimension_scores["J"],
                            "P": dimension_scores["P"]
                        },
                        "percentages": {
                            "J": dimension_scores["J"] * 100,
                            "P": dimension_scores["P"] * 100
                        }
                    }
                }
            }
            
            return results
    
    async def _compute_results(self):
        """Compute results for the traditional assessment"""
        state = st.session_state["dynamic_assessment_state"]
        
        assessment_type = state["assessment_type"]
        responses = state["responses"]
        
        # Use the appropriate assessment to compute results
        if assessment_type == "big_five":
            results = self.big_five_assessment.compute_results(responses)
        else:  # MBTI
            results = self.mbti_assessment.compute_results(responses)
        
        # Add emotional data if available
        if "emotional_data" in state and state["emotional_data"]:
            results["emotional_data"] = state["emotional_data"]
        
        # Add interpretation using the personality agent
        try:
            emotional_context = {}
            if "emotional_data" in state and state["emotional_data"]:
                # Get the most recent emotional data
                latest_emotion = state["emotional_data"][-1]["emotion"]
                emotional_context = {
                    "primary_emotion": latest_emotion.get("primary_emotion", "neutral"),
                    "intensity": latest_emotion.get("intensity", 5),
                    "secondary_emotions": latest_emotion.get("secondary_emotions", [])
                }
            
            # Get interpretation
            interpretation_result = await self.personality_agent.conduct_assessment(
                assessment_type=assessment_type,
                responses=responses,
                emotional_context=emotional_context
            )
            
            # Add interpretation to results
            results["interpretation"] = interpretation_result.get("interpretation", {})
            results["emotion_indicators"] = interpretation_result.get("emotion_indicators", {})
            
        except Exception as e:
            # If interpretation fails, continue without it
            results["interpretation_error"] = str(e)
        
        return results
    
    def _adapt_upcoming_questions(self):
        """Adapt upcoming questions based on current responses and emotional state"""
        state = st.session_state["dynamic_assessment_state"]
        
        # Skip adaptation for now, but log that it happened
        # In a real implementation, this would use the personality_agent's question_adaptation_prompt
        # to generate personalized versions of upcoming questions
        
        # Example adaptation (not implemented here):
        # if state["current_question_index"] < len(state["current_questions"]) - 1:
        #     next_question_index = state["current_question_index"] + 1
        #     original_question = state["current_questions"][next_question_index]["text"]
        #     
        #     # Adapt question based on user context
        #     adapted_question = "Personalized version of " + original_question
        #     
        #     # Store the adaptation
        #     state["adaptations"][next_question_index] = adapted_question