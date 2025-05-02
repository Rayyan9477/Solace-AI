"""
Landing page component for the mental health chatbot.
Provides the initial entry point with options for voice or text-based interaction.
"""

import streamlit as st
from typing import Dict, Any, Callable, Optional
from .base_component import BaseComponent

class LandingPageComponent(BaseComponent):
    """Component for rendering the landing page"""
    
    def __init__(self, 
                 on_start_voice: Optional[Callable] = None,
                 on_start_text: Optional[Callable] = None,
                 voice_enabled: bool = True):
        """
        Initialize the landing page component
        
        Args:
            on_start_voice: Callback for when user chooses voice interaction
            on_start_text: Callback for when user chooses text interaction
            voice_enabled: Whether voice interaction is enabled
        """
        super().__init__(key="landing_page")
        self.on_start_voice = on_start_voice
        self.on_start_text = on_start_text
        self.voice_enabled = voice_enabled
        
        # Apply custom CSS
        self.apply_landing_css()
    
    def render(self):
        """Render the landing page"""
        # Hide default Streamlit elements for a cleaner look
        self.hide_streamlit_elements()
        
        # Header with gradient background
        st.markdown("""
        <div class="header-container">
            <div class="header-content">
                <h1>Mental Health Support Assistant</h1>
                <p class="subtitle">A safe space for emotional well-being and personal growth</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main content
        st.markdown("""
        <div class="intro-text">
            <p>Welcome to your personal mental health companion, designed to provide empathetic support 
            and valuable insights about your well-being.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards section
        st.markdown("<h2 class='section-title'>How I Can Help You</h2>", unsafe_allow_html=True)
        
        # First row of cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="icon">💬</div>
                <h3>Supportive Conversations</h3>
                <p>Have compassionate, judgment-free conversations tailored to your emotional needs.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="icon">🧠</div>
                <h3>Personality Insights</h3>
                <p>Receive personalized insights into your personality traits and tendencies.</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Second row of cards
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="icon">🌱</div>
                <h3>Coping Strategies</h3>
                <p>Learn evidence-based techniques to manage stress, anxiety, and other challenges.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="feature-card">
                <div class="icon">📊</div>
                <h3>Mental Health Assessment</h3>
                <p>Complete a comprehensive assessment to better understand your current well-being.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Choice section
        st.markdown("<h2 class='section-title'>Choose Your Experience</h2>", unsafe_allow_html=True)
        
        # Voice vs Text choice
        col_voice, col_text = st.columns(2)
        
        with col_voice:
            st.markdown("""
            <div class="choice-card voice-card">
                <div class="choice-icon">🎤</div>
                <h3>Voice Interaction</h3>
                <p>Speak naturally with your assistant and receive voice responses for a more personal experience.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if self.voice_enabled:
                if st.button("Start Voice Mode", key="btn_voice_mode", use_container_width=True):
                    if self.on_start_voice:
                        self.on_start_voice()
            else:
                st.warning("Voice mode is currently unavailable. Please install the required dependencies.")
            
        with col_text:
            st.markdown("""
            <div class="choice-card text-card">
                <div class="choice-icon">⌨️</div>
                <h3>Text Interaction</h3>
                <p>Type your messages and read responses for a more private and controlled experience.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start Text Mode", key="btn_text_mode", use_container_width=True):
                if self.on_start_text:
                    self.on_start_text()
        
        # Testimonials section
        st.markdown("<h2 class='section-title'>What Others Are Saying</h2>", unsafe_allow_html=True)
        
        # Testimonial cards
        testimonials_col1, testimonials_col2, testimonials_col3 = st.columns(3)
        
        with testimonials_col1:
            st.markdown("""
            <div class="testimonial-card">
                <div class="quote">"The personality insights were surprisingly accurate. It's helped me understand my reactions better."</div>
                <div class="author">- Alex M.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with testimonials_col2:
            st.markdown("""
            <div class="testimonial-card">
                <div class="quote">"I appreciate having a space to express my feelings without judgment. The coping strategies have been really helpful."</div>
                <div class="author">- Jamie L.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with testimonials_col3:
            st.markdown("""
            <div class="testimonial-card">
                <div class="quote">"The assessment gave me insights I hadn't considered before. It's like having a therapist in my pocket."</div>
                <div class="author">- Sam K.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>Your privacy and security are our top priorities. All conversations are confidential.</p>
            <p class="disclaimer">This is not a replacement for professional mental health care.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def hide_streamlit_elements(self):
        """Hide default Streamlit elements for a cleaner look"""
        hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
        st.markdown(hide_st_style, unsafe_allow_html=True)
    
    def apply_landing_css(self):
        """Apply custom CSS for the landing page"""
        landing_css = """
        /* Global styles */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        
        .stApp {
            font-family: 'DM Sans', sans-serif;
        }
        
        /* Header styles */
        .header-container {
            background: linear-gradient(135deg, #9896f0 0%, #FBC8D4 100%);
            border-radius: 15px;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(152, 150, 240, 0.2);
        }
        
        .header-content h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            animation: fadeIn 1s ease;
        }
        
        .subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            animation: fadeIn 1.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Intro text */
        .intro-text {
            font-size: 1.2rem;
            line-height: 1.6;
            margin: 2rem 0;
            color: #444;
            text-align: center;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Section titles */
        .section-title {
            font-size: 2rem;
            font-weight: 600;
            margin: 3rem 0 1.5rem;
            color: #333;
            text-align: center;
            position: relative;
        }
        
        .section-title:after {
            content: "";
            display: block;
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #9896f0, #FBC8D4);
            margin: 0.8rem auto 0;
            border-radius: 2px;
        }
        
        /* Feature cards */
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        
        .feature-card .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h3 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #333;
        }
        
        .feature-card p {
            color: #666;
            line-height: 1.6;
        }
        
        /* Choice cards */
        .choice-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .choice-card:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(90deg, #9896f0, #FBC8D4);
        }
        
        .voice-card:before {
            background: linear-gradient(90deg, #9896f0, #74ebd5);
        }
        
        .text-card:before {
            background: linear-gradient(90deg, #FBC8D4, #ff9a9e);
        }
        
        .choice-card .choice-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
        }
        
        .choice-card h3 {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #333;
        }
        
        .choice-card p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(90deg, #9896f0, #FBC8D4);
            color: white;
            border: none;
            font-weight: 600;
            border-radius: 30px;
            padding: 0.8rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(152, 150, 240, 0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(152, 150, 240, 0.4);
        }
        
        /* Testimonial cards */
        .testimonial-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.03);
            height: 100%;
            position: relative;
        }
        
        .testimonial-card:before {
            content: "";
            position: absolute;
            top: 10px;
            left: 15px;
            font-size: 3rem;
            color: rgba(152, 150, 240, 0.2);
            font-family: serif;
        }
        
        .testimonial-card .quote {
            font-style: italic;
            color: #555;
            line-height: 1.6;
            padding-top: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .testimonial-card .author {
            font-weight: 600;
            color: #333;
            text-align: right;
        }
        
        /* Footer */
        .footer {
            margin-top: 3rem;
            padding-top: 2rem;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }
        
        .disclaimer {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #999;
        }
        """
        
        self.apply_custom_css(landing_css)