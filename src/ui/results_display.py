"""
Results display component for the mental health chatbot.
Visualizes personality and mental health assessment results.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
from .base_component import BaseComponent

logger = logging.getLogger(__name__)

class ResultsDisplayComponent(BaseComponent):
    """Component for rendering assessment results with a modern UI"""
    
    def __init__(self, 
                 on_continue: Optional[Callable] = None,
                 voice_enabled: bool = False):
        """
        Initialize the results display component
        
        Args:
            on_continue: Callback function for when user continues
            voice_enabled: Whether voice interaction is enabled
        """
        super().__init__(key="results_display")
        self.on_continue = on_continue
        self.voice_enabled = voice_enabled
        
        # Apply custom CSS
        self.apply_results_css()
    
    def render(self, assessment_results: Dict[str, Any], 
               empathy_response: str, 
               immediate_actions: List[str]):
        """
        Render the assessment results
        
        Args:
            assessment_results: Dictionary containing assessment results
            empathy_response: Empathetic response based on assessment
            immediate_actions: List of recommended actions
        """
        # Header with animation
        st.markdown("""
        <div class="results-header">
            <h1>Your Personalized Profile</h1>
            <p class="results-subtitle">Based on your personality and mental health assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different sections of the results
        tab1, tab2, tab3 = st.tabs(["Overview", "Personality Profile", "Mental Health Insights"])
        
        with tab1:
            self._render_overview(assessment_results, empathy_response, immediate_actions)
        
        with tab2:
            self._render_personality_profile(assessment_results.get("personality", {}))
        
        with tab3:
            self._render_mental_health_insights(assessment_results.get("mental_health", {}))
        
        # Bottom actions
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Continue to Chat", key="btn_continue_chat", use_container_width=True):
                if self.on_continue:
                    self.on_continue()
        
        with col2:
            if st.button("Save Results as PDF", key="btn_save_pdf", use_container_width=True):
                st.info("PDF generation is not implemented in this demo version.")
    
    def _render_overview(self, assessment_results: Dict[str, Any], 
                         empathy_response: str, 
                         immediate_actions: List[str]):
        """Render the overview tab"""
        # Display empathetic response
        st.markdown(f"""
        <div class="empathy-card">
            <h2>Understanding Your Experience</h2>
            <p class="empathy-text">{empathy_response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display key insights
        personality = assessment_results.get("personality", {})
        mental_health = assessment_results.get("mental_health", {})
        
        # Create columns for personality snapshot and mental health snapshot
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="snapshot-card">
                <h2>Personality Snapshot</h2>
            """, unsafe_allow_html=True)
            
            traits = personality.get("traits", {})
            if traits:
                # Display top traits
                st.markdown("<div class='trait-list'>", unsafe_allow_html=True)
                
                for trait_name, trait_data in traits.items():
                    category = trait_data.get("category", "average")
                    emoji = {
                        "high": "⬆️",
                        "average": "➡️",
                        "low": "⬇️"
                    }.get(category, "➡️")
                    
                    st.markdown(f"""
                    <div class="trait-item">
                        <span class="trait-name">{trait_name.capitalize()}</span>
                        <span class="trait-emoji">{emoji}</span>
                        <span class="trait-category">{category.capitalize()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No personality data available.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="snapshot-card">
                <h2>Mental Health Snapshot</h2>
            """, unsafe_allow_html=True)
            
            # Display overall status
            overall_status = mental_health.get("overall_status", "mild")
            status_color = {
                "severe": "🔴",
                "moderate": "🟠",
                "mild": "🟢"
            }.get(overall_status, "⚪")
            
            st.markdown(f"""
            <div class="status-indicator">
                <span class="status-label">Overall Status:</span>
                <span class="status-value">{status_color} {overall_status.capitalize()}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display top concerns
            severity_levels = mental_health.get("severity_levels", {})
            if severity_levels:
                st.markdown("<div class='concern-list'>", unsafe_allow_html=True)
                
                for category, severity in severity_levels.items():
                    if category != "suicidal":  # Don't display suicidal severity
                        emoji = {
                            "severe": "🔴",
                            "moderate": "🟠",
                            "mild": "🟢"
                        }.get(severity, "⚪")
                        
                        st.markdown(f"""
                        <div class="concern-item">
                            <span class="concern-name">{category.capitalize()}</span>
                            <span class="concern-severity">{emoji} {severity.capitalize()}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No mental health data available.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display immediate actions
        st.markdown("""
        <div class="actions-card">
            <h2>Recommended Next Steps</h2>
            <div class="actions-list">
        """, unsafe_allow_html=True)
        
        for i, action in enumerate(immediate_actions):
            st.markdown(f"""
            <div class="action-item">
                <span class="action-number">{i+1}</span>
                <span class="action-text">{action}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_personality_profile(self, personality: Dict[str, Any]):
        """Render the personality profile tab"""
        if not personality:
            st.info("No personality data available. Please complete the assessment to see your personality profile.")
            return
        
        # Display personality traits as radar chart
        traits = personality.get("traits", {})
        if traits:
            st.markdown("<h2 class='section-title'>Your Personality Traits</h2>", unsafe_allow_html=True)
            
            # Create radar chart
            categories = []
            values = []
            
            for trait, data in traits.items():
                categories.append(trait.capitalize())
                values.append(data.get("score", 50))  # Default to 50 if no score
            
            # Create the radar chart
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add the first point at the end to close the polygon
            values_plot = values.copy()
            values_plot += values[:1]
            
            # Draw the polygon and fill it
            ax.plot(angles, values_plot, linewidth=1, linestyle='solid')
            ax.fill(angles, values_plot, alpha=0.1)
            
            # Set the labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            
            # Add a style to make it look more modern
            ax.spines['polar'].set_visible(False)
            ax.grid(True, color='#eeeeee')
            plt.tight_layout()
            
            # Display the chart
            st.pyplot(fig)
            
            # Display trait descriptions
            st.markdown("<h2 class='section-title'>What Your Traits Mean</h2>", unsafe_allow_html=True)
            
            for trait, data in traits.items():
                category = data.get("category", "average")
                emoji = {
                    "high": "⬆️",
                    "average": "➡️",
                    "low": "⬇️"
                }.get(category, "➡️")
                
                descriptions = {
                    "extraversion": {
                        "high": "You tend to be outgoing, energetic, and draw energy from social interactions.",
                        "average": "You balance social interaction with alone time, adapting to different situations.",
                        "low": "You tend to be more reserved and find energy in solitary activities."
                    },
                    "agreeableness": {
                        "high": "You tend to be compassionate, cooperative, and value harmony.",
                        "average": "You balance cooperation with standing up for your own needs.",
                        "low": "You tend to be more analytical, skeptical, and competitive."
                    },
                    "conscientiousness": {
                        "high": "You tend to be organized, responsible, and detail-oriented.",
                        "average": "You balance structure with flexibility in your approach to tasks.",
                        "low": "You tend to be more spontaneous, flexible, and less bound by schedules."
                    },
                    "neuroticism": {
                        "high": "You tend to experience emotions intensely and may be more sensitive to stress.",
                        "average": "You have a balanced emotional response to situations.",
                        "low": "You tend to be emotionally stable and less easily affected by stress."
                    },
                    "openness": {
                        "high": "You tend to be curious, creative, and open to new experiences.",
                        "average": "You balance curiosity with appreciation for the familiar.",
                        "low": "You tend to be practical, conventional, and prefer familiar routines."
                    }
                }
                
                description = descriptions.get(trait, {}).get(category, "")
                
                st.markdown(f"""
                <div class="trait-description-card">
                    <div class="trait-header">
                        <span class="trait-name-large">{trait.capitalize()}</span>
                        <span class="trait-emoji-large">{emoji}</span>
                        <span class="trait-category-large">{category.capitalize()}</span>
                    </div>
                    <p class="trait-description">{description}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No personality trait data available.")
    
    def _render_mental_health_insights(self, mental_health: Dict[str, Any]):
        """Render the mental health insights tab"""
        if not mental_health:
            st.info("No mental health data available. Please complete the assessment to see your mental health insights.")
            return
        
        # Display overall status
        overall_status = mental_health.get("overall_status", "mild")
        status_color = {
            "severe": "🔴",
            "moderate": "🟠",
            "mild": "🟢"
        }.get(overall_status, "⚪")
        
        st.markdown(f"""
        <div class="status-card">
            <h2>Overall Status</h2>
            <div class="status-indicator-large">
                <span class="status-emoji">{status_color}</span>
                <span class="status-text">{overall_status.capitalize()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display category scores as bar chart
        scores = mental_health.get("scores", {})
        severity_levels = mental_health.get("severity_levels", {})
        
        if scores:
            st.markdown("<h2 class='section-title'>Mental Health Dimensions</h2>", unsafe_allow_html=True)
            
            # Create a horizontal bar chart
            categories = []
            values = []
            colors = []
            
            for category, score in scores.items():
                if category != "suicidal":  # Don't display suicidal score
                    categories.append(category.capitalize())
                    values.append(score)
                    
                    # Set color based on severity
                    severity = severity_levels.get(category, "mild")
                    if severity == "severe":
                        colors.append("#ff6b6b")  # Red
                    elif severity == "moderate":
                        colors.append("#ffa94d")  # Orange
                    else:
                        colors.append("#69db7c")  # Green
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = range(len(categories))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("Score")
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Display the chart
            st.pyplot(fig)
            
            # Display category descriptions
            st.markdown("<h2 class='section-title'>Understanding Your Mental Health</h2>", unsafe_allow_html=True)
            
            # Define descriptions for each category and severity level
            descriptions = {
                "depression": {
                    "severe": "You're experiencing significant symptoms of depression that are likely impacting your daily life.",
                    "moderate": "You're showing some signs of depression that may be affecting parts of your life.",
                    "mild": "You have some mild symptoms of depression."
                },
                "anxiety": {
                    "severe": "You're experiencing high levels of anxiety that are likely interfering with daily activities.",
                    "moderate": "You're showing moderate anxiety symptoms that may be challenging at times.",
                    "mild": "You have some mild symptoms of anxiety."
                },
                "stress": {
                    "severe": "Your stress levels are very high and may be overwhelming.",
                    "moderate": "You're experiencing moderate stress that may be difficult to manage at times.",
                    "mild": "You're experiencing low levels of stress."
                },
                "sleep": {
                    "severe": "You're having significant sleep difficulties that are likely affecting your wellbeing.",
                    "moderate": "You're experiencing some sleep issues that may be affecting your rest.",
                    "mild": "You're having minimal sleep difficulties."
                },
                "social": {
                    "severe": "You're experiencing significant feelings of social isolation or loneliness.",
                    "moderate": "You're feeling somewhat disconnected from others.",
                    "mild": "You're generally feeling socially connected."
                },
                "cognitive": {
                    "severe": "You're having significant difficulty with concentration and focus.",
                    "moderate": "You're experiencing some challenges with concentration.",
                    "mild": "You're having minimal cognitive difficulties."
                },
                "physical": {
                    "severe": "You're experiencing many physical symptoms that may be related to mental health.",
                    "moderate": "You're having some physical symptoms that may be stress-related.",
                    "mild": "You're experiencing few physical symptoms."
                }
            }
            
            for category, severity in severity_levels.items():
                if category != "suicidal" and category in descriptions:
                    # Get emoji based on severity
                    emoji = {
                        "severe": "🔴",
                        "moderate": "🟠",
                        "mild": "🟢"
                    }.get(severity, "⚪")
                    
                    # Get description
                    description = descriptions.get(category, {}).get(severity, "No description available.")
                    
                    # Display card
                    st.markdown(f"""
                    <div class="health-category-card">
                        <div class="category-header">
                            <span class="category-name">{category.capitalize()}</span>
                            <span class="category-severity">{emoji} {severity.capitalize()}</span>
                        </div>
                        <p class="category-description">{description}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No mental health score data available.")
    
    async def generate_empathetic_response(self, assessment_results: Dict[str, Any]) -> str:
        """
        Generate an empathetic response based on assessment results using Gemini 2.0
        
        Args:
            assessment_results: Dictionary containing assessment results
            
        Returns:
            Empathetic response text
        """
        try:
            # Import Gemini LLM
            from src.models.gemini_llm import GeminiLLM
            from src.config.settings import AppConfig
            import os
            
            # Get API key
            api_key = AppConfig.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("No Gemini API key found, using default response")
                return self._get_default_empathetic_response(assessment_results)
            
            # Create Gemini LLM instance
            gemini_llm = GeminiLLM(api_key=api_key)
            
            # Extract personality data
            personality = assessment_results.get("personality", {})
            traits = personality.get("traits", {})
            
            # Extract mental health data
            mental_health = assessment_results.get("mental_health", {})
            overall_status = mental_health.get("overall_status", "mild")
            severity_levels = mental_health.get("severity_levels", {})
            
            # Create prompt for Gemini
            prompt = f"""Based on a mental health and personality assessment, I need to generate an empathetic, supportive response for the user. 
Here are their results:

## Personality Profile:
"""

            # Add personality traits to prompt
            if traits:
                for trait_name, trait_data in traits.items():
                    score = trait_data.get("score", 50)
                    category = trait_data.get("category", "average")
                    prompt += f"- {trait_name.capitalize()}: {category} ({score}/100)\n"
            else:
                prompt += "- No detailed personality data available\n"
                
            prompt += "\n## Mental Health Assessment:\n"
            prompt += f"- Overall status: {overall_status}\n"
            
            # Add specific mental health dimensions
            if severity_levels:
                for category, severity in severity_levels.items():
                    if category != "suicidal":  # Exclude sensitive categories
                        prompt += f"- {category.capitalize()}: {severity}\n"
            else:
                prompt += "- No detailed mental health data available\n"
            
            prompt += """
Please generate a compassionate, empathetic response that:
1. Validates their experiences and emotions
2. Highlights their personal strengths based on their personality profile
3. Acknowledges any challenges they might be facing
4. Offers genuine hope and encouragement
5. Feels warm and personalized, not clinical or generic
6. Uses a conversational, supportive tone
7. Is culturally sensitive and inclusive
8. Avoids diagnostic language or making promises

The response should be 2-3 paragraphs and feel like it's coming from a supportive mental health professional.
"""
            
            # Generate response with Gemini
            result = await gemini_llm.generate_text(prompt, {})
            
            # Extract response text
            if "text" in result and result["text"]:
                return result["text"]
            else:
                logger.warning("Empty response from Gemini, using default")
                return self._get_default_empathetic_response(assessment_results)
                
        except Exception as e:
            logger.error(f"Error generating empathetic response: {str(e)}")
            return self._get_default_empathetic_response(assessment_results)
    
    def _get_default_empathetic_response(self, assessment_results: Dict[str, Any]) -> str:
        """Generate a default empathetic response if Gemini fails"""
        # Extract basic severity information
        mental_health = assessment_results.get("mental_health", {})
        overall_status = mental_health.get("overall_status", "mild")
        
        # Return appropriate default response based on severity
        if overall_status == "severe":
            return (
                "I can see from your responses that you're facing some significant challenges right now. "
                "It's important to recognize the strength it takes to acknowledge these difficulties. "
                "Your assessment suggests you're experiencing considerable distress in several areas, but please "
                "remember that seeking help is a sign of courage, not weakness. "
                "While these feelings can be overwhelming, there are effective ways to address them with proper support."
            )
        elif overall_status == "moderate":
            return (
                "Based on your responses, I can see that you're experiencing some moderate challenges in several areas. "
                "Your personality profile shows both strengths and opportunities for growth. It's important to "
                "acknowledge that what you're feeling is valid, and many people go through similar experiences. "
                "With the right support and strategies, these difficulties can become more manageable over time."
            )
        else:  # mild or default
            return (
                "Based on your responses, I can see that you're experiencing some mild challenges while showing "
                "resilience in many areas. Your personality profile indicates several strengths that can help you "
                "navigate life's difficulties. While everyone faces struggles at times, your assessment suggests "
                "you have a good foundation to build upon. Remember that taking care of your mental wellbeing "
                "is always valuable, even when things are generally going well."
            )
    
    def apply_results_css(self):
        """Apply custom CSS for the results display"""
        results_css = """
        /* Global styles */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        
        .stApp {
            font-family: 'DM Sans', sans-serif;
        }
        
        /* Results header */
        .results-header {
            text-align: center;
            padding: 2rem 1rem;
            background: linear-gradient(135deg, #9896f0 0%, #FBC8D4 100%);
            border-radius: 15px;
            color: #fff;
            margin-bottom: 2rem;
            animation: fadeIn 1s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .results-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .results-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        /* Section titles */
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem;
            color: #333;
            position: relative;
        }
        
        .section-title:after {
            content: "";
            display: block;
            width: 50px;
            height: 3px;
            background: linear-gradient(90deg, #9896f0, #FBC8D4);
            margin-top: 0.5rem;
            border-radius: 2px;
        }
        
        /* Cards */
        .empathy-card, .snapshot-card, .actions-card, 
        .trait-description-card, .health-category-card, .status-card {
            background: #fff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.8s ease;
        }
        
        .empathy-card h2, .snapshot-card h2, .actions-card h2,
        .status-card h2 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #333;
        }
        
        .empathy-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #555;
        }
        
        /* Trait list */
        .trait-list, .concern-list {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }
        
        .trait-item, .concern-item {
            display: flex;
            align-items: center;
            padding: 0.5rem 0;
        }
        
        .trait-name, .concern-name {
            flex: 1;
            font-weight: 500;
            color: #333;
        }
        
        .trait-emoji, .concern-severity, .trait-category {
            margin-left: 0.5rem;
        }
        
        .trait-category {
            background: #f8f9fa;
            padding: 0.2rem 0.5rem;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #666;
        }
        
        /* Status indicator */
        .status-indicator {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .status-label {
            font-weight: 500;
            color: #333;
            margin-right: 0.5rem;
        }
        
        .status-indicator-large {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            padding: 1.5rem;
        }
        
        .status-emoji {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .status-text {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
        }
        
        /* Actions list */
        .actions-list {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }
        
        .action-item {
            display: flex;
            align-items: center;
            padding: 0.8rem 1rem;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.2s ease;
        }
        
        .action-item:hover {
            background: #f1f3f5;
            transform: translateY(-2px);
        }
        
        .action-number {
            background: linear-gradient(135deg, #9896f0 0%, #FBC8D4 100%);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.8rem;
            margin-right: 1rem;
        }
        
        .action-text {
            flex: 1;
            color: #444;
        }
        
        /* Trait cards */
        .trait-description-card, .health-category-card {
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        
        .trait-description-card:hover, .health-category-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        
        .trait-header, .category-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        
        .trait-name-large, .category-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            flex: 1;
        }
        
        .trait-emoji-large, .category-severity {
            margin: 0 0.5rem;
            font-size: 1.2rem;
        }
        
        .trait-category-large {
            background: #f0f0f0;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            color: #555;
        }
        
        .trait-description, .category-description {
            color: #555;
            line-height: 1.6;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 10px 16px;
            background-color: #f8f9fa;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            border-bottom: 2px solid #9896f0 !important;
        }
        
        /* Section divider */
        .section-divider {
            height: 1px;
            background: #eee;
            margin: 2rem 0;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(90deg, #9896f0, #FBC8D4);
            color: white;
            border: none;
            font-weight: 600;
            border-radius: 30px;
            padding: 0.6rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(152, 150, 240, 0.4);
        }
        """
        
        self.apply_custom_css(results_css)