"""
Component for rendering diagnosis results in a user-friendly format.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class DiagnosisResultsComponent:
    """Component for rendering diagnosis results"""
    
    def __init__(self, on_continue: Optional[callable] = None):
        """
        Initialize the diagnosis results component
        
        Args:
            on_continue: Callback function to call when user continues
        """
        self.on_continue = on_continue
    
    def render(self, assessment_results: Dict[str, Any], empathy_response: str, immediate_actions: List[str]):
        """
        Render the diagnosis results
        
        Args:
            assessment_results: Dictionary containing assessment results
            empathy_response: String containing empathetic response
            immediate_actions: List of immediate actions
        """
        st.header("Your Personalized Support Plan")
        
        # Display empathetic response
        st.subheader("Understanding Your Experience")
        st.markdown(f"""
        <div style='background-color:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px;'>
        {empathy_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Display mental health results
        mental_health = assessment_results.get("mental_health", {})
        if mental_health:
            st.subheader("Mental Health Assessment")
            
            # Display overall status
            overall_status = mental_health.get("overall_status", "mild")
            status_color = {
                "severe": "🔴",
                "moderate": "🟠",
                "mild": "🟢"
            }.get(overall_status, "⚪")
            
            st.markdown(f"**Overall Status:** {status_color} {overall_status.capitalize()}")
            
            # Display category scores
            scores = mental_health.get("scores", {})
            severity_levels = mental_health.get("severity_levels", {})
            
            if scores:
                # Create columns for visualization
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Create a bar chart for the scores
                    categories = []
                    values = []
                    colors = []
                    
                    for category, score in scores.items():
                        if category != "suicidal":  # Don't display suicidal score in chart
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
                    bars = ax.barh(categories, values, color=colors)
                    ax.set_xlabel("Score")
                    ax.set_title("Mental Health Assessment Scores")
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.1f}", 
                                ha='left', va='center')
                    
                    st.pyplot(fig)
                
                with col2:
                    # Display severity levels as text
                    st.markdown("### Severity Levels")
                    for category, severity in severity_levels.items():
                        if category != "suicidal":  # Don't display suicidal severity
                            emoji = {
                                "severe": "🔴",
                                "moderate": "🟠",
                                "mild": "🟢"
                            }.get(severity, "⚪")
                            st.markdown(f"**{category.capitalize()}:** {emoji} {severity.capitalize()}")
        
        # Display personality results
        personality = assessment_results.get("personality", {})
        if personality:
            st.subheader("Personality Profile")
            
            traits = personality.get("traits", {})
            if traits:
                # Create a radar chart for personality traits
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
                
                # Add title
                plt.title("Personality Traits", size=15, y=1.1)
                
                st.pyplot(fig)
                
                # Display trait descriptions
                st.markdown("### Trait Descriptions")
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
                    st.markdown(f"**{trait.capitalize()}:** {emoji} {description}")
        
        # Display immediate actions
        st.subheader("Recommended Next Steps")
        st.markdown("""
        <div style='background-color:#e9ecef; padding:20px; border-radius:10px;'>
        """, unsafe_allow_html=True)
        
        for i, action in enumerate(immediate_actions):
            st.markdown(f"{i+1}. {action}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Continue button
        if st.button("Continue to Chat"):
            if self.on_continue:
                self.on_continue()
            else:
                # Default behavior if no callback provided
                if "step" in st.session_state:
                    st.session_state["step"] = 4  # Move to chat interface
                    st.rerun()
