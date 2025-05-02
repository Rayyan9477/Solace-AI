"""
Base UI component for the mental health chatbot application.
All UI components should inherit from this class.
"""

import streamlit as st
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseUIComponent(ABC):
    """Base class for all UI components in the application"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the UI component
        
        Args:
            name: Name of the component
            description: Description of what the component does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def render(self, **kwargs):
        """
        Render the UI component
        
        Args:
            **kwargs: Additional parameters for rendering
        """
        pass
    
    def show_loading(self, message: str = "Loading..."):
        """Show a loading spinner with message"""
        return st.spinner(message)
    
    def show_success(self, message: str):
        """Show a success message"""
        return st.success(message)
    
    def show_error(self, message: str):
        """Show an error message"""
        return st.error(message)
    
    def show_warning(self, message: str):
        """Show a warning message"""
        return st.warning(message)
    
    def show_info(self, message: str):
        """Show an info message"""
        return st.info(message)
    
    def create_styled_container(self, border_radius: int = 10, 
                               bg_color: str = "#f8f9fa", 
                               padding: int = 20,
                               margin_bottom: int = 20):
        """
        Create a styled container with custom CSS
        
        Returns:
            A Streamlit container with custom CSS
        """
        container = st.container()
        
        # Apply custom CSS
        container_style = f"""
        <style>
        [data-testid="stContainer"] {{
            background-color: {bg_color};
            padding: {padding}px;
            border-radius: {border_radius}px;
            margin-bottom: {margin_bottom}px;
        }}
        </style>
        """
        
        container.markdown(container_style, unsafe_allow_html=True)
        return container

# Add alias for backward compatibility
BaseComponent = BaseUIComponent