"""
Base UI component for the mental health chatbot application.
All UI components should inherit from this class.
Note: This is now a non-UI base class for future UI service integration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def render(self, **kwargs):
        """
        Render the UI component
        
        Args:
            **kwargs: Additional parameters for rendering
        
        Returns:
            Dict containing component data for external UI service
        """
        pass
    
    def show_loading(self, message: str = "Loading..."):
        """Log a loading message"""
        self.logger.info(f"Loading: {message}")
        return {"type": "loading", "message": message}
    
    def show_success(self, message: str):
        """Log a success message"""
        self.logger.info(f"Success: {message}")
        return {"type": "success", "message": message}
    
    def show_error(self, message: str):
        """Log an error message"""
        self.logger.error(f"Error: {message}")
        return {"type": "error", "message": message}
    
    def show_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(f"Warning: {message}")
        return {"type": "warning", "message": message}
    
    def show_info(self, message: str):
        """Log an info message"""
        self.logger.info(f"Info: {message}")
        return {"type": "info", "message": message}
    
    def create_styled_container(self, border_radius: int = 10, 
                               bg_color: str = "#f8f9fa", 
                               padding: int = 20,
                               margin_bottom: int = 20):
        """
        Create a styled container configuration for external UI service
        
        Returns:
            Dict containing container styling configuration
        """
        return {
            "type": "container",
            "style": {
                "background_color": bg_color,
                "padding": padding,
                "border_radius": border_radius,
                "margin_bottom": margin_bottom
            }
        }

# Add alias for backward compatibility
BaseComponent = BaseUIComponent
# Add alias for backward compatibility
BaseComponent = BaseUIComponent