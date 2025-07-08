"""
UI Manager Module

Provides user interface management capabilities for the application.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import logging
import os
from pathlib import Path

from src.components.base_module import Module
from src.config.settings import AppConfig

class UIManager(Module):
    """
    UI Manager Module for the Contextual-Chatbot.
    
    Manages user interface components and interactions:
    - UI rendering and state management
    - Interface event handling
    - UI component registration
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.ui_components = {}
        self.active_ui = None
        self.ui_type = "cli"  # Default UI type: cli, streamlit, web
        
        # Initialize config
        self._load_config()
    
    def _load_config(self):
        """Load configuration values"""
        if not self.config:
            return
            
        self.ui_type = self.config.get("ui_type", "cli")
    
    async def initialize(self) -> bool:
        """Initialize the module"""
        await super().initialize()
        
        try:
            # Register services regardless of UI type
            self._register_services()
            
            # CLI UI requires no special setup
            if self.ui_type == "cli":
                self.logger.info("Initialized CLI UI")
                return True
                
            # Try to set up Streamlit UI components if selected
            elif self.ui_type == "streamlit":
                try:
                    # We don't import Streamlit here as it's not required for initialization
                    # Actual UI components will be loaded on demand
                    self.logger.info("UI type set to Streamlit")
                    return True
                except Exception as e:
                    self.logger.error(f"Error initializing Streamlit UI: {str(e)}")
                    self.ui_type = "cli"  # Fallback to CLI
                    return True
            
            # Web UI
            elif self.ui_type == "web":
                try:
                    # Web UI setup would go here
                    self.logger.info("UI type set to Web")
                    return True
                except Exception as e:
                    self.logger.error(f"Error initializing Web UI: {str(e)}")
                    self.ui_type = "cli"  # Fallback to CLI
                    return True
                    
            else:
                self.logger.warning(f"Unknown UI type: {self.ui_type}, falling back to CLI")
                self.ui_type = "cli"
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize UI manager: {str(e)}")
            self.health_status = "degraded"
            self.ui_type = "cli"  # Fallback to CLI
            return True  # Still return True as CLI fallback should work
    
    def _register_services(self):
        """Register services provided by this module"""
        self.expose_service("render_message", self.render_message)
        self.expose_service("render_diagnostic_results", self.render_diagnostic_results)
        self.expose_service("get_ui_type", self.get_ui_type)
        self.expose_service("register_component", self.register_component)
    
    def get_ui_type(self) -> str:
        """
        Get the current UI type
        
        Returns:
            UI type string: "cli", "streamlit", or "web"
        """
        return self.ui_type
    
    def register_component(self, component_id: str, component: Any) -> bool:
        """
        Register a UI component
        
        Args:
            component_id: Unique ID for the component
            component: The component object
            
        Returns:
            Success status
        """
        try:
            self.ui_components[component_id] = component
            self.logger.debug(f"Registered UI component: {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering UI component: {str(e)}")
            return False
    
    async def render_message(self, message: str, role: str = "assistant", 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Render a message in the UI
        
        Args:
            message: Message text to render
            role: Role of the message sender (assistant, user, system)
            metadata: Optional metadata for the message
            
        Returns:
            Success status
        """
        if not self.initialized:
            self.logger.warning("UI manager not initialized")
            return False
        
        try:
            # Different rendering based on UI type
            if self.ui_type == "cli":
                from src.utils.console_utils import safe_print
                prefix = f"{role.capitalize()}: " if role != "system" else ""
                safe_print(f"{prefix}{message}")
                return True
                
            elif self.ui_type == "streamlit":
                # In a real implementation, we would use a callback or event system
                # Here we just log it as we can't directly modify the Streamlit UI
                self.logger.info(f"Render message in Streamlit UI: {role}: {message[:50]}...")
                return True
                
            elif self.ui_type == "web":
                # Web UI rendering would go here
                self.logger.info(f"Render message in Web UI: {role}: {message[:50]}...")
                return True
                
            else:
                # Fallback to CLI
                from src.utils.console_utils import safe_print
                prefix = f"{role.capitalize()}: " if role != "system" else ""
                safe_print(f"{prefix}{message}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error rendering message: {str(e)}")
            return False
    
    async def render_diagnostic_results(self, results: Dict[str, Any]) -> bool:
        """
        Render diagnostic results in the UI
        
        Args:
            results: Diagnostic results dictionary
            
        Returns:
            Success status
        """
        if not self.initialized:
            self.logger.warning("UI manager not initialized")
            return False
        
        try:
            # Different rendering based on UI type
            if self.ui_type == "cli":
                from src.utils.console_utils import safe_print
                
                safe_print("\n===== Diagnostic Results =====")
                
                # Mental health section
                if "mental_health" in results:
                    mh = results["mental_health"]
                    safe_print(f"\nMental Health Status: {mh.get('overall_status', 'Unknown').capitalize()}")
                    
                    # Areas of concern
                    if "areas_of_concern" in mh and mh["areas_of_concern"]:
                        safe_print("\nAreas of Concern:")
                        for area in mh["areas_of_concern"]:
                            safe_print(f"- {area.get('name', 'Unknown')}: {area.get('severity', 'Unknown')} " +
                                     f"({area.get('score', 0)}/10)")
                    
                    # Strengths
                    if "strengths" in mh and mh["strengths"]:
                        safe_print("\nStrengths:")
                        for strength in mh["strengths"]:
                            safe_print(f"- {strength.get('name', 'Unknown')} ({strength.get('score', 0)}/10)")
                
                # Personality section
                if "personality" in results:
                    pers = results["personality"]
                    safe_print("\nPersonality Profile:")
                    
                    if "summary" in pers:
                        safe_print(f"\n{pers['summary']}")
                    
                    # Traits
                    if "traits" in pers and pers["traits"]:
                        safe_print("\nKey Traits:")
                        for trait in pers["traits"]:
                            safe_print(f"- {trait.get('name', 'Unknown')}: {trait.get('score', 0)}/10")
                
                # Recommendations
                if "recommendations" in results and results["recommendations"]:
                    safe_print("\nRecommendations:")
                    for i, rec in enumerate(results["recommendations"], 1):
                        safe_print(f"{i}. {rec}")
                
                safe_print("\n===============================")
                return True
                
            elif self.ui_type == "streamlit":
                # For Streamlit, we would normally use the DiagnosisResultsComponent
                # Here we just log it as we can't directly render in Streamlit
                self.logger.info("Rendering diagnostic results in Streamlit UI")
                return True
                
            else:
                # Fallback to CLI rendering
                from src.utils.console_utils import safe_print
                safe_print("\n===== Diagnostic Results =====")
                safe_print(str(results))
                safe_print("\n===============================")
                return True
                
        except Exception as e:
            self.logger.error(f"Error rendering diagnostic results: {str(e)}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the module"""
        # No special cleanup needed for most UI types
        return await super().shutdown()
