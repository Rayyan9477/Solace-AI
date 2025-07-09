"""
UI Manager Module

Provides user interface management capabilities for the application.
Supports API endpoints for mobile app integration.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import logging
import os
import json
from pathlib import Path

from src.components.base_module import Module
from src.config.settings import AppConfig

class UIManager(Module):
    """
    UI Manager Module for the Contextual-Chatbot.
    
    Manages user interface components and interactions:
    - UI rendering and state management
    - API endpoints for mobile app integration
    - Interface event handling
    - UI component registration
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.ui_components = {}
        self.active_ui = None
        self.ui_type = "cli"  # Default UI type: cli, api, web
        self.api_endpoints = {}
        
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
                
            # API backend for mobile app
            elif self.ui_type == "api":
                try:
                    # Register API endpoints for mobile app
                    self._register_api_endpoints()
                    self.logger.info("UI type set to API (for mobile app)")
                    return True
                except Exception as e:
                    self.logger.error(f"Error initializing API UI: {str(e)}")
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
        
        # New services for API/mobile integration
        self.expose_service("register_api_endpoint", self.register_api_endpoint)
        self.expose_service("handle_api_request", self.handle_api_request)
        self.expose_service("send_mobile_notification", self.send_mobile_notification)
    
    def _register_api_endpoints(self):
        """Register API endpoints for mobile app integration"""
        # These are just placeholders for the actual API endpoints
        # The actual implementation would depend on the mobile app requirements
        self.register_api_endpoint("/api/chat", self._api_chat_handler)
        self.register_api_endpoint("/api/diagnosis", self._api_diagnosis_handler)
        self.register_api_endpoint("/api/user", self._api_user_handler)
        self.register_api_endpoint("/api/therapy", self._api_therapy_handler)
        
        self.logger.info(f"Registered {len(self.api_endpoints)} API endpoints")
    
    async def _api_chat_handler(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat API requests from mobile app"""
        try:
            # Get chat module
            chat_module = self.get_dependency("chat")
            if not chat_module:
                return {"error": "Chat module not available"}
            
            # Process message
            message = request_data.get("message", "")
            user_id = request_data.get("user_id", "default_user")
            
            # Get the process_message service
            process_message = chat_module.get_service("process_message")
            if not process_message:
                return {"error": "Chat service not available"}
            
            # Process the message
            response = await process_message(message, user_id)
            return {"response": response}
            
        except Exception as e:
            self.logger.error(f"Error in chat API handler: {str(e)}")
            return {"error": str(e)}
    
    async def _api_diagnosis_handler(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle diagnosis API requests from mobile app"""
        try:
            # Implementation would go here
            return {"status": "not_implemented"}
        except Exception as e:
            self.logger.error(f"Error in diagnosis API handler: {str(e)}")
            return {"error": str(e)}
    
    async def _api_user_handler(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user profile API requests from mobile app"""
        try:
            # Implementation would go here
            return {"status": "not_implemented"}
        except Exception as e:
            self.logger.error(f"Error in user API handler: {str(e)}")
            return {"error": str(e)}
    
    async def _api_therapy_handler(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle therapy resource API requests from mobile app"""
        try:
            # Implementation would go here
            return {"status": "not_implemented"}
        except Exception as e:
            self.logger.error(f"Error in therapy API handler: {str(e)}")
            return {"error": str(e)}
    
    def register_api_endpoint(self, endpoint: str, handler: Callable) -> bool:
        """
        Register an API endpoint for mobile app integration
        
        Args:
            endpoint: The API endpoint path
            handler: The handler function for the endpoint
            
        Returns:
            Success status
        """
        try:
            self.api_endpoints[endpoint] = handler
            self.logger.debug(f"Registered API endpoint: {endpoint}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering API endpoint: {str(e)}")
            return False
    
    async def handle_api_request(self, endpoint: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an API request from the mobile app
        
        Args:
            endpoint: The API endpoint path
            request_data: The request data from the mobile app
            
        Returns:
            Response data for the mobile app
        """
        if endpoint not in self.api_endpoints:
            self.logger.error(f"Unknown API endpoint: {endpoint}")
            return {"error": "Unknown endpoint"}
        
        try:
            handler = self.api_endpoints[endpoint]
            response = await handler(request_data)
            return response
        except Exception as e:
            self.logger.error(f"Error handling API request: {str(e)}")
            return {"error": str(e)}
    
    async def send_mobile_notification(self, user_id: str, notification: Dict[str, Any]) -> bool:
        """
        Send a notification to the mobile app
        
        Args:
            user_id: The user ID to send the notification to
            notification: The notification data
            
        Returns:
            Success status
        """
        try:
            # In a real implementation, this would send a push notification
            # to the mobile app using a service like Firebase Cloud Messaging
            self.logger.info(f"Sending mobile notification to user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending mobile notification: {str(e)}")
            return False
    
    def get_ui_type(self) -> str:
        """
        Get the current UI type
        
        Returns:
            UI type string: "cli", "api", or "web"
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
                
            elif self.ui_type == "api":
                # For API, we just log it as messages are sent via API endpoints
                self.logger.info(f"Message for API: {role}: {message[:50]}...")
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
                
            elif self.ui_type == "api":
                # For API, we just log it as results are sent via API endpoints
                self.logger.info("Diagnostic results ready for API")
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
