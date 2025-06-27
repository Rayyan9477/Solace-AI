"""
UI Component Manager for handling UI navigation and state management.
"""

from typing import Dict, Any, List, Optional, Callable, Type
import streamlit as st
import logging

from .base_component import BaseUIComponent

logger = logging.getLogger(__name__)

class UIComponentManager:
    """
    Manages UI components and navigation between them.
    Acts as a central coordinator for all UI components.
    """
    
    def __init__(self):
        """Initialize the UI manager"""
        self.components: Dict[str, BaseUIComponent] = {}
        self.current_route: Optional[str] = None
        self.route_history: List[str] = []
        self.max_history = 10
        
        # Initialize global UI state if needed
        if "ui_manager_state" not in st.session_state:
            st.session_state.ui_manager_state = {
                "current_route": None,
                "route_history": [],
                "mode": None,  # 'voice' or 'text'
                "theme": "default",
                "show_sidebar": True
            }
    
    def register_component(self, route: str, component: BaseUIComponent) -> None:
        """
        Register a UI component with the manager
        
        Args:
            route: Route identifier for the component
            component: The UI component instance to register
        """
        if route in self.components:
            logger.warning(f"Overwriting existing component at route '{route}'")
            
        self.components[route] = component
        logger.debug(f"Registered component '{component.name}' at route '{route}'")
    
    def navigate_to(self, route: str, add_to_history: bool = True) -> None:
        """
        Navigate to a specific route
        
        Args:
            route: The route to navigate to
            add_to_history: Whether to add this route to history
        """
        if route not in self.components:
            logger.error(f"Cannot navigate to unknown route: '{route}'")
            return
            
        # Add current route to history before changing
        if add_to_history and self.current_route:
            self.route_history.append(self.current_route)
            # Limit history size
            if len(self.route_history) > self.max_history:
                self.route_history.pop(0)
            
            # Update session state
            st.session_state.ui_manager_state["route_history"] = self.route_history
        
        # Set new route
        self.current_route = route
        st.session_state.ui_manager_state["current_route"] = route
        logger.debug(f"Navigated to route: '{route}'")
        
        # Force a rerun to apply the route change
        st.rerun()
    
    def go_back(self) -> None:
        """Navigate back to the previous route"""
        if not self.route_history:
            logger.warning("Cannot go back - no route history available")
            return
            
        # Pop the last route from history
        prev_route = self.route_history.pop()
        
        # Update session state
        st.session_state.ui_manager_state["route_history"] = self.route_history
        
        # Navigate to the previous route without adding to history
        self.navigate_to(prev_route, add_to_history=False)
    
    def render_current(self, **kwargs) -> None:
        """
        Render the current route's component
        
        Args:
            **kwargs: Additional parameters to pass to the component
        """
        # Restore state from session
        self.current_route = st.session_state.ui_manager_state["current_route"]
        self.route_history = st.session_state.ui_manager_state["route_history"]
        
        # If no route is set, use the default (first registered route)
        if not self.current_route and self.components:
            self.current_route = next(iter(self.components.keys()))
            st.session_state.ui_manager_state["current_route"] = self.current_route
        
        # Render the current component if available
        if self.current_route and self.current_route in self.components:
            self.components[self.current_route].render(**kwargs)
        else:
            st.error("No UI components registered or current route is invalid.")
    
    def set_interaction_mode(self, mode: str) -> None:
        """
        Set the interaction mode (voice or text)
        
        Args:
            mode: The interaction mode ('voice' or 'text')
        """
        if mode not in ['voice', 'text']:
            logger.warning(f"Invalid interaction mode: {mode}. Must be 'voice' or 'text'")
            return
            
        st.session_state.ui_manager_state["mode"] = mode
        logger.debug(f"Set interaction mode to: {mode}")
    
    def get_interaction_mode(self) -> Optional[str]:
        """
        Get the current interaction mode
        
        Returns:
            The current interaction mode ('voice' or 'text') or None if not set
        """
        return st.session_state.ui_manager_state.get("mode")