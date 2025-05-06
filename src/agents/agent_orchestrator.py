"""
Agent Orchestrator Module for coordinating multiple specialized agents.

This module manages agent interactions, message passing, and workflow coordination
using the module system.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
import time
import json

from components.base_module import Module, get_module_manager
from utils.logger import get_logger

class AgentOrchestrator(Module):
    """
    Orchestrates interactions between multiple specialized agents.
    
    This class manages agent dependencies, message passing, and coordinated 
    workflows across different agent types.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the agent orchestrator"""
        super().__init__(module_id, config)
        self.agent_modules = {}
        self.workflows = {}
        self.current_workflows = {}
        self.workflow_history = {}
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator and register available agents"""
        await super().initialize()
        
        self.logger.info("Initializing Agent Orchestrator")
        
        # Register workflow patterns
        self._register_workflows()
        
        # Expose services
        self.expose_service("execute_workflow", self.execute_workflow)
        self.expose_service("register_agent", self.register_agent)
        self.expose_service("send_message", self.send_message)
        
        return True
    
    def _register_workflows(self) -> None:
        """Register standard workflow patterns"""
        # Register the basic chat workflow
        self.register_workflow(
            "basic_chat",
            ["safety_agent", "chat_agent"],
            {
                "description": "Basic chat workflow with safety checks",
                "default_timeout": 30
            }
        )
        
        # Register the diagnosis workflow
        self.register_workflow(
            "diagnosis",
            ["safety_agent", "emotion_agent", "diagnosis_agent"],
            {
                "description": "Psychological diagnosis workflow",
                "default_timeout": 60
            }
        )
        
        # Register the integrated diagnosis workflow
        self.register_workflow(
            "integrated_diagnosis",
            ["safety_agent", "emotion_agent", "personality_agent", "integrated_diagnosis_agent"],
            {
                "description": "Integrated psychological diagnosis with personality assessment",
                "default_timeout": 90
            }
        )
        
        # Register the search workflow
        self.register_workflow(
            "search",
            ["safety_agent", "search_agent", "crawler_agent"],
            {
                "description": "Web search workflow with safety checks",
                "default_timeout": 45
            }
        )
        
        self.logger.debug(f"Registered {len(self.workflows)} standard workflows")
    
    def register_agent(self, agent_id: str, agent_module: Module) -> bool:
        """
        Register an agent module with the orchestrator.
        
        Args:
            agent_id: Identifier for the agent
            agent_module: Agent module instance
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if agent_id in self.agent_modules:
            self.logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.agent_modules[agent_id] = agent_module
        self.logger.debug(f"Registered agent: {agent_id}")
        return True
    
    def register_workflow(self, workflow_id: str, agent_sequence: List[str], 
                         workflow_config: Dict[str, Any] = None) -> bool:
        """
        Register a workflow pattern with the orchestrator.
        
        Args:
            workflow_id: Identifier for the workflow
            agent_sequence: Ordered list of agent IDs for message passing
            workflow_config: Configuration for the workflow
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if workflow_id in self.workflows:
            self.logger.warning(f"Workflow {workflow_id} already registered")
            return False
        
        self.workflows[workflow_id] = {
            "agent_sequence": agent_sequence,
            "config": workflow_config or {}
        }
        
        self.logger.debug(f"Registered workflow: {workflow_id} with {len(agent_sequence)} agents")
        return True
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any],
                             session_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a registered workflow with the given input data.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            session_id: Optional session identifier for tracking
            context: Additional context for the workflow
            
        Returns:
            Results of the workflow execution
        """
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return {"error": f"Workflow {workflow_id} not found"}
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Get workflow definition
        workflow = self.workflows[workflow_id]
        agent_sequence = workflow["agent_sequence"]
        
        # Prepare workflow state
        workflow_state = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "input": input_data,
            "context": context or {},
            "results": {},
            "start_time": time.time(),
            "current_step": 0,
            "steps_completed": 0,
            "agent_sequence": agent_sequence,
            "status": "in_progress"
        }
        
        # Store current workflow
        self.current_workflows[session_id] = workflow_state
        
        # Log workflow start
        self.logger.info(f"Starting workflow: {workflow_id}", 
                      {"session_id": session_id, "workflow": workflow_id})
        
        # Execute each agent in sequence
        current_data = input_data
        
        for idx, agent_id in enumerate(agent_sequence):
            # Update workflow state
            workflow_state["current_step"] = idx
            workflow_state["current_agent"] = agent_id
            
            # Check if agent is available
            if agent_id not in self.agent_modules:
                error_msg = f"Agent {agent_id} not found, workflow cannot continue"
                self.logger.error(error_msg, {"session_id": session_id})
                workflow_state["status"] = "failed"
                workflow_state["error"] = error_msg
                break
            
            # Process with current agent
            try:
                self.logger.debug(f"Executing agent {agent_id} in workflow {workflow_id}", 
                              {"session_id": session_id})
                
                agent = self.agent_modules[agent_id]
                agent_process = getattr(agent, "process", None)
                
                if not agent_process or not callable(agent_process):
                    error_msg = f"Agent {agent_id} does not have a valid process method"
                    self.logger.error(error_msg, {"session_id": session_id})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
                
                # Execute agent's process method
                result = await agent_process(current_data, context=workflow_state["context"])
                
                # Store result
                workflow_state["results"][agent_id] = result
                
                # Update data for next agent
                current_data = result
                
                # Update workflow state
                workflow_state["steps_completed"] += 1
                
            except Exception as e:
                error_msg = f"Error processing agent {agent_id}: {str(e)}"
                self.logger.error(error_msg, {"session_id": session_id, "exception": str(e)})
                workflow_state["status"] = "failed"
                workflow_state["error"] = error_msg
                break
        
        # Complete workflow
        workflow_state["end_time"] = time.time()
        workflow_state["duration"] = workflow_state["end_time"] - workflow_state["start_time"]
        
        # Set final status
        if workflow_state["status"] != "failed":
            workflow_state["status"] = "completed"
        
        # Log workflow completion
        log_level = "info" if workflow_state["status"] == "completed" else "error"
        getattr(self.logger, log_level)(
            f"Workflow {workflow_id} {workflow_state['status']} in {workflow_state['duration']:.2f}s",
            {"session_id": session_id, "workflow": workflow_id, "status": workflow_state["status"]}
        )
        
        # Store in history and remove from current
        self.workflow_history[session_id] = workflow_state
        self.current_workflows.pop(session_id, None)
        
        # Return the final output
        return {
            "output": current_data,
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": workflow_state["status"],
            "duration": workflow_state["duration"],
            "steps_completed": workflow_state["steps_completed"]
        }
    
    async def send_message(self, sender_id: str, recipient_id: str, 
                         message: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            message: Message content to send
            session_id: Optional session identifier for tracking
            
        Returns:
            Response from the recipient agent
        """
        if recipient_id not in self.agent_modules:
            error_msg = f"Recipient agent {recipient_id} not found"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Log message passing
        self.logger.debug(f"Message from {sender_id} to {recipient_id}", 
                      {"session_id": session_id, "sender": sender_id, "recipient": recipient_id})
        
        try:
            # Get the recipient agent
            recipient = self.agent_modules[recipient_id]
            
            # Get the receive_message method if available
            receive_method = getattr(recipient, "receive_message", None)
            
            if not receive_method or not callable(receive_method):
                # Fall back to process method
                receive_method = getattr(recipient, "process", None)
                
                if not receive_method or not callable(receive_method):
                    error_msg = f"Agent {recipient_id} has no valid message handling method"
                    self.logger.error(error_msg)
                    return {"error": error_msg}
            
            # Add metadata to message
            message_with_meta = {
                **message,
                "_meta": {
                    "sender": sender_id,
                    "timestamp": time.time(),
                    "session_id": session_id
                }
            }
            
            # Call the receive method
            response = await receive_method(message_with_meta)
            
            # Log successful message handling
            self.logger.debug(f"Message from {sender_id} to {recipient_id} processed successfully", 
                          {"session_id": session_id})
            
            return response
            
        except Exception as e:
            error_msg = f"Error sending message from {sender_id} to {recipient_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow by session ID.
        
        Args:
            session_id: Session identifier for the workflow
            
        Returns:
            Current workflow status or history if completed
        """
        # Check current workflows
        if session_id in self.current_workflows:
            return {
                "status": "in_progress",
                "workflow": self.current_workflows[session_id]
            }
        
        # Check workflow history
        if session_id in self.workflow_history:
            return {
                "status": "completed",
                "workflow": self.workflow_history[session_id]
            }
        
        return {
            "status": "not_found",
            "error": f"No workflow found for session {session_id}"
        }
    
    async def shutdown(self) -> bool:
        """Shutdown the orchestrator"""
        self.logger.info("Shutting down Agent Orchestrator")
        
        # Clean up any ongoing workflows
        for session_id, workflow in self.current_workflows.items():
            workflow["status"] = "aborted"
            workflow["end_time"] = time.time()
            workflow["duration"] = workflow["end_time"] - workflow["start_time"]
            self.workflow_history[session_id] = workflow
            
            self.logger.warning(f"Workflow {workflow['workflow_id']} aborted during shutdown", 
                            {"session_id": session_id})
        
        self.current_workflows.clear()
        
        return await super().shutdown()