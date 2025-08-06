"""
Agent Orchestrator Module for coordinating multiple specialized agents.

This module manages agent interactions, message passing, and workflow coordination
using the module system with integrated SupervisorAgent oversight.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
import time
import json
from datetime import datetime

from src.components.base_module import Module, get_module_manager
from src.utils.logger import get_logger
from src.utils.vector_db_integration import get_conversation_tracker, search_relevant_data
from src.agents.supervisor_agent import SupervisorAgent, ValidationLevel
from src.monitoring.supervisor_metrics import MetricsCollector, QualityMetrics
from src.auditing.audit_system import AuditTrail, AuditEventType, AuditSeverity

# Import diagnosis services
try:
    from src.services.diagnosis import (
        IDiagnosisOrchestrator, IDiagnosisAgentAdapter,
        DiagnosisRequest, DiagnosisType
    )
    from src.infrastructure.di.container import get_container
    DIAGNOSIS_SERVICES_AVAILABLE = True
except ImportError:
    DIAGNOSIS_SERVICES_AVAILABLE = False

class AgentOrchestrator(Module):
    """
    Orchestrates interactions between multiple specialized agents.
    
    This class manages agent dependencies, message passing, and coordinated 
    workflows across different agent types.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the agent orchestrator with supervision capabilities"""
        super().__init__(module_id, config)
        self.agent_modules = {}
        self.workflows = {}
        self.current_workflows = {}
        self.workflow_history = {}
        
        # Initialize context store for shared context between agents
        self.context_store = {}
        
        # Store user_id for conversation tracking
        self.user_id = config.get("user_id", "default_user") if config else "default_user"
        self.conversation_tracker = None  # Will be initialized when needed
        
        # Initialize diagnosis system integration
        self.diagnosis_orchestrator = None
        self.diagnosis_adapter = None
        self.diagnosis_integration_enabled = (
            config.get("diagnosis_integration_enabled", True) if config else True
        ) and DIAGNOSIS_SERVICES_AVAILABLE
        
        # Initialize supervision system
        self.supervision_enabled = config.get("supervision_enabled", True) if config else True
        self.supervisor_agent = None
        self.metrics_collector = None
        self.audit_trail = None
        
        if self.supervision_enabled:
            try:
                # Initialize SupervisorAgent
                model_provider = config.get("model_provider") if config else None
                self.supervisor_agent = SupervisorAgent(model_provider, config)
                
                # Initialize metrics collection
                self.metrics_collector = MetricsCollector()
                
                # Initialize audit trail
                self.audit_trail = AuditTrail()
                
                self.logger.info("Supervision system initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize supervision system: {str(e)}")
                self.supervision_enabled = False
        
        self.logger.info(f"Agent orchestrator configured for user {self.user_id} with supervision {'enabled' if self.supervision_enabled else 'disabled'}")
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator and register available agents"""
        await super().initialize()
        
        self.logger.info("Initializing Agent Orchestrator")
        
        # Register workflow patterns
        self._register_workflows()
        
        # Initialize diagnosis system integration
        await self._initialize_diagnosis_integration()
        
        # Expose services
        self.expose_service("execute_workflow", self.execute_workflow)
        self.expose_service("register_agent", self.register_agent)
        self.expose_service("send_message", self.send_message)
        self.expose_service("get_context", self.get_context)
        self.expose_service("update_context", self.update_context)
        
        # Expose diagnosis service if available
        if self.diagnosis_integration_enabled:
            self.expose_service("diagnose", self.diagnose)
        
        return True
    
    async def _initialize_diagnosis_integration(self) -> None:
        """Initialize integration with the unified diagnosis system."""
        if not self.diagnosis_integration_enabled:
            self.logger.info("Diagnosis integration disabled or not available")
            return
        
        try:
            # Get DI container
            container = get_container()
            
            # Resolve diagnosis services
            self.diagnosis_orchestrator = await container.resolve(IDiagnosisOrchestrator)
            self.diagnosis_adapter = await container.resolve(IDiagnosisAgentAdapter)
            
            self.logger.info("Diagnosis system integration initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize diagnosis integration: {str(e)}")
            self.diagnosis_integration_enabled = False
    
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
        
        # Register unified diagnosis workflow using new diagnosis system
        if DIAGNOSIS_SERVICES_AVAILABLE:
            self.register_workflow(
                "unified_diagnosis",
                ["safety_agent", "emotion_agent", "personality_agent", "unified_diagnosis_service"],
                {
                    "description": "Unified diagnosis using enhanced integrated diagnostic system",
                    "default_timeout": 120,
                    "use_diagnosis_service": True
                }
            )
            
            # Register comprehensive diagnosis workflow
            self.register_workflow(
                "comprehensive_diagnosis",
                ["safety_agent", "emotion_agent", "personality_agent", "unified_diagnosis_service"],
                {
                    "description": "Comprehensive diagnosis with all available systems",
                    "default_timeout": 150,
                    "use_diagnosis_service": True,
                    "diagnosis_type": "comprehensive"
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
        
        # Register enhanced chat workflow with Gemini 2.0
        self.register_workflow(
            "enhanced_empathetic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "chat_agent"],
            {
                "description": "Enhanced chat workflow with emotion analysis and personalized empathetic responses using Gemini 2.0",
                "default_timeout": 45
            }
        )
        
        # Register therapeutic chat workflow with actionable steps
        self.register_workflow(
            "therapeutic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "chat_agent"],
            {
                "description": "Therapeutic chat workflow with practical actionable steps based on evidence-based techniques",
                "default_timeout": 60
            }
        )
        
        # Register growth-oriented therapeutic friction workflow
        self.register_workflow(
            "therapeutic_friction_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapeutic_friction_agent", "chat_agent"],
            {
                "description": "Advanced therapeutic workflow with growth-oriented challenges and strategic friction",
                "default_timeout": 75,
                "use_legacy_friction_agent": True
            }
        )
        
        # Register coordinated sub-agent therapeutic friction workflow
        self.register_workflow(
            "coordinated_therapeutic_friction",
            ["safety_agent", "emotion_agent", "personality_agent", "friction_coordinator", "chat_agent"],
            {
                "description": "Advanced therapeutic workflow using coordinated sub-agent friction analysis",
                "default_timeout": 90,
                "use_sub_agent_coordination": True
            }
        )
        
        # Register integrated therapeutic workflow combining both approaches
        self.register_workflow(
            "integrated_therapeutic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "therapeutic_friction_agent", "chat_agent"],
            {
                "description": "Comprehensive therapeutic workflow combining evidence-based techniques with growth-oriented friction",
                "default_timeout": 90,
                "use_legacy_friction_agent": True
            }
        )
        
        # Register comprehensive coordinated therapeutic workflow
        self.register_workflow(
            "comprehensive_coordinated_therapeutic",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "friction_coordinator", "chat_agent"],
            {
                "description": "Comprehensive therapeutic workflow with evidence-based techniques and coordinated sub-agent friction analysis",
                "default_timeout": 120,
                "use_sub_agent_coordination": True,
                "enable_agent_integration": True
            }
        )
        
        self.logger.debug(f"Registered {len(self.workflows)} standard workflows")
    
    def register_agent(self, agent_id: str, agent_module: Module) -> bool:
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
        
        # Update context store with initial context
        if context:
            await self.update_context(session_id, context)
        
        # Get full context for this session
        full_context = await self.get_context(session_id)
        
        # Prepare workflow state
        workflow_state = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "input": input_data,
            "context": full_context,
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
            
            # Check if this step should use the diagnosis service
            if (agent_id == "unified_diagnosis_service" and 
                workflow["config"].get("use_diagnosis_service", False) and 
                self.diagnosis_integration_enabled and 
                self.diagnosis_orchestrator):
                
                # Use unified diagnosis service instead of agent
                try:
                    self.logger.debug(f"Executing diagnosis service in workflow {workflow_id}", 
                                  {"session_id": session_id})
                    
                    # Record service interaction start
                    agent_start_time = time.time()
                    
                    result = await self._execute_diagnosis_service(
                        current_data, workflow_state, workflow["config"]
                    )
                    
                    # Calculate processing time
                    agent_processing_time = time.time() - agent_start_time
                    
                except Exception as e:
                    error_msg = f"Diagnosis service execution failed: {str(e)}"
                    self.logger.error(error_msg, {"session_id": session_id})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
            else:
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
                    
                    # Refresh context before processing
                    workflow_state["context"] = await self.get_context(session_id)
                    
                    # Record agent interaction start
                    agent_start_time = time.time()
                    
                    # Execute agent's process method with updated context
                    result = await agent_process(current_data, context=workflow_state["context"])
                    
                    # Calculate processing time
                    agent_processing_time = time.time() - agent_start_time
                    
                except Exception as e:
                    error_msg = f"Error processing agent {agent_id}: {str(e)}"
                    self.logger.error(error_msg, {"session_id": session_id, "exception": str(e)})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
                
                # Perform supervision validation if enabled
                if self.supervision_enabled and self.supervisor_agent:
                    try:
                        validation_result = await self.supervisor_agent.validate_agent_response(
                            agent_name=agent_id,
                            input_data=current_data,
                            output_data=result,
                            session_id=session_id
                        )
                        
                        # Record metrics
                        if self.metrics_collector:
                            self.metrics_collector.record_validation_metrics(
                                agent_name=agent_id,
                                validation_result=validation_result,
                                processing_time=agent_processing_time,
                                session_id=session_id
                            )
                        
                        # Log audit event
                        if self.audit_trail:
                            self.audit_trail.log_agent_interaction(
                                session_id=session_id,
                                user_id=self.user_id,
                                agent_name=agent_id,
                                user_input=str(current_data.get("message", "")),
                                agent_response=str(result.get("response", "")),
                                validation_result=validation_result,
                                processing_time=agent_processing_time
                            )
                        
                        # Handle validation results
                        if validation_result.validation_level == ValidationLevel.BLOCKED:
                            # Block the response and use alternative
                            self.logger.warning(f"Response blocked for {agent_id}", 
                                              {"session_id": session_id, "reason": validation_result.blocking_issues})
                            
                            if validation_result.alternative_response:
                                result = {"response": validation_result.alternative_response}
                            else:
                                error_msg = f"Agent {agent_id} response blocked due to safety concerns"
                                workflow_state["status"] = "failed"
                                workflow_state["error"] = error_msg
                                break
                            
                            # Log response blocking
                            if self.audit_trail:
                                self.audit_trail.log_response_blocked(
                                    session_id=session_id,
                                    user_id=self.user_id,
                                    agent_name=agent_id,
                                    blocked_content=str(result.get("response", "")),
                                    reason="; ".join(validation_result.blocking_issues),
                                    alternative_provided=bool(validation_result.alternative_response)
                                )
                        
                        elif validation_result.validation_level == ValidationLevel.CRITICAL:
                            # Log critical issues but continue
                            self.logger.error(f"Critical validation issues for {agent_id}", 
                                            {"session_id": session_id, "issues": validation_result.critical_issues})
                        
                        # Store validation result in workflow state
                        workflow_state["validation_results"] = workflow_state.get("validation_results", {})
                        workflow_state["validation_results"][agent_id] = {
                            "validation_level": validation_result.validation_level.value,
                            "overall_score": validation_result.overall_score,
                            "critical_issues": validation_result.critical_issues,
                            "recommendations": validation_result.recommendations
                        }
                        
                    except Exception as validation_error:
                        self.logger.error(f"Validation error for {agent_id}: {str(validation_error)}", 
                                        {"session_id": session_id})
                        # Continue processing even if validation fails
                
            # Store result in workflow state
            workflow_state["results"][agent_id] = result
            
            # Extract and store context updates from the result if available
            if isinstance(result, dict) and "context_updates" in result:
                # Update context with agent's new information
                context_updates = result.pop("context_updates")
                if context_updates:
                    await self.update_context(session_id, context_updates)
                    self.logger.debug(f"Updated context from {agent_id}", 
                                  {"session_id": session_id, "context_keys": list(context_updates.keys())})
            
            # Update data for next agent
            current_data = result
            
            # Update workflow state
            workflow_state["steps_completed"] += 1
        
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
        
        # Add final context to the return data
        final_context = await self.get_context(session_id)
        
        # Return the final output
        return {
            "output": current_data,
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": workflow_state["status"],
            "duration": workflow_state["duration"],
            "steps_completed": workflow_state["steps_completed"],
            "context": final_context
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
    
    async def get_context(self, session_id: str, context_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get context data for a session
        
        Args:
            session_id: Session identifier
            context_type: Optional specific context type to retrieve (emotion, safety, personality, etc.)
            
        Returns:
            Context data for the session
        """
        # Initialize context if not exists
        if session_id not in self.context_store:
            self.context_store[session_id] = {}
            
        # Return specific context type if requested
        if context_type:
            return {
                context_type: self.context_store[session_id].get(context_type, {})
            }
            
        # Return all context
        return self.context_store[session_id]
        
    async def update_context(self, session_id: str, context_data: Dict[str, Any], merge: bool = True) -> bool:
        """
        Update context data for a session
        
        Args:
            session_id: Session identifier
            context_data: New context data to update
            merge: If True, merge with existing context; if False, replace it
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize context if not exists
            if session_id not in self.context_store:
                self.context_store[session_id] = {}
                
            # Update context based on merge strategy
            if merge:
                # Recursively merge nested dictionaries
                self._deep_merge(self.context_store[session_id], context_data)
            else:
                # Replace context entirely
                self.context_store[session_id] = context_data
                
            self.logger.debug(f"Updated context for session {session_id}", 
                          {"session_id": session_id, "context_keys": list(context_data.keys())})
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating context for session {session_id}: {str(e)}")
            return False
            
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dict into target dict
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            else:
                # Update or add non-dictionary items
                target[key] = value
    
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
        
        # Shutdown supervision system
        if self.supervision_enabled:
            try:
                if self.metrics_collector:
                    # Export final metrics
                    from src.monitoring.supervisor_metrics import MetricsExporter
                    exporter = MetricsExporter(self.metrics_collector)
                    export_path = f"logs/final_metrics_{int(time.time())}.json"
                    exporter.export_to_json(export_path)
                    self.logger.info(f"Final metrics exported to {export_path}")
                
                if self.audit_trail:
                    # Cleanup expired audit records
                    cleaned = self.audit_trail.cleanup_expired_records()
                    self.logger.info(f"Cleaned up {cleaned} expired audit records")
                
                self.logger.info("Supervision system shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during supervision system shutdown: {str(e)}")
        
        return await super().shutdown()

    async def process_message(self, message: str, user_id: str = None, workflow_id: str = "enhanced_empathetic_chat") -> Dict[str, Any]:
        """
        Process a user message through the appropriate workflow and track the conversation
        
        Args:
            message: The user message to process
            user_id: User identifier (optional, uses the default from initialization if not provided)
            workflow_id: ID of the workflow to use for processing
        
        Returns:
            Result of processing the message
        """
        # Generate session ID for this interaction
        session_id = f"session_{int(time.time())}"
        
        self.logger.info(f"Processing message with workflow {workflow_id}", 
                      {"session_id": session_id, "message_length": len(message)})
        
        # Set up initial context
        initial_context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id or self.user_id
        }
        
        # Enhance context with relevant data from vector database
        try:
            # Add relevant past conversations
            past_conversations = search_relevant_data(message, ["conversation"], limit=3)
            if past_conversations:
                initial_context["relevant_conversations"] = past_conversations
                self.logger.debug(f"Added {len(past_conversations)} relevant conversations to context")
                
            # Add most recent diagnosis if available
            from utils.vector_db_integration import get_user_data
            latest_diagnosis = get_user_data("diagnosis")
            if latest_diagnosis:
                initial_context["latest_diagnosis"] = latest_diagnosis
                self.logger.debug("Added latest diagnosis to context")
                
            # Add most recent personality assessment if available
            latest_personality = get_user_data("personality")
            if latest_personality:
                initial_context["personality"] = latest_personality
                self.logger.debug("Added personality profile to context")
                
        except Exception as e:
            self.logger.warning(f"Error enhancing context from vector DB: {str(e)}")
        
        # Execute the workflow
        result = await self.execute_workflow(
            workflow_id=workflow_id,
            input_data={"message": message},
            session_id=session_id,
            context=initial_context
        )
        
        # Extract response and emotion data
        response = ""
        emotion_data = None
        
        if isinstance(result, dict):
            # Extract the main response text
            if "output" in result and isinstance(result["output"], dict):
                response = result["output"].get("response", "")
            elif "response" in result:
                response = result["response"]
            
            # Extract emotion data if available
            if "emotion_agent" in result.get("steps_completed", []):
                emotion_result = result.get("results", {}).get("emotion_agent", {})
                if emotion_result and isinstance(emotion_result, dict):
                    emotion_data = emotion_result.get("emotion_analysis")
        
        # Track the conversation in our central vector DB
        metadata = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "duration": result.get("duration") if isinstance(result, dict) else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the conversation to the tracker
        if response:
            try:
                # Get conversation tracker from central vector DB
                if self.conversation_tracker is None:
                    self.conversation_tracker = get_conversation_tracker()
                
                if self.conversation_tracker:
                    conversation_id = self.conversation_tracker.add_conversation(
                        user_message=message,
                        assistant_response=response,
                        emotion_data=emotion_data,
                        metadata=metadata
                    )
                    if conversation_id:
                        self.logger.info(f"Tracked conversation: {conversation_id}")
                        if isinstance(result, dict):
                            result["conversation_id"] = conversation_id
                else:
                    self.logger.warning("Conversation tracker not available")
            except Exception as e:
                self.logger.error(f"Error tracking conversation: {str(e)}")
        
        return result
    
    # Supervision and Monitoring Methods
    
    async def get_supervision_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive supervision summary."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            summary = {
                "supervision_status": "active",
                "time_window_hours": time_window_hours,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get performance metrics
            if self.metrics_collector:
                from src.monitoring.supervisor_metrics import PerformanceDashboard
                dashboard = PerformanceDashboard(self.metrics_collector)
                summary["real_time_metrics"] = dashboard.get_real_time_metrics()
                summary["system_analytics"] = dashboard.get_system_analytics()
            
            # Get supervisor performance
            if self.supervisor_agent:
                summary["supervisor_metrics"] = self.supervisor_agent.get_performance_metrics()
            
            # Get audit statistics
            if self.audit_trail:
                from datetime import timedelta
                start_time = datetime.now() - timedelta(hours=time_window_hours)
                
                # Get critical events
                from src.auditing.audit_system import AuditEventType, AuditSeverity
                critical_events = self.audit_trail.get_events_by_type(
                    AuditEventType.CRISIS_DETECTED, start_time
                )
                blocked_responses = self.audit_trail.get_events_by_type(
                    AuditEventType.RESPONSE_BLOCKED, start_time
                )
                
                summary["audit_summary"] = {
                    "critical_events": len(critical_events),
                    "blocked_responses": len(blocked_responses),
                    "total_interactions": len(self.audit_trail.get_events_by_type(
                        AuditEventType.AGENT_INTERACTION, start_time
                    ))
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating supervision summary: {str(e)}")
            return {"error": f"Failed to generate summary: {str(e)}"}
    
    async def get_agent_quality_report(self, agent_name: str = None) -> Dict[str, Any]:
        """Get quality report for specific agent or all agents."""
        if not self.supervision_enabled or not self.metrics_collector:
            return {"error": "Supervision or metrics not available"}
        
        try:
            from src.monitoring.supervisor_metrics import PerformanceDashboard
            from datetime import timedelta
            
            dashboard = PerformanceDashboard(self.metrics_collector)
            report = dashboard.get_agent_performance_report(
                agent_name=agent_name,
                time_window=timedelta(days=1)
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating agent quality report: {str(e)}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    async def get_session_analysis(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a specific session."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            analysis = {
                "session_id": session_id,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Get audit trail for session
            if self.audit_trail:
                audit_events = self.audit_trail.get_session_audit_trail(session_id)
                analysis["audit_events_count"] = len(audit_events)
                
                # Categorize events
                event_summary = {}
                for event in audit_events:
                    event_type = event.event_type.value
                    event_summary[event_type] = event_summary.get(event_type, 0) + 1
                
                analysis["event_summary"] = event_summary
                
                # Check for critical issues
                critical_events = [e for e in audit_events if e.severity.value in ["critical", "emergency"]]
                analysis["critical_issues"] = len(critical_events)
                analysis["critical_details"] = [
                    {
                        "event_type": e.event_type.value,
                        "severity": e.severity.value,
                        "description": e.event_description,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in critical_events
                ]
            
            # Get supervisor session summary
            if self.supervisor_agent:
                supervisor_summary = await self.supervisor_agent.get_session_summary(session_id)
                analysis["supervisor_summary"] = supervisor_summary
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating session analysis: {str(e)}")
            return {"error": f"Failed to generate analysis: {str(e)}"}
    
    async def export_compliance_report(self, compliance_standard: str, 
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """Export compliance report for regulatory purposes."""
        if not self.supervision_enabled or not self.audit_trail:
            return {"error": "Supervision or audit trail not available"}
        
        try:
            from src.auditing.audit_system import ComplianceStandard
            from datetime import datetime
            
            # Parse compliance standard
            try:
                standard = ComplianceStandard(compliance_standard.lower())
            except ValueError:
                return {"error": f"Invalid compliance standard: {compliance_standard}"}
            
            # Parse dates
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Generate compliance report
            compliance_report = self.audit_trail.generate_compliance_report(
                compliance_standard=standard,
                start_date=start_dt,
                end_date=end_dt
            )
            
            # Export audit data
            export_path = f"exports/compliance_{compliance_standard}_{start_date}_{end_date}.json"
            self.audit_trail.export_audit_data(
                output_path=export_path,
                start_date=start_dt,
                end_date=end_dt
            )
            
            return {
                "compliance_report": {
                    "report_id": compliance_report.report_id,
                    "compliance_standard": compliance_report.compliance_standard.value,
                    "reporting_period": {
                        "start": compliance_report.reporting_period["start"].isoformat(),
                        "end": compliance_report.reporting_period["end"].isoformat()
                    },
                    "total_events": compliance_report.total_events,
                    "violations_found": compliance_report.violations_found,
                    "compliance_score": compliance_report.compliance_score,
                    "critical_findings": compliance_report.critical_findings,
                    "recommendations": compliance_report.recommendations
                },
                "export_path": export_path,
                "generated_timestamp": compliance_report.generated_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting compliance report: {str(e)}")
            return {"error": f"Failed to export report: {str(e)}"}
    
    async def configure_supervision(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure supervision system parameters."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            result = {"configured": [], "errors": []}
            
            # Configure supervisor agent
            if "supervisor_settings" in config and self.supervisor_agent:
                # This would update supervisor configuration
                result["configured"].append("supervisor_settings")
            
            # Configure metrics collection
            if "metrics_settings" in config and self.metrics_collector:
                metrics_config = config["metrics_settings"]
                
                # Update thresholds
                if "thresholds" in metrics_config:
                    self.metrics_collector.metric_thresholds.update(metrics_config["thresholds"])
                    result["configured"].append("metrics_thresholds")
            
            # Configure audit settings
            if "audit_settings" in config and self.audit_trail:
                # This would update audit configuration
                result["configured"].append("audit_settings")
            
            self.logger.info(f"Supervision configuration updated: {result['configured']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error configuring supervision: {str(e)}")
            return {"error": f"Configuration failed: {str(e)}"}
    
    def get_supervision_status(self) -> Dict[str, Any]:
        """Get current supervision system status."""
        return {
            "supervision_enabled": self.supervision_enabled,
            "supervisor_agent_active": self.supervisor_agent is not None,
            "metrics_collector_active": self.metrics_collector is not None,
            "audit_trail_active": self.audit_trail is not None,
            "active_workflows": len(self.current_workflows),
            "total_agents": len(self.agent_modules),
            "status_timestamp": datetime.now().isoformat()
        }
    
    # Diagnosis Service Integration Methods
    
    async def _execute_diagnosis_service(self, 
                                       current_data: Dict[str, Any],
                                       workflow_state: Dict[str, Any],
                                       workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute diagnosis using the unified diagnosis service."""
        try:
            # Convert current data to diagnosis request format
            diagnosis_request = await self.diagnosis_adapter.adapt_agent_request(
                agent_input=current_data,
                context={
                    "user_id": workflow_state.get("context", {}).get("user_id", self.user_id),
                    "session_id": workflow_state["session_id"],
                    "workflow_id": workflow_state["workflow_id"],
                    "agent_type": "orchestrator",
                    "diagnosis_type": workflow_config.get("diagnosis_type", "comprehensive")
                }
            )
            
            # Perform diagnosis
            diagnosis_result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Convert result back to agent format
            agent_format = workflow_config.get("agent_format", "comprehensive")
            adapted_result = await self.diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, agent_format
            )
            
            # Add context updates from diagnosis
            if "context_updates" in adapted_result:
                workflow_state["context"].update(adapted_result["context_updates"])
            
            self.logger.info(f"Diagnosis service executed successfully for session {workflow_state['session_id']}")
            return adapted_result
            
        except Exception as e:
            self.logger.error(f"Error executing diagnosis service: {str(e)}")
            raise
    
    async def diagnose(self, 
                      message: str, 
                      user_id: str = None,
                      diagnosis_type: str = "comprehensive",
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Direct diagnosis service endpoint."""
        if not self.diagnosis_integration_enabled or not self.diagnosis_orchestrator:
            return {"error": "Diagnosis service not available"}
        
        try:
            # Create diagnosis request
            session_id = f"direct_diagnosis_{int(time.time())}"
            
            # Convert diagnosis type string to enum
            try:
                diag_type = DiagnosisType(diagnosis_type.lower())
            except ValueError:
                diag_type = DiagnosisType.COMPREHENSIVE
            
            diagnosis_request = DiagnosisRequest(
                user_id=user_id or self.user_id,
                session_id=session_id,
                message=message,
                conversation_history=[],
                context=context or {},
                diagnosis_type=diag_type
            )
            
            # Perform diagnosis
            diagnosis_result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Adapt result for return
            adapted_result = await self.diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, "comprehensive"
            )
            
            return adapted_result
            
        except Exception as e:
            self.logger.error(f"Error in direct diagnosis: {str(e)}")
            return {"error": str(e)}
    
    def get_diagnosis_status(self) -> Dict[str, Any]:
        """Get diagnosis system integration status."""
        return {
            "diagnosis_integration_enabled": self.diagnosis_integration_enabled,
            "diagnosis_services_available": DIAGNOSIS_SERVICES_AVAILABLE,
            "diagnosis_orchestrator_active": self.diagnosis_orchestrator is not None,
            "diagnosis_adapter_active": self.diagnosis_adapter is not None,
            "status_timestamp": datetime.now().isoformat()
        }