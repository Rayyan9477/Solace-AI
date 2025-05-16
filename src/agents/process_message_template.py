"""
Method to add to the AgentOrchestrator class for processing messages and tracking conversations.
This is a template that will be integrated into the agent_orchestrator.py file.
"""

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
        "user_id": user_id or self.conversation_tracker.user_id
    }
    
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
    
    # Track the conversation in our new tracker
    metadata = {
        "workflow_id": workflow_id,
        "session_id": session_id,
        "duration": result.get("duration") if isinstance(result, dict) else None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add the conversation to the tracker
    if response:
        try:
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
        except Exception as e:
            self.logger.error(f"Error tracking conversation: {str(e)}")
    
    return result
