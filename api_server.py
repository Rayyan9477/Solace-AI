"""
API Server for Contextual-Chatbot

Provides REST API endpoints for mobile app integration, supporting:
- Chat functionality
- Diagnostic assessments
- Voice integration
- User management
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Body, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

# Import application components
from src.main import initialize_application
from src.config.settings import AppConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir.parent, 'logs', 'api_server.log'))
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Contextual-Chatbot API",
    description="API for the Contextual-Chatbot application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
app_state = {
    "initialized": False,
    "app_manager": None
}

# Request/response models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    emotion: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class AssessmentQuestionResponse(BaseModel):
    question_id: str
    response: Any

class DiagnosticAssessmentRequest(BaseModel):
    user_id: str
    assessment_type: str
    responses: Dict[str, Any]

class UserProfileRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    try:
        logger.info("Initializing application...")
        # Initialize with API UI type
        config = AppConfig(ui_type="api")
        app_state["app_manager"] = await initialize_application(config)
        app_state["initialized"] = True
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        app_state["initialized"] = False

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shut down the application on server shutdown"""
    try:
        if app_state["app_manager"]:
            logger.info("Shutting down application...")
            await app_state["app_manager"].shutdown()
            logger.info("Application shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down application: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health of the API server"""
    if not app_state["initialized"]:
        return {"status": "initializing"}
    
    if app_state["app_manager"]:
        health_status = await app_state["app_manager"].get_health_status()
        return {"status": "healthy", "details": health_status}
    
    return {"status": "unhealthy", "details": "Application manager not available"}

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the chat request
        response = await ui_manager.handle_api_request("/api/chat", request.dict())
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Assessment endpoints
@app.get("/api/assessment/questions/{assessment_type}")
async def get_assessment_questions(assessment_type: str):
    """Get assessment questions for a specific assessment type"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the assessment questions request
        response = await ui_manager.handle_api_request("/api/assessment/questions", {
            "assessment_type": assessment_type
        })
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error getting assessment questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assessment/submit")
async def submit_assessment(request: DiagnosticAssessmentRequest):
    """Submit a completed assessment for analysis"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the assessment submission
        response = await ui_manager.handle_api_request("/api/assessment/submit", request.dict())
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error submitting assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice endpoints
@app.post("/api/voice/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio to text"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Process the transcription request
        response = await ui_manager.handle_api_request("/api/voice/transcribe", {
            "audio_data": audio_data
        })
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/synthesize")
async def synthesize_speech(text: str = Body(..., embed=True)):
    """Synthesize text to speech"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the synthesis request
        response = await ui_manager.handle_api_request("/api/voice/synthesize", {
            "text": text
        })
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# User management endpoints
@app.get("/api/user/{user_id}")
async def get_user_profile(user_id: str):
    """Get a user's profile"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the user profile request
        response = await ui_manager.handle_api_request("/api/user", {
            "user_id": user_id
        })
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/update")
async def update_user_profile(request: UserProfileRequest):
    """Update a user's profile"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the user profile update
        response = await ui_manager.handle_api_request("/api/user/update", request.dict())
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Therapy resources endpoints
@app.get("/api/therapy/resources")
async def get_therapy_resources(category: Optional[str] = None):
    """Get therapy resources"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get UI manager to handle API request
        ui_manager = app_state["app_manager"].get_module("ui")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Process the therapy resources request
        response = await ui_manager.handle_api_request("/api/therapy", {
            "category": category
        })
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error getting therapy resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
