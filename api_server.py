"""
API Server for Contextual-Chatbot

Provides REST API endpoints for mobile app integration, supporting:
- Chat functionality
- Diagnostic assessments
- Voice integration
- User management
"""

import os
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

# Import application components
from src.main import Application
from src.config.settings import AppConfig

# Set up logging
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'api_server.log')
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
        # Initialize application
        application = Application()
        success = await application.initialize()
        if success:
            app_state["app_manager"] = application
            app_state["initialized"] = True
            logger.info("Application initialized successfully")
        else:
            logger.error("Application initialization failed")
            app_state["initialized"] = False
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
        health_status = await app_state["app_manager"].health_check()
        return {"status": "healthy", "details": health_status}
    
    return {"status": "unhealthy", "details": "Application manager not available"}

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get module manager to handle API request  
        module_manager = app_state["app_manager"].module_manager
        ui_manager = module_manager.get_module("ui_manager")
        
        if not ui_manager:
            raise HTTPException(status_code=503, detail="UI manager not available")
        
        # Get chat agent from module manager
        chat_agent = module_manager.get_module("chat_agent")
        
        if not chat_agent:
            raise HTTPException(status_code=503, detail="Chat agent not available")
        
        # Process the chat message
        result = await chat_agent.process_message(
            request.message, 
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        
        response = {
            "response": result.get('response', 'Sorry, I encountered an error processing your message.'),
            "emotion": result.get('emotion'),
            "metadata": result.get('metadata', {})
        }
        
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
        # Get assessment questions based on type
        from src.config.settings import AppConfig
        
        if assessment_type.lower() == "phq9":
            questions = [{"id": i, "question": q} for i, q in enumerate(AppConfig.PHQ9_QUESTIONS)]
        elif assessment_type.lower() == "gad7":
            questions = [{"id": i, "question": q} for i, q in enumerate(AppConfig.GAD7_QUESTIONS)]
        elif assessment_type.lower() == "big_five":
            # Load Big Five questions from data file
            try:
                import json
                with open(AppConfig.DATA_DIR / "personality" / "big_five_questions.json", "r") as f:
                    big_five_data = json.load(f)
                    questions = big_five_data.get("questions", [])
            except FileNotFoundError:
                questions = [{"id": i, "question": f"Sample Big Five question {i+1}"} for i in range(20)]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown assessment type: {assessment_type}")
        
        response = {
            "assessment_type": assessment_type,
            "questions": questions,
            "instructions": f"Please answer all questions for the {assessment_type.upper()} assessment."
        }
        
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
        # Get diagnosis agent from module manager
        module_manager = app_state["app_manager"].module_manager
        diagnosis_agent = module_manager.get_module("diagnosis_agent")
        
        if not diagnosis_agent:
            raise HTTPException(status_code=503, detail="Diagnosis agent not available")
        
        # Process the assessment submission
        result = await diagnosis_agent.process_assessment(
            assessment_type=request.assessment_type,
            responses=request.responses,
            user_id=request.user_id
        )
        
        response = {
            "assessment_type": request.assessment_type,
            "user_id": request.user_id,
            "results": result.get('results', {}),
            "recommendations": result.get('recommendations', []),
            "severity": result.get('severity', 'unknown'),
            "next_steps": result.get('next_steps', [])
        }
        
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
        # Get voice module from module manager
        module_manager = app_state["app_manager"].module_manager
        voice_module = module_manager.get_module("voice")
        
        if not voice_module:
            raise HTTPException(status_code=503, detail="Voice module not available")
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Process the transcription request
        result = await voice_module.transcribe_audio(audio_data)
        
        response = {
            "transcription": result.get('text', ''),
            "confidence": result.get('confidence', 0.0),
            "language": result.get('language', 'en'),
            "duration": result.get('duration', 0.0)
        }
        
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
        # Get voice module from module manager
        module_manager = app_state["app_manager"].module_manager
        voice_module = module_manager.get_module("voice")
        
        if not voice_module:
            raise HTTPException(status_code=503, detail="Voice module not available")
        
        # Process the synthesis request
        result = await voice_module.synthesize_speech(text)
        
        response = {
            "audio_data": result.get('audio_data'),
            "duration": result.get('duration', 0.0),
            "sample_rate": result.get('sample_rate', 22050),
            "format": result.get('format', 'wav')
        }
        
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
        # Get central vector DB module to retrieve user profile
        module_manager = app_state["app_manager"].module_manager
        central_db = module_manager.get_module("central_vector_db")
        
        if not central_db:
            raise HTTPException(status_code=503, detail="Central vector DB not available")
        
        # Process the user profile request
        result = await central_db.get_user_profile(user_id)
        
        response = {
            "user_id": user_id,
            "profile": result.get('profile', {}),
            "preferences": result.get('preferences', {}),
            "assessment_history": result.get('assessment_history', []),
            "conversation_summary": result.get('conversation_summary', {}),
            "last_active": result.get('last_active')
        }
        
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
        # Get central vector DB module to update user profile
        module_manager = app_state["app_manager"].module_manager
        central_db = module_manager.get_module("central_vector_db")
        
        if not central_db:
            raise HTTPException(status_code=503, detail="Central vector DB not available")
        
        # Process the user profile update
        result = await central_db.update_user_profile(
            user_id=request.user_id,
            name=request.name,
            preferences=request.preferences or {},
            metadata=request.metadata or {}
        )
        
        response = {
            "user_id": request.user_id,
            "updated": result.get('success', False),
            "message": result.get('message', 'Profile updated successfully'),
            "profile": result.get('profile', {})
        }
        
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
        # Get therapy resources from knowledge base
        module_manager = app_state["app_manager"].module_manager
        
        # Load therapy resources based on category
        from src.config.settings import AppConfig
        
        if category:
            # Filter resources by category - implement specific filtering logic
            resources = {
                "category": category,
                "resources": [
                    {"type": "article", "title": f"Understanding {category}", "url": "#"},
                    {"type": "exercise", "title": f"Coping strategies for {category}", "description": "..."}
                ]
            }
        else:
            # Return general therapy resources
            resources = {
                "crisis_resources": AppConfig.CRISIS_RESOURCES,
                "categories": ["anxiety", "depression", "stress", "trauma", "relationships"],
                "general_resources": [
                    {"type": "hotline", "name": "National Crisis Hotline", "number": "988"},
                    {"type": "text", "name": "Crisis Text Line", "number": "741741"}
                ]
            }
        
        response = resources
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error getting therapy resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
