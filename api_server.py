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
from fastapi import FastAPI, Depends, HTTPException, Body, File, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

# Import application components
from src.main import Application
from src.config.settings import AppConfig
from src.config.security import SecurityConfig
from src.auth.models import (
    UserCreate, UserLogin, UserResponse, Token, TokenData, RefreshTokenRequest,
    PasswordReset, PasswordResetConfirm, PasswordChange, UserUpdate,
    ChatRequestSecure, DiagnosticAssessmentRequestSecure, UserProfileRequestSecure
)
from src.auth.jwt_utils import jwt_manager
from src.auth.dependencies import (
    get_current_user, get_current_active_user, get_optional_user,
    require_admin, require_therapist_or_admin, require_authenticated,
    require_chat_access, require_assessment_access
)
from src.middleware.security import (
    SecurityHeadersMiddleware, RequestLoggingMiddleware, IPFilterMiddleware,
    ContentTypeValidationMiddleware, CSRFProtectionMiddleware, limiter, rate_limit_handler
)

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

# Create FastAPI app with security configuration
app = FastAPI(
    title="Solace-AI API",
    description="Secure API for the Solace-AI Mental Health Support application",
    version="1.0.0",
    docs_url="/docs" if SecurityConfig.is_development() else None,
    redoc_url="/redoc" if SecurityConfig.is_development() else None,
    openapi_url="/openapi.json" if SecurityConfig.is_development() else None
)

# Add security middleware in correct order
# 1. Security headers (first)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Request logging for audit
app.add_middleware(RequestLoggingMiddleware)

# 3. Content type validation
app.add_middleware(ContentTypeValidationMiddleware)

# 4. IP filtering for admin endpoints
app.add_middleware(IPFilterMiddleware)

# 5. CSRF protection (SEC-006)
# Enabled in production; JWT-protected requests bypass CSRF check
app.add_middleware(CSRFProtectionMiddleware, enabled=not SecurityConfig.is_development())

# 6. Rate limiting
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# 7. CORS with specific origins (CRITICAL FIX)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SecurityConfig.ALLOWED_ORIGINS,
    allow_credentials=False,  # Disabled for security unless specifically needed
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With"
    ],
    expose_headers=["X-Total-Count", "X-Page-Count"],
    max_age=600  # Cache preflight for 10 minutes
)

# Application state
app_state = {
    "initialized": False,
    "app_manager": None
}

# Enhanced secure request/response models
class ChatResponse(BaseModel):
    response: str
    emotion: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None

class AssessmentQuestionResponse(BaseModel):
    question_id: str
    response: Any

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    request_id: Optional[str] = None

# Supervision API models
class SupervisionStatusResponse(BaseModel):
    supervision_enabled: bool
    supervisor_agent_active: bool
    metrics_collector_active: bool
    audit_trail_active: bool
    active_workflows: int
    total_agents: int
    status_timestamp: str

class SupervisionSummaryResponse(BaseModel):
    supervision_status: str
    time_window_hours: int
    timestamp: str
    real_time_metrics: Optional[Dict[str, Any]] = None
    supervisor_metrics: Optional[Dict[str, Any]] = None
    audit_summary: Optional[Dict[str, Any]] = None

class AgentQualityReportResponse(BaseModel):
    agent_name: str
    time_window: str
    performance_summary: Dict[str, Any]
    quality_indicators: Dict[str, Any]
    top_issues: List[str]
    recommendations: List[str]

class SessionAnalysisResponse(BaseModel):
    session_id: str
    analysis_timestamp: str
    audit_events_count: int
    event_summary: Dict[str, int]
    critical_issues: int
    critical_details: List[Dict[str, Any]]
    supervisor_summary: Optional[Dict[str, Any]] = None

class ComplianceReportRequest(BaseModel):
    compliance_standard: str
    start_date: str
    end_date: str

class ComplianceReportResponse(BaseModel):
    compliance_report: Dict[str, Any]
    export_path: str
    generated_timestamp: str

class SupervisionConfigRequest(BaseModel):
    config: Dict[str, Any]

class SupervisionConfigResponse(BaseModel):
    configured: List[str]
    errors: List[str]

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

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(SecurityConfig.get_rate_limit("auth"))
async def register_user(request: Request, user_data: UserCreate):
    """Register a new user with secure password handling"""
    try:
        # In production, integrate with your user database
        # This is a placeholder implementation
        
        # Register user with secure validation
        from src.services.user_service import user_service
        
        new_user = user_service.register_user(user_data)
        
        if not new_user:
            logger.warning(f"Registration failed for user: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed - user may already exist or password requirements not met"
            )
        
        logger.info(f"New user registered: {new_user.username}")
        return new_user
        
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed"
        )

@app.post("/api/auth/login", response_model=Token)
@limiter.limit(SecurityConfig.get_rate_limit("auth"))
async def login_user(request: Request, user_credentials: UserLogin):
    """Authenticate user and return JWT tokens"""
    try:
        # In production, validate against your user database
        # This is a placeholder implementation
        from datetime import datetime
        import uuid
        
        # Authenticate user with secure validation
        from src.services.user_service import user_service
        
        authenticated_user = user_service.authenticate_user(
            user_credentials.username, 
            user_credentials.password
        )
        
        if not authenticated_user:
            logger.warning(f"Authentication failed for user: {user_credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create tokens for authenticated user
        token_data = jwt_manager.create_user_tokens(authenticated_user)
        
        logger.info(f"User logged in: {user_credentials.username}")
        return Token(**token_data)
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@app.post("/api/auth/refresh", response_model=Token)
@limiter.limit(SecurityConfig.get_rate_limit("auth"))
async def refresh_token(request: Request, refresh_request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        access_token, refresh_token = jwt_manager.refresh_access_token(refresh_request.refresh_token)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@app.post("/api/auth/logout")
@limiter.limit(SecurityConfig.get_rate_limit("auth"))
async def logout_user(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    """Logout user by blacklisting current tokens"""
    try:
        # In a full implementation, you would blacklist both access and refresh tokens
        logger.info(f"User logged out: {current_user.username}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get current user profile information"""
    try:
        # In production, fetch from database using current_user.sub
        from datetime import datetime
        
        user_profile = UserResponse(
            id=current_user.sub,
            username=current_user.username or "unknown",
            email=f"{current_user.username}@example.com",
            full_name="User Name",
            role=current_user.role,
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        
        return user_profile
        
    except Exception as e:
        logger.error(f"Get user profile failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

# Health check endpoint (public)
@app.get("/health")
@limiter.limit(SecurityConfig.get_rate_limit("health"))
async def health_check(request: Request):
    """Check the health of the API server"""
    if not app_state["initialized"]:
        return {"status": "initializing"}
    
    if app_state["app_manager"]:
        health_status = await app_state["app_manager"].health_check()
        return {"status": "healthy", "details": health_status}
    
    return {"status": "unhealthy", "details": "Application manager not available"}

# Chat endpoint - REQUIRES AUTHENTICATION
@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit(SecurityConfig.get_rate_limit("chat"))
async def chat(
    request: Request,
    chat_request: ChatRequestSecure,
    current_user: TokenData = Depends(require_chat_access)
):
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
        
        # Use authenticated user's ID instead of request user_id
        user_id = current_user.sub
        
        # Process the chat message with security context
        result = await chat_agent.process_message(
            chat_request.message,
            user_id=user_id,
            metadata={
                **(chat_request.metadata or {}),
                "authenticated_user": current_user.username,
                "user_role": current_user.role
            }
        )
        
        from datetime import datetime
        
        response = {
            "response": result.get('response', 'Sorry, I encountered an error processing your message.'),
            "emotion": result.get('emotion'),
            "metadata": result.get('metadata', {}),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Assessment endpoints - REQUIRE AUTHENTICATION
@app.get("/api/assessment/questions/{assessment_type}")
@limiter.limit(SecurityConfig.get_rate_limit("assessment"))
async def get_assessment_questions(
    request: Request,
    assessment_type: str,
    current_user: TokenData = Depends(require_assessment_access)
):
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
@limiter.limit(SecurityConfig.get_rate_limit("assessment"))
async def submit_assessment(
    request: Request,
    assessment_request: DiagnosticAssessmentRequestSecure,
    current_user: TokenData = Depends(require_assessment_access)
):
    """Submit a completed assessment for analysis"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get diagnosis agent from module manager
        module_manager = app_state["app_manager"].module_manager
        diagnosis_agent = module_manager.get_module("diagnosis_agent")
        
        if not diagnosis_agent:
            raise HTTPException(status_code=503, detail="Diagnosis agent not available")
        
        # Use authenticated user's ID for security
        user_id = current_user.sub
        
        # Process the assessment submission with authentication context
        result = await diagnosis_agent.process_assessment(
            assessment_type=assessment_request.assessment_type,
            responses=assessment_request.responses,
            user_id=user_id
        )
        
        response = {
            "assessment_type": assessment_request.assessment_type,
            "user_id": user_id,
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

# Voice endpoints - REQUIRE AUTHENTICATION
@app.post("/api/voice/transcribe")
@limiter.limit(SecurityConfig.get_rate_limit("upload"))
async def transcribe_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_active_user)
):
    """Transcribe audio to text"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get voice module from module manager
        module_manager = app_state["app_manager"].module_manager
        voice_module = module_manager.get_module("voice")
        
        if not voice_module:
            raise HTTPException(status_code=503, detail="Voice module not available")
        
        # Validate file type and size for security
        if audio_file.content_type not in SecurityConfig.INPUT_LIMITS["allowed_audio_types"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format: {audio_file.content_type}"
            )
        
        # Check file size (limit to prevent DoS)
        max_size = SecurityConfig.INPUT_LIMITS["max_file_size_mb"] * 1024 * 1024
        if audio_file.size and audio_file.size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {SecurityConfig.INPUT_LIMITS['max_file_size_mb']}MB"
            )
        
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Log audio transcription request for audit
        logger.info(
            f"Audio transcription request from user: {current_user.username}",
            extra={
                "event_type": "audio_transcription",
                "user_id": current_user.sub,
                "username": current_user.username,
                "file_type": audio_file.content_type,
                "file_size": len(audio_data)
            }
        )
        
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
@limiter.limit(SecurityConfig.get_rate_limit("upload"))
async def synthesize_speech(
    request: Request,
    text: str = Body(..., embed=True),
    current_user: TokenData = Depends(get_current_active_user)
):
    """Synthesize text to speech"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get voice module from module manager
        module_manager = app_state["app_manager"].module_manager
        voice_module = module_manager.get_module("voice")
        
        if not voice_module:
            raise HTTPException(status_code=503, detail="Voice module not available")
        
        # Validate text length for security
        if len(text) > SecurityConfig.INPUT_LIMITS["max_message_length"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text too long. Maximum length: {SecurityConfig.INPUT_LIMITS['max_message_length']} characters"
            )
        
        # Basic content sanitization
        import re
        sanitized_text = re.sub(r'<[^>]+>', '', text).strip()
        
        # Process the synthesis request
        result = await voice_module.synthesize_speech(sanitized_text)
        
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

# User management endpoints - REQUIRE AUTHENTICATION
@app.get("/api/user/{user_id}")
@limiter.limit(SecurityConfig.get_rate_limit("user_profile"))
async def get_user_profile(
    request: Request,
    user_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get a user's profile"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get central vector DB module to retrieve user profile
        module_manager = app_state["app_manager"].module_manager
        central_db = module_manager.get_module("central_vector_db")
        
        if not central_db:
            raise HTTPException(status_code=503, detail="Central vector DB not available")
        
        # Security check: users can only access their own profile unless admin/therapist
        if user_id != current_user.sub and current_user.role not in ["admin", "therapist"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Can only access your own profile"
            )
        
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
@limiter.limit(SecurityConfig.get_rate_limit("user_profile"))
async def update_user_profile(
    request: Request,
    profile_request: UserProfileRequestSecure,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Update a user's profile"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get central vector DB module to update user profile
        module_manager = app_state["app_manager"].module_manager
        central_db = module_manager.get_module("central_vector_db")
        
        if not central_db:
            raise HTTPException(status_code=503, detail="Central vector DB not available")
        
        # Use authenticated user's ID for security
        user_id = current_user.sub
        
        # Security check: prevent updating other users' profiles
        if profile_request.user_id and profile_request.user_id != user_id:
            if current_user.role not in ["admin", "therapist"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Can only update your own profile"
                )
            user_id = profile_request.user_id
        
        # Process the user profile update
        result = await central_db.update_user_profile(
            user_id=user_id,
            name=profile_request.name,
            preferences=profile_request.preferences or {},
            metadata=profile_request.metadata or {}
        )
        
        response = {
            "user_id": user_id,
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

# Supervision and Quality Assurance Endpoints
@app.get("/api/supervision/status", response_model=SupervisionStatusResponse)
async def get_supervision_status():
    """Get current supervision system status"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Get supervision status
        status = agent_orchestrator.get_supervision_status()
        
        return SupervisionStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting supervision status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supervision/summary", response_model=SupervisionSummaryResponse)
async def get_supervision_summary(time_window_hours: int = 24):
    """Get comprehensive supervision summary"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Get supervision summary
        summary = await agent_orchestrator.get_supervision_summary(time_window_hours)
        
        if "error" in summary:
            raise HTTPException(status_code=400, detail=summary["error"])
        
        return SupervisionSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting supervision summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supervision/agent-quality/{agent_name}", response_model=AgentQualityReportResponse)
async def get_agent_quality_report(agent_name: str):
    """Get quality report for specific agent"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Get agent quality report
        report = await agent_orchestrator.get_agent_quality_report(agent_name)
        
        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])
        
        return AgentQualityReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Error getting agent quality report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supervision/session-analysis/{session_id}", response_model=SessionAnalysisResponse)
async def get_session_analysis(session_id: str):
    """Get comprehensive analysis for a specific session"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Get session analysis
        analysis = await agent_orchestrator.get_session_analysis(session_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return SessionAnalysisResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Error getting session analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supervision/compliance-report", response_model=ComplianceReportResponse)
async def export_compliance_report(request: ComplianceReportRequest):
    """Export compliance report for regulatory purposes"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Export compliance report
        result = await agent_orchestrator.export_compliance_report(
            compliance_standard=request.compliance_standard,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return ComplianceReportResponse(**result)
        
    except Exception as e:
        logger.error(f"Error exporting compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supervision/configure", response_model=SupervisionConfigResponse)
async def configure_supervision(request: SupervisionConfigRequest):
    """Configure supervision system parameters"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Configure supervision system
        result = await agent_orchestrator.configure_supervision(request.config)
        
        return SupervisionConfigResponse(**result)
        
    except Exception as e:
        logger.error(f"Error configuring supervision system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supervision/metrics/export")
async def export_supervision_metrics(format: str = "json", time_window_hours: int = 24):
    """Export supervision metrics data"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Check if supervision is enabled
        if not agent_orchestrator.supervision_enabled or not agent_orchestrator.metrics_collector:
            raise HTTPException(status_code=400, detail="Supervision metrics not available")
        
        # Export metrics
        from src.monitoring.supervisor_metrics import MetricsExporter
        from datetime import datetime, timedelta
        import tempfile
        
        exporter = MetricsExporter(agent_orchestrator.metrics_collector)
        
        # Create temporary export file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as temp_file:
            export_path = temp_file.name
        
        if format.lower() == "json":
            time_window = timedelta(hours=time_window_hours)
            exporter.export_to_json(export_path, time_window)
        elif format.lower() == "csv":
            time_window = timedelta(hours=time_window_hours)
            exporter.export_to_csv(export_path, time_window=time_window)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Read exported data
        with open(export_path, 'r') as f:
            exported_data = f.read()
        
        # Clean up temp file
        import os
        os.unlink(export_path)
        
        # Return appropriate response
        from fastapi.responses import PlainTextResponse
        
        if format.lower() == "json":
            return JSONResponse(content=json.loads(exported_data))
        else:
            return PlainTextResponse(content=exported_data, media_type="text/csv")
        
    except Exception as e:
        logger.error(f"Error exporting supervision metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supervision/alerts")
async def get_active_alerts():
    """Get all active supervision alerts"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Check if supervision is enabled
        if not agent_orchestrator.supervision_enabled or not agent_orchestrator.metrics_collector:
            return {"alerts": [], "count": 0}
        
        # Get active alerts
        alerts = agent_orchestrator.metrics_collector.get_active_alerts()
        
        # Convert alerts to serializable format
        serializable_alerts = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "title": alert.title,
                "description": alert.description,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            if alert.resolution_timestamp:
                alert_dict["resolution_timestamp"] = alert.resolution_timestamp.isoformat()
            
            serializable_alerts.append(alert_dict)
        
        return {
            "alerts": serializable_alerts,
            "count": len(serializable_alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supervision/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve a specific alert"""
    if not app_state["initialized"] or not app_state["app_manager"]:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    
    try:
        # Get agent orchestrator from module manager
        module_manager = app_state["app_manager"].module_manager
        agent_orchestrator = module_manager.get_module("agent_orchestrator")
        
        if not agent_orchestrator:
            raise HTTPException(status_code=503, detail="Agent orchestrator not available")
        
        # Check if supervision is enabled
        if not agent_orchestrator.supervision_enabled or not agent_orchestrator.metrics_collector:
            raise HTTPException(status_code=400, detail="Supervision metrics not available")
        
        # Resolve alert
        agent_orchestrator.metrics_collector.resolve_alert(alert_id)
        
        return {
            "alert_id": alert_id,
            "resolved": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
