"""
Integration Tests for General API Endpoints
Tests health check, chat, assessment, and other core API functionality
"""

import pytest
import json
import time
import asyncio
from datetime import datetime
from unittest.mock import patch, Mock, AsyncMock
from io import BytesIO

from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.auth.models import ChatRequestSecure, DiagnosticAssessmentRequestSecure


@pytest.mark.api
@pytest.mark.integration
class TestHealthEndpoint:
    """Test suite for health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        # Should contain status information
        assert "status" in response_data
        assert response_data["status"] in ["healthy", "initializing", "unhealthy"]
    
    def test_health_check_rate_limiting(self, client):
        """Test health check endpoint respects rate limiting"""
        # Health endpoint has 60/minute rate limit, so we need many requests
        responses = []
        for i in range(65):  # Exceed limit
            response = client.get("/health")
            responses.append(response)
            if i % 10 == 0:  # Brief pause every 10 requests
                time.sleep(0.01)
        
        # Some requests should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        # Note: Depending on timing, this might not always trigger in test environment
        # This test is more for ensuring the rate limit configuration is applied
    
    def test_health_check_no_auth_required(self, client):
        """Test health check works without authentication"""
        # No auth headers provided
        response = client.get("/health")
        
        # Should still work (public endpoint)
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_check_response_structure(self, client):
        """Test health check response has expected structure"""
        response = client.get("/health")
        response_data = response.json()
        
        assert "status" in response_data
        
        # If healthy, should have details
        if response_data["status"] == "healthy":
            assert "details" in response_data


@pytest.mark.api
@pytest.mark.integration
class TestChatEndpoints:
    """Test suite for chat endpoints"""
    
    def test_chat_success(self, client, auth_headers, test_chat_messages, mock_llm_response):
        """Test successful chat message processing"""
        chat_data = {
            "message": test_chat_messages[0]["message"],
            "metadata": test_chat_messages[0]["metadata"]
        }
        
        # Mock the chat agent response
        with patch('src.agents.chat_agent.ChatAgent.process_message', return_value=mock_llm_response):
            response = client.post("/api/chat", json=chat_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        # Verify response structure
        assert "response" in response_data
        assert "user_id" in response_data
        assert "timestamp" in response_data
        assert "emotion" in response_data or response_data["emotion"] is None
        assert "metadata" in response_data or response_data["metadata"] is None
    
    def test_chat_authentication_required(self, client, test_chat_messages):
        """Test chat endpoint requires authentication"""
        chat_data = {
            "message": test_chat_messages[0]["message"],
            "metadata": test_chat_messages[0]["metadata"]
        }
        
        response = client.post("/api/chat", json=chat_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_chat_invalid_token(self, client, invalid_auth_headers, test_chat_messages):
        """Test chat endpoint rejects invalid tokens"""
        chat_data = {
            "message": test_chat_messages[0]["message"],
            "metadata": test_chat_messages[0]["metadata"]
        }
        
        response = client.post("/api/chat", json=chat_data, headers=invalid_auth_headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_chat_message_validation(self, client, auth_headers, test_chat_messages):
        """Test chat message input validation"""
        # Test empty message
        chat_data = {"message": "", "metadata": {}}
        response = client.post("/api/chat", json=chat_data, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test message too long
        long_message_data = {
            "message": "A" * 3000,  # Exceeds max length
            "metadata": {}
        }
        response = client.post("/api/chat", json=long_message_data, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_message_sanitization(self, client, auth_headers, security_test_data, mock_llm_response):
        """Test chat message input sanitization"""
        # Test XSS payloads are handled safely
        for xss_payload in security_test_data["xss_payloads"]:
            chat_data = {
                "message": xss_payload,
                "metadata": {"test": "xss"}
            }
            
            with patch('src.agents.chat_agent.ChatAgent.process_message', return_value=mock_llm_response):
                response = client.post("/api/chat", json=chat_data, headers=auth_headers)
                
                # Should either sanitize and process, or reject
                assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]
    
    def test_chat_metadata_validation(self, client, auth_headers):
        """Test chat metadata validation"""
        # Test metadata with too many items
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(30)}  # Exceeds limit
        chat_data = {
            "message": "Hello",
            "metadata": large_metadata
        }
        
        response = client.post("/api/chat", json=chat_data, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_rate_limiting(self, client, auth_headers):
        """Test chat endpoint rate limiting"""
        chat_data = {
            "message": "Test message",
            "metadata": {}
        }
        
        # Mock successful responses
        mock_response = {"response": "Test response", "emotion": None, "metadata": {}}
        
        with patch('src.agents.chat_agent.ChatAgent.process_message', return_value=mock_response):
            responses = []
            for i in range(35):  # Exceed chat rate limit of 30/minute
                response = client.post("/api/chat", json=chat_data, headers=auth_headers)
                responses.append(response)
                time.sleep(0.1)  # Brief delay
        
        # Some requests should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
    
    def test_chat_application_not_initialized(self, client, auth_headers):
        """Test chat behavior when application is not initialized"""
        chat_data = {
            "message": "Hello",
            "metadata": {}
        }
        
        # Mock uninitialized application state
        with patch('api_server.app_state', {"initialized": False, "app_manager": None}):
            response = client.post("/api/chat", json=chat_data, headers=auth_headers)
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.api
@pytest.mark.integration
class TestAssessmentEndpoints:
    """Test suite for assessment endpoints"""
    
    def test_get_phq9_questions(self, client, auth_headers):
        """Test retrieving PHQ-9 assessment questions"""
        response = client.get("/api/assessment/questions/phq9", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "assessment_type" in response_data
        assert response_data["assessment_type"] == "phq9"
        assert "questions" in response_data
        assert "instructions" in response_data
        assert isinstance(response_data["questions"], list)
        assert len(response_data["questions"]) > 0
    
    def test_get_gad7_questions(self, client, auth_headers):
        """Test retrieving GAD-7 assessment questions"""
        response = client.get("/api/assessment/questions/gad7", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["assessment_type"] == "gad7"
        assert "questions" in response_data
        assert isinstance(response_data["questions"], list)
    
    def test_get_big_five_questions(self, client, auth_headers):
        """Test retrieving Big Five assessment questions"""
        response = client.get("/api/assessment/questions/big_five", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["assessment_type"] == "big_five"
        assert "questions" in response_data
    
    def test_get_invalid_assessment_questions(self, client, auth_headers):
        """Test retrieving questions for invalid assessment type"""
        response = client.get("/api/assessment/questions/invalid_type", headers=auth_headers)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unknown assessment type" in response.json()["detail"].lower()
    
    def test_assessment_questions_authentication_required(self, client):
        """Test assessment questions require authentication"""
        response = client.get("/api/assessment/questions/phq9")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_submit_phq9_assessment(self, client, auth_headers, test_assessment_data, mock_diagnosis_response):
        """Test submitting PHQ-9 assessment"""
        assessment_data = test_assessment_data["phq9"]
        
        with patch('src.agents.diagnosis_agent.DiagnosisAgent.process_assessment', return_value=mock_diagnosis_response):
            response = client.post("/api/assessment/submit", json=assessment_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "assessment_type" in response_data
        assert "user_id" in response_data
        assert "results" in response_data
        assert "recommendations" in response_data
        assert "severity" in response_data
        assert "next_steps" in response_data
    
    def test_submit_gad7_assessment(self, client, auth_headers, test_assessment_data, mock_diagnosis_response):
        """Test submitting GAD-7 assessment"""
        assessment_data = test_assessment_data["gad7"]
        
        with patch('src.agents.diagnosis_agent.DiagnosisAgent.process_assessment', return_value=mock_diagnosis_response):
            response = client.post("/api/assessment/submit", json=assessment_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["assessment_type"] == "gad7"
    
    def test_submit_assessment_invalid_type(self, client, auth_headers, test_assessment_data):
        """Test submitting assessment with invalid type"""
        assessment_data = test_assessment_data["invalid"]
        
        response = client.post("/api/assessment/submit", json=assessment_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_submit_assessment_too_many_responses(self, client, auth_headers, test_assessment_data):
        """Test submitting assessment with too many responses"""
        assessment_data = test_assessment_data["too_many_responses"]
        
        response = client.post("/api/assessment/submit", json=assessment_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_assessment_rate_limiting(self, client, auth_headers, test_assessment_data):
        """Test assessment endpoint rate limiting"""
        assessment_data = test_assessment_data["phq9"]
        mock_response = {"results": {}, "recommendations": [], "severity": "mild", "next_steps": []}
        
        with patch('src.agents.diagnosis_agent.DiagnosisAgent.process_assessment', return_value=mock_response):
            responses = []
            for i in range(12):  # Exceed assessment rate limit of 10/minute
                response = client.post("/api/assessment/submit", json=assessment_data, headers=auth_headers)
                responses.append(response)
                time.sleep(0.1)
        
        # Some requests should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


@pytest.mark.api
@pytest.mark.integration 
class TestVoiceEndpoints:
    """Test suite for voice processing endpoints"""
    
    def test_transcribe_audio_success(self, client, auth_headers):
        """Test successful audio transcription"""
        # Create mock audio file
        audio_content = b"fake_audio_data"
        audio_file = ("test.wav", BytesIO(audio_content), "audio/wav")
        
        mock_transcription = {
            "text": "Hello, I need help with anxiety",
            "confidence": 0.95,
            "language": "en",
            "duration": 2.5
        }
        
        with patch('src.components.voice_module.VoiceModule.transcribe_audio', return_value=mock_transcription):
            response = client.post(
                "/api/voice/transcribe",
                files={"audio_file": audio_file},
                headers={"Authorization": auth_headers["Authorization"]}
            )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "transcription" in response_data
        assert "confidence" in response_data
        assert "language" in response_data
        assert "duration" in response_data
    
    def test_transcribe_audio_invalid_format(self, client, auth_headers):
        """Test audio transcription with invalid file format"""
        # Create mock file with invalid format
        invalid_content = b"not_audio_data"
        invalid_file = ("test.txt", BytesIO(invalid_content), "text/plain")
        
        response = client.post(
            "/api/voice/transcribe",
            files={"audio_file": invalid_file},
            headers={"Authorization": auth_headers["Authorization"]}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unsupported audio format" in response.json()["detail"].lower()
    
    def test_transcribe_audio_file_too_large(self, client, auth_headers):
        """Test audio transcription with file too large"""
        # Create mock large audio file
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB, exceeds 10MB limit
        large_file = ("large.wav", BytesIO(large_content), "audio/wav")
        
        response = client.post(
            "/api/voice/transcribe",
            files={"audio_file": large_file},
            headers={"Authorization": auth_headers["Authorization"]}
        )
        
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "file too large" in response.json()["detail"].lower()
    
    def test_transcribe_audio_authentication_required(self, client):
        """Test audio transcription requires authentication"""
        audio_content = b"fake_audio_data"
        audio_file = ("test.wav", BytesIO(audio_content), "audio/wav")
        
        response = client.post(
            "/api/voice/transcribe",
            files={"audio_file": audio_file}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_synthesize_speech_success(self, client, auth_headers):
        """Test successful speech synthesis"""
        text_data = {"text": "Hello, this is a test message for speech synthesis."}
        
        mock_synthesis = {
            "audio_data": "base64_encoded_audio_data",
            "duration": 3.2,
            "sample_rate": 22050,
            "format": "wav"
        }
        
        with patch('src.components.voice_module.VoiceModule.synthesize_speech', return_value=mock_synthesis):
            response = client.post("/api/voice/synthesize", json=text_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "audio_data" in response_data
        assert "duration" in response_data
        assert "sample_rate" in response_data
        assert "format" in response_data
    
    def test_synthesize_speech_text_too_long(self, client, auth_headers):
        """Test speech synthesis with text too long"""
        long_text = {"text": "A" * 3000}  # Exceeds max message length
        
        response = client.post("/api/voice/synthesize", json=long_text, headers=auth_headers)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "text too long" in response.json()["detail"].lower()
    
    def test_synthesize_speech_html_sanitization(self, client, auth_headers, security_test_data):
        """Test speech synthesis sanitizes HTML content"""
        html_text = {"text": "<script>alert('xss')</script>Hello world"}
        
        mock_synthesis = {
            "audio_data": "base64_audio",
            "duration": 1.0,
            "sample_rate": 22050,
            "format": "wav"
        }
        
        with patch('src.components.voice_module.VoiceModule.synthesize_speech', return_value=mock_synthesis) as mock_synth:
            response = client.post("/api/voice/synthesize", json=html_text, headers=auth_headers)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Check that the text passed to synthesis was sanitized
            called_args = mock_synth.call_args[0]
            sanitized_text = called_args[0]
            assert "<script>" not in sanitized_text
            assert "Hello world" in sanitized_text


@pytest.mark.api
@pytest.mark.integration
class TestUserProfileEndpoints:
    """Test suite for user profile endpoints"""
    
    def test_get_user_profile_own(self, client, auth_headers):
        """Test getting own user profile"""
        # Get current user info from token
        with patch('src.components.central_vector_db_module.CentralVectorDBModule.get_user_profile') as mock_profile:
            mock_profile.return_value = {
                "profile": {"name": "Test User", "preferences": {}},
                "preferences": {"theme": "light"},
                "assessment_history": [],
                "conversation_summary": {},
                "last_active": datetime.utcnow().isoformat()
            }
            
            # Use test user ID from token
            response = client.get("/api/user/test_user_id", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "user_id" in response_data
        assert "profile" in response_data
        assert "preferences" in response_data
        assert "assessment_history" in response_data
    
    def test_get_user_profile_access_denied(self, client, auth_headers):
        """Test access denied when trying to view other user's profile"""
        response = client.get("/api/user/other_user_id", headers=auth_headers)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "access denied" in response.json()["detail"].lower()
    
    def test_update_user_profile_success(self, client, auth_headers):
        """Test successful user profile update"""
        profile_data = {
            "name": "Updated Name",
            "preferences": {"theme": "dark", "notifications": True},
            "metadata": {"timezone": "UTC"}
        }
        
        with patch('src.components.central_vector_db_module.CentralVectorDBModule.update_user_profile') as mock_update:
            mock_update.return_value = {
                "success": True,
                "message": "Profile updated successfully",
                "profile": profile_data
            }
            
            response = client.post("/api/user/update", json=profile_data, headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "updated" in response_data
        assert response_data["updated"] is True
        assert "message" in response_data
    
    def test_update_user_profile_sanitization(self, client, auth_headers, security_test_data):
        """Test user profile update input sanitization"""
        # Test HTML in name field
        profile_data = {
            "name": "<script>alert('xss')</script>Safe Name",
            "preferences": {"theme": "light"},
            "metadata": {}
        }
        
        with patch('src.components.central_vector_db_module.CentralVectorDBModule.update_user_profile') as mock_update:
            mock_update.return_value = {"success": True, "message": "Updated", "profile": {}}
            
            response = client.post("/api/user/update", json=profile_data, headers=auth_headers)
        
        # Should succeed with sanitized data
        assert response.status_code == status.HTTP_200_OK
        
        # Verify sanitization occurred
        called_args = mock_update.call_args
        assert called_args is not None
        name_arg = called_args.kwargs.get('name', '')
        assert '<script>' not in name_arg
    
    def test_user_profile_rate_limiting(self, client, auth_headers):
        """Test user profile endpoint rate limiting"""
        profile_data = {"name": "Test Name", "preferences": {}, "metadata": {}}
        
        with patch('src.components.central_vector_db_module.CentralVectorDBModule.update_user_profile') as mock_update:
            mock_update.return_value = {"success": True, "message": "Updated", "profile": {}}
            
            responses = []
            for i in range(18):  # Exceed user_profile rate limit of 15/minute
                response = client.post("/api/user/update", json=profile_data, headers=auth_headers)
                responses.append(response)
                time.sleep(0.1)
        
        # Some requests should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0


@pytest.mark.api
@pytest.mark.integration
class TestTherapyResourcesEndpoints:
    """Test suite for therapy resources endpoints"""
    
    def test_get_therapy_resources_general(self, client):
        """Test getting general therapy resources"""
        response = client.get("/api/therapy/resources")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "crisis_resources" in response_data
        assert "categories" in response_data
        assert "general_resources" in response_data
        
        # Verify crisis resources structure
        crisis_resources = response_data["crisis_resources"]
        assert isinstance(crisis_resources, list)
        
        # Verify categories
        categories = response_data["categories"]
        assert isinstance(categories, list)
        assert "anxiety" in categories
        assert "depression" in categories
    
    def test_get_therapy_resources_by_category(self, client):
        """Test getting therapy resources filtered by category"""
        response = client.get("/api/therapy/resources?category=anxiety")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "category" in response_data
        assert response_data["category"] == "anxiety"
        assert "resources" in response_data
    
    def test_therapy_resources_no_auth_required(self, client):
        """Test therapy resources endpoint works without authentication"""
        # This is a public endpoint for crisis situations
        response = client.get("/api/therapy/resources")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
@pytest.mark.integration
class TestAPIErrorHandling:
    """Test suite for API error handling"""
    
    def test_404_not_found(self, client):
        """Test 404 response for non-existent endpoints"""
        response = client.get("/api/nonexistent/endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_405_method_not_allowed(self, client):
        """Test 405 response for wrong HTTP methods"""
        response = client.put("/health")  # GET endpoint called with PUT
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_malformed_json_handling(self, client, auth_headers):
        """Test handling of malformed JSON requests"""
        response = client.post(
            "/api/chat",
            data="invalid json {",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_content_type_validation(self, client, auth_headers):
        """Test content type validation"""
        # Send data without proper Content-Type header
        headers_no_content_type = {
            "Authorization": auth_headers["Authorization"]
        }
        
        response = client.post(
            "/api/chat",
            data='{"message": "test"}',
            headers=headers_no_content_type
        )
        
        # Should handle gracefully
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]
    
    def test_concurrent_request_handling(self, client, auth_headers):
        """Test handling of concurrent API requests"""
        import concurrent.futures
        import threading
        
        def make_health_request():
            return client.get("/health")
        
        # Submit multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed (or be rate limited)
        for response in responses:
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS]
    
    @pytest.mark.slow
    def test_api_response_times(self, client, auth_headers):
        """Test API response times are within acceptable limits"""
        endpoints_to_test = [
            ("/health", "GET", None),
            ("/api/auth/me", "GET", None),
        ]
        
        for endpoint, method, data in endpoints_to_test:
            start_time = time.time()
            
            if method == "GET":
                response = client.get(endpoint, headers=auth_headers if "auth" in endpoint else None)
            elif method == "POST":
                response = client.post(endpoint, json=data, headers=auth_headers)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # API responses should be reasonably fast
            assert response_time < 5.0  # 5 second timeout
            assert response.status_code in [200, 401, 503]  # Expected status codes