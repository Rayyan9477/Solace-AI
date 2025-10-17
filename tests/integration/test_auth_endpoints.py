"""
Integration Tests for Authentication API Endpoints
Tests authentication flows, JWT token handling, and security mechanisms
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.auth.models import UserCreate, UserLogin, Token, UserResponse
from src.config.security import SecurityConfig


@pytest.mark.auth
@pytest.mark.integration
class TestAuthEndpoints:
    """Test suite for authentication endpoints"""
    
    def test_user_registration_success(self, client, test_users):
        """Test successful user registration"""
        user_data = test_users["valid_user"]
        
        response = client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        
        # Verify response structure
        assert "id" in response_data
        assert response_data["username"] == user_data["username"]
        assert response_data["email"] == user_data["email"]
        assert response_data["full_name"] == user_data["full_name"]
        assert response_data["role"] == user_data["role"]
        assert response_data["is_active"] is True
        assert "created_at" in response_data
        
        # Ensure password is not returned
        assert "password" not in response_data
        assert "password_hash" not in response_data
    
    def test_user_registration_weak_password(self, client, test_users):
        """Test registration fails with weak password"""
        user_data = test_users["invalid_password_user"]
        
        response = client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in response.json()["detail"].lower()
    
    def test_user_registration_invalid_username(self, client, test_users):
        """Test registration fails with invalid username format"""
        user_data = test_users["invalid_username_user"]
        
        response = client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        errors = response.json()["detail"]
        username_error = next((e for e in errors if e["loc"] == ["body", "username"]), None)
        assert username_error is not None
    
    def test_user_registration_duplicate_username(self, client, test_users):
        """Test registration fails with duplicate username"""
        user_data = test_users["valid_user"]
        
        # First registration
        response1 = client.post("/api/auth/register", json=user_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Duplicate registration
        response2 = client.post("/api/auth/register", json=user_data)
        assert response2.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exist" in response2.json()["detail"].lower()
    
    def test_user_registration_missing_fields(self, client):
        """Test registration fails with missing required fields"""
        incomplete_data = {
            "username": "testuser",
            # Missing email, password
        }
        
        response = client.post("/api/auth/register", json=incomplete_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        errors = response.json()["detail"]
        
        # Check for email and password errors
        field_errors = [error["loc"][1] for error in errors]
        assert "email" in field_errors
        assert "password" in field_errors
    
    def test_user_registration_invalid_email(self, client):
        """Test registration fails with invalid email format"""
        user_data = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "SecureTest123!",
            "full_name": "Test User",
            "role": "user"
        }
        
        response = client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        errors = response.json()["detail"]
        email_error = next((e for e in errors if e["loc"] == ["body", "email"]), None)
        assert email_error is not None
    
    def test_user_login_success(self, client):
        """Test successful user login with default users"""
        login_data = {
            "username": "demo",
            "password": "DemoUser123!"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        # Verify token structure
        assert "access_token" in response_data
        assert "refresh_token" in response_data
        assert response_data["token_type"] == "bearer"
        assert "expires_in" in response_data
        assert "user" in response_data
        
        # Verify user data in token response
        user_data = response_data["user"]
        assert user_data["username"] == "demo"
        assert user_data["is_active"] is True
        assert "password" not in user_data
    
    def test_user_login_invalid_credentials(self, client):
        """Test login fails with invalid credentials"""
        login_data = {
            "username": "demo",
            "password": "wrong_password"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid" in response.json()["detail"].lower()
    
    def test_user_login_nonexistent_user(self, client):
        """Test login fails for nonexistent user"""
        login_data = {
            "username": "nonexistent_user",
            "password": "any_password"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid" in response.json()["detail"].lower()
    
    def test_user_login_missing_credentials(self, client):
        """Test login fails with missing credentials"""
        response = client.post("/api/auth/login", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        errors = response.json()["detail"]
        field_errors = [error["loc"][1] for error in errors]
        assert "username" in field_errors
        assert "password" in field_errors
    
    def test_user_login_with_email(self, client):
        """Test login with email instead of username"""
        login_data = {
            "username": "admin@solace-ai.com",  # Using email
            "password": "SecureAdmin123!"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["user"]["username"] == "admin"
    
    def test_token_refresh_success(self, client):
        """Test successful token refresh"""
        # First login to get tokens
        login_data = {
            "username": "demo",
            "password": "DemoUser123!"
        }
        login_response = client.post("/api/auth/login", json=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Use refresh token to get new access token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/api/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "access_token" in response_data
        assert "refresh_token" in response_data
        assert response_data["token_type"] == "bearer"
        assert "expires_in" in response_data
    
    def test_token_refresh_invalid_token(self, client):
        """Test token refresh fails with invalid refresh token"""
        refresh_data = {"refresh_token": "invalid_refresh_token"}
        response = client.post("/api/auth/refresh", json=refresh_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid" in response.json()["detail"].lower()
    
    def test_token_refresh_missing_token(self, client):
        """Test token refresh fails with missing refresh token"""
        response = client.post("/api/auth/refresh", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        errors = response.json()["detail"]
        refresh_token_error = next((e for e in errors if e["loc"] == ["body", "refresh_token"]), None)
        assert refresh_token_error is not None
    
    def test_user_logout_success(self, client, auth_headers):
        """Test successful user logout"""
        response = client.post("/api/auth/logout", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "successfully logged out" in response_data["message"].lower()
    
    def test_user_logout_no_auth_header(self, client):
        """Test logout fails without authentication header"""
        response = client.post("/api/auth/logout")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_user_logout_invalid_token(self, client, invalid_auth_headers):
        """Test logout fails with invalid token"""
        response = client.post("/api/auth/logout", headers=invalid_auth_headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_profile_success(self, client, auth_headers):
        """Test getting current user profile with valid token"""
        response = client.get("/api/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "id" in response_data
        assert "username" in response_data
        assert "email" in response_data
        assert "role" in response_data
        assert "is_active" in response_data
        assert "created_at" in response_data
        
        # Sensitive data should not be included
        assert "password" not in response_data
        assert "password_hash" not in response_data
    
    def test_get_current_user_profile_no_auth(self, client):
        """Test getting user profile fails without authentication"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_profile_invalid_token(self, client, invalid_auth_headers):
        """Test getting user profile fails with invalid token"""
        response = client.get("/api/auth/me", headers=invalid_auth_headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_jwt_token_expiration(self, client, expired_jwt_token):
        """Test expired JWT token is rejected"""
        headers = {
            "Authorization": f"Bearer {expired_jwt_token}",
            "Content-Type": "application/json"
        }
        
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_authentication_rate_limiting(self, client):
        """Test rate limiting on authentication endpoints"""
        login_data = {
            "username": "demo",
            "password": "wrong_password"  # Intentionally wrong to avoid lockout
        }
        
        # Make requests up to the rate limit
        responses = []
        for i in range(7):  # Exceed auth rate limit of 5/minute
            response = client.post("/api/auth/login", json=login_data)
            responses.append(response)
            time.sleep(0.1)  # Small delay between requests
        
        # At least one response should be rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
    
    def test_account_lockout_mechanism(self, client, test_users):
        """Test account lockout after failed login attempts"""
        # First register a test user
        user_data = test_users["valid_user"].copy()
        user_data["username"] = "lockout_test"
        client.post("/api/auth/register", json=user_data)
        
        login_data = {
            "username": "lockout_test",
            "password": "wrong_password"
        }
        
        # Make multiple failed login attempts
        for i in range(6):  # Exceed lockout threshold of 5
            response = client.post("/api/auth/login", json=login_data)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            time.sleep(0.1)  # Brief delay
        
        # Even correct password should fail after lockout
        correct_login_data = {
            "username": "lockout_test",
            "password": user_data["password"]
        }
        response = client.post("/api/auth/login", json=correct_login_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_registration_input_validation(self, client):
        """Test input validation on registration endpoint"""
        invalid_data_sets = [
            # Username too short
            {
                "username": "ab",
                "email": "test@example.com",
                "password": "SecureTest123!",
                "full_name": "Test User",
                "role": "user"
            },
            # Username too long
            {
                "username": "a" * 100,
                "email": "test@example.com", 
                "password": "SecureTest123!",
                "full_name": "Test User",
                "role": "user"
            },
            # Full name too long
            {
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecureTest123!",
                "full_name": "a" * 200,
                "role": "user"
            },
            # Invalid role
            {
                "username": "testuser",
                "email": "test@example.com",
                "password": "SecureTest123!",
                "full_name": "Test User",
                "role": "invalid_role"
            }
        ]
        
        for invalid_data in invalid_data_sets:
            response = client.post("/api/auth/register", json=invalid_data)
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_login_input_validation(self, client):
        """Test input validation on login endpoint"""
        invalid_login_sets = [
            # Empty username
            {"username": "", "password": "any_password"},
            # Username too long
            {"username": "a" * 100, "password": "any_password"},
            # Empty password
            {"username": "demo", "password": ""},
            # Password too long
            {"username": "demo", "password": "a" * 200}
        ]
        
        for invalid_data in invalid_login_sets:
            response = client.post("/api/auth/login", json=invalid_data)
            assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_malicious_input_handling(self, client, security_test_data):
        """Test handling of malicious input in authentication"""
        # Test SQL injection attempts
        for sql_payload in security_test_data["sql_injection"]:
            login_data = {
                "username": sql_payload,
                "password": "any_password"
            }
            response = client.post("/api/auth/login", json=login_data)
            # Should not crash, should return standard auth failure
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Test XSS attempts in registration
        for xss_payload in security_test_data["xss_payloads"]:
            user_data = {
                "username": "testuser123",
                "email": "test@example.com",
                "password": "SecureTest123!",
                "full_name": xss_payload,  # XSS in full_name
                "role": "user"
            }
            response = client.post("/api/auth/register", json=user_data)
            # Should either validate/sanitize or reject
            assert response.status_code in [
                status.HTTP_201_CREATED, 
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    def test_concurrent_authentication_requests(self, client):
        """Test handling of concurrent authentication requests"""
        import concurrent.futures
        import threading
        
        def login_request():
            login_data = {
                "username": "demo",
                "password": "DemoUser123!"
            }
            return client.post("/api/auth/login", json=login_data)
        
        # Submit multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(login_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed (or be rate limited)
        for response in responses:
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS]
    
    def test_authentication_security_headers(self, client):
        """Test security headers are present in authentication responses"""
        login_data = {
            "username": "demo",
            "password": "DemoUser123!"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        # Check for security headers
        headers = response.headers
        assert "x-content-type-options" in headers
        assert "x-frame-options" in headers
        assert "x-xss-protection" in headers
        
        # Content-Type should be application/json
        assert headers["content-type"] == "application/json"
    
    @pytest.mark.slow
    def test_authentication_performance(self, client):
        """Test authentication endpoint performance"""
        login_data = {
            "username": "demo", 
            "password": "DemoUser123!"
        }
        
        start_time = time.time()
        response = client.post("/api/auth/login", json=login_data)
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        
        # Authentication should complete within reasonable time
        response_time = end_time - start_time
        assert response_time < 2.0  # Should complete within 2 seconds


@pytest.mark.auth
@pytest.mark.integration
class TestAuthEndpointsAsync:
    """Async tests for authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_async_user_registration(self, async_client, test_users):
        """Test async user registration"""
        user_data = test_users["valid_user"].copy()
        user_data["username"] = "async_test_user"
        
        response = await async_client.post("/api/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        assert response_data["username"] == user_data["username"]
    
    @pytest.mark.asyncio
    async def test_async_user_login(self, async_client):
        """Test async user login"""
        login_data = {
            "username": "demo",
            "password": "DemoUser123!"
        }
        
        response = await async_client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "access_token" in response_data
        assert "refresh_token" in response_data
    
    @pytest.mark.asyncio
    async def test_async_concurrent_registrations(self, async_client, test_users):
        """Test concurrent async user registrations"""
        import asyncio
        
        async def register_user(username_suffix):
            user_data = test_users["valid_user"].copy()
            user_data["username"] = f"async_user_{username_suffix}"
            user_data["email"] = f"async_user_{username_suffix}@example.com"
            return await async_client.post("/api/auth/register", json=user_data)
        
        # Create multiple concurrent registration tasks
        tasks = [register_user(i) for i in range(3)]
        responses = await asyncio.gather(*tasks)
        
        # All registrations should succeed
        for response in responses:
            assert response.status_code == status.HTTP_201_CREATED