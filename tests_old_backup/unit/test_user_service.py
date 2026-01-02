"""
Comprehensive Unit Tests for User Service
Tests secure authentication, registration, and user management functionality
"""

import pytest
import hashlib
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from passlib.context import CryptContext

from src.services.user_service import UserService
from src.auth.models import UserCreate, UserLogin, UserResponse


@pytest.mark.unit
class TestUserService:
    """Test suite for UserService class"""
    
    @pytest.fixture
    def user_service(self):
        """Create fresh UserService instance for each test"""
        return UserService()
    
    @pytest.fixture
    def valid_user_data(self):
        """Valid user registration data"""
        return {
            "username": "testuser123",
            "email": "test@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User",
            "role": "user"
        }
    
    @pytest.fixture
    def weak_password_user_data(self):
        """User data with weak password"""
        return {
            "username": "weakuser",
            "email": "weak@example.com", 
            "password": "weak",
            "full_name": "Weak User",
            "role": "user"
        }
    
    def test_user_service_initialization(self, user_service):
        """Test UserService proper initialization"""
        # Test password context is configured correctly
        assert isinstance(user_service.pwd_context, CryptContext)
        assert "bcrypt" in user_service.pwd_context.schemes()
        
        # Test default users are created
        assert len(user_service._users) >= 2  # admin and demo users
        assert "admin" in user_service._users
        assert "demo" in user_service._users
        
        # Test default admin user properties
        admin_user = user_service._users["admin"]
        assert admin_user["role"] == "admin"
        assert admin_user["is_active"] is True
        assert admin_user["account_locked"] is False
        assert admin_user["failed_login_attempts"] == 0
    
    def test_password_hashing_security(self, user_service):
        """Test password hashing uses secure bcrypt with proper rounds"""
        password = "TestPassword123!"
        hashed = user_service.pwd_context.hash(password)
        
        # Test hash format (bcrypt should start with $2b$)
        assert hashed.startswith("$2b$")
        
        # Test hash verification
        assert user_service.pwd_context.verify(password, hashed)
        assert not user_service.pwd_context.verify("wrong_password", hashed)
        
        # Test different passwords create different hashes
        hash1 = user_service.pwd_context.hash(password)
        hash2 = user_service.pwd_context.hash(password)
        assert hash1 != hash2  # Due to salt
    
    def test_user_id_generation(self, user_service):
        """Test secure user ID generation"""
        user_id1 = user_service._generate_user_id()
        user_id2 = user_service._generate_user_id()
        
        # Test IDs are different
        assert user_id1 != user_id2
        
        # Test ID format and length
        assert len(user_id1) >= 20  # URL-safe base64 encoding
        assert isinstance(user_id1, str)
    
    @pytest.mark.parametrize("password,expected", [
        ("SecurePass123!", True),  # Valid password
        ("AnotherGood1@", True),   # Valid password
        ("weak", False),           # Too short
        ("password123", False),    # No uppercase
        ("PASSWORD123", False),    # No lowercase
        ("Password", False),       # No digit
        ("Password123", False),    # No special char
        ("", False),               # Empty
        ("a" * 200, False),        # Too long (if max length enforced)
    ])
    def test_password_strength_validation(self, user_service, password, expected):
        """Test password strength validation logic"""
        result = user_service._validate_password_strength(password)
        assert result == expected
    
    def test_successful_user_registration(self, user_service, valid_user_data):
        """Test successful user registration with valid data"""
        user_create = UserCreate(**valid_user_data)
        result = user_service.register_user(user_create)
        
        # Test successful registration
        assert result is not None
        assert isinstance(result, UserResponse)
        
        # Test returned user data
        assert result.username == valid_user_data["username"]
        assert result.email == valid_user_data["email"]
        assert result.full_name == valid_user_data["full_name"]
        assert result.role == valid_user_data["role"]
        assert result.is_active is True
        assert isinstance(result.created_at, datetime)
        
        # Test user is stored internally
        assert valid_user_data["username"] in user_service._users
        stored_user = user_service._users[valid_user_data["username"]]
        assert stored_user["email"] == valid_user_data["email"]
        assert stored_user["account_locked"] is False
    
    def test_registration_duplicate_username(self, user_service, valid_user_data):
        """Test registration fails with duplicate username"""
        user_create = UserCreate(**valid_user_data)
        
        # First registration should succeed
        result1 = user_service.register_user(user_create)
        assert result1 is not None
        
        # Second registration with same username should fail
        result2 = user_service.register_user(user_create)
        assert result2 is None
    
    def test_registration_duplicate_email(self, user_service, valid_user_data):
        """Test registration fails with duplicate email"""
        # First user
        user_create1 = UserCreate(**valid_user_data)
        result1 = user_service.register_user(user_create1)
        assert result1 is not None
        
        # Second user with same email but different username
        user_data2 = valid_user_data.copy()
        user_data2["username"] = "different_username"
        user_create2 = UserCreate(**user_data2)
        result2 = user_service.register_user(user_create2)
        assert result2 is None
    
    def test_registration_weak_password_rejected(self, user_service, weak_password_user_data):
        """Test registration rejects weak passwords"""
        user_create = UserCreate(**weak_password_user_data)
        result = user_service.register_user(user_create)
        assert result is None
    
    def test_successful_authentication(self, user_service):
        """Test successful user authentication"""
        # Use default admin user for authentication test
        username = "admin"
        password = "SecureAdmin123!"
        
        result = user_service.authenticate_user(username, password)
        
        assert result is not None
        assert isinstance(result, UserResponse)
        assert result.username == username
        assert result.role == "admin"
        assert result.is_active is True
        assert isinstance(result.last_login, datetime)
    
    def test_authentication_invalid_username(self, user_service):
        """Test authentication fails with invalid username"""
        result = user_service.authenticate_user("nonexistent_user", "any_password")
        assert result is None
    
    def test_authentication_invalid_password(self, user_service):
        """Test authentication fails with invalid password"""
        result = user_service.authenticate_user("admin", "wrong_password")
        assert result is None
    
    def test_authentication_inactive_user(self, user_service, valid_user_data):
        """Test authentication fails for inactive users"""
        # Create and register user
        user_create = UserCreate(**valid_user_data)
        user_service.register_user(user_create)
        
        # Deactivate user
        user_service._users[valid_user_data["username"]]["is_active"] = False
        
        # Test authentication fails
        result = user_service.authenticate_user(
            valid_user_data["username"], 
            valid_user_data["password"]
        )
        assert result is None
    
    def test_authentication_locked_account(self, user_service, valid_user_data):
        """Test authentication fails for locked accounts"""
        # Create and register user
        user_create = UserCreate(**valid_user_data)
        user_service.register_user(user_create)
        
        # Lock account
        user_service._users[valid_user_data["username"]]["account_locked"] = True
        
        # Test authentication fails
        result = user_service.authenticate_user(
            valid_user_data["username"], 
            valid_user_data["password"]
        )
        assert result is None
    
    def test_failed_login_attempt_tracking(self, user_service, valid_user_data):
        """Test failed login attempts are tracked correctly"""
        # Create and register user
        user_create = UserCreate(**valid_user_data)
        user_service.register_user(user_create)
        
        username = valid_user_data["username"]
        
        # Test initial failed login attempts
        assert user_service._users[username]["failed_login_attempts"] == 0
        
        # Make failed login attempts
        for i in range(3):
            result = user_service.authenticate_user(username, "wrong_password")
            assert result is None
            assert user_service._users[username]["failed_login_attempts"] == i + 1
    
    def test_account_lockout_after_failed_attempts(self, user_service, valid_user_data):
        """Test account gets locked after 5 failed login attempts"""
        # Create and register user
        user_create = UserCreate(**valid_user_data)
        user_service.register_user(user_create)
        
        username = valid_user_data["username"]
        
        # Make 5 failed login attempts
        for i in range(5):
            result = user_service.authenticate_user(username, "wrong_password")
            assert result is None
        
        # Account should be locked
        assert user_service._users[username]["account_locked"] is True
        assert user_service._users[username]["failed_login_attempts"] == 5
        
        # Even correct password should fail now
        result = user_service.authenticate_user(username, valid_user_data["password"])
        assert result is None
    
    def test_failed_attempts_reset_on_successful_login(self, user_service):
        """Test failed login attempts reset after successful authentication"""
        username = "admin"
        password = "SecureAdmin123!"
        
        # Make some failed attempts first
        for i in range(3):
            user_service.authenticate_user(username, "wrong_password")
        
        assert user_service._users[username]["failed_login_attempts"] == 3
        
        # Successful login should reset counter
        result = user_service.authenticate_user(username, password)
        assert result is not None
        assert user_service._users[username]["failed_login_attempts"] == 0
    
    def test_find_user_by_username(self, user_service):
        """Test finding user by username"""
        result = user_service._find_user("admin")
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == "admin"
    
    def test_find_user_by_email(self, user_service):
        """Test finding user by email"""
        result = user_service._find_user("admin@solace-ai.com")
        assert result is not None
        assert result["username"] == "admin"
        assert result["email"] == "admin@solace-ai.com"
    
    def test_find_nonexistent_user(self, user_service):
        """Test finding nonexistent user returns None"""
        result = user_service._find_user("nonexistent")
        assert result is None
    
    def test_find_user_by_email_only(self, user_service):
        """Test finding user specifically by email"""
        result = user_service._find_user_by_email("demo@solace-ai.com")
        assert result is not None
        assert result["username"] == "demo"
        assert result["email"] == "demo@solace-ai.com"
    
    def test_unlock_account_success(self, user_service, valid_user_data):
        """Test successful account unlock"""
        # Create user and lock account
        user_create = UserCreate(**valid_user_data)
        user_service.register_user(user_create)
        
        username = valid_user_data["username"]
        user_service._users[username]["account_locked"] = True
        user_service._users[username]["failed_login_attempts"] = 5
        
        # Unlock account
        result = user_service.unlock_account(username)
        assert result is True
        assert user_service._users[username]["account_locked"] is False
        assert user_service._users[username]["failed_login_attempts"] == 0
    
    def test_unlock_nonexistent_account(self, user_service):
        """Test unlocking nonexistent account returns False"""
        result = user_service.unlock_account("nonexistent")
        assert result is False
    
    def test_get_user_count(self, user_service):
        """Test getting total user count"""
        initial_count = user_service.get_user_count()
        assert initial_count >= 2  # At least admin and demo
        
        # Add a user and test count increases
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecureTest123!",
            "full_name": "Test User",
            "role": "user"
        }
        user_create = UserCreate(**user_data)
        user_service.register_user(user_create)
        
        new_count = user_service.get_user_count()
        assert new_count == initial_count + 1
    
    def test_list_users_admin_function(self, user_service):
        """Test listing all users (admin function)"""
        users = user_service.list_users()
        
        # Test return format
        assert isinstance(users, list)
        assert len(users) >= 2  # At least admin and demo
        
        # Test user data format
        for user in users:
            assert "username" in user
            assert "email" in user
            assert "full_name" in user
            assert "role" in user
            assert "is_active" in user
            assert "account_locked" in user
            assert "created_at" in user
            # Should not contain sensitive data
            assert "password_hash" not in user
            assert "password" not in user
    
    def test_authentication_with_email(self, user_service):
        """Test authentication using email instead of username"""
        result = user_service.authenticate_user("admin@solace-ai.com", "SecureAdmin123!")
        assert result is not None
        assert result.username == "admin"
        assert result.email == "admin@solace-ai.com"
    
    @pytest.mark.parametrize("role", ["user", "therapist", "admin"])
    def test_user_registration_different_roles(self, user_service, role):
        """Test user registration with different roles"""
        user_data = {
            "username": f"test_{role}",
            "email": f"test_{role}@example.com",
            "password": "SecureTest123!",
            "full_name": f"Test {role.title()}",
            "role": role
        }
        
        user_create = UserCreate(**user_data)
        result = user_service.register_user(user_create)
        
        assert result is not None
        assert result.role == role
        assert user_service._users[user_data["username"]]["role"] == role
    
    def test_authentication_exception_handling(self, user_service):
        """Test authentication handles exceptions gracefully"""
        # Mock the password verification to raise an exception
        with patch.object(user_service.pwd_context, 'verify', side_effect=Exception("Test error")):
            result = user_service.authenticate_user("admin", "SecureAdmin123!")
            assert result is None
    
    def test_registration_exception_handling(self, user_service, valid_user_data):
        """Test registration handles exceptions gracefully"""
        # Mock password hashing to raise an exception
        with patch.object(user_service.pwd_context, 'hash', side_effect=Exception("Hash error")):
            user_create = UserCreate(**valid_user_data)
            result = user_service.register_user(user_create)
            assert result is None
    
    def test_unlock_account_exception_handling(self, user_service):
        """Test unlock account handles exceptions gracefully"""
        # Mock _find_user to raise an exception
        with patch.object(user_service, '_find_user', side_effect=Exception("Find error")):
            result = user_service.unlock_account("admin")
            assert result is False
    
    def test_password_hash_verification_timing_attack_resistance(self, user_service):
        """Test password verification is resistant to timing attacks"""
        import time
        
        # Test with valid user
        start_time = time.time()
        user_service.authenticate_user("admin", "wrong_password")
        valid_user_time = time.time() - start_time
        
        # Test with invalid user  
        start_time = time.time()
        user_service.authenticate_user("nonexistent_user", "wrong_password")
        invalid_user_time = time.time() - start_time
        
        # Times should be relatively similar (within reasonable variance)
        # This is a basic timing test - in production, use constant-time comparison
        time_difference = abs(valid_user_time - invalid_user_time)
        assert time_difference < 0.1  # 100ms tolerance


@pytest.mark.unit
class TestUserServiceSecurity:
    """Security-focused tests for UserService"""
    
    def test_password_not_stored_in_plaintext(self, user_service):
        """Test passwords are never stored in plaintext"""
        user_data = {
            "username": "security_test",
            "email": "security@example.com",
            "password": "PlaintextPassword123!",
            "full_name": "Security Test",
            "role": "user"
        }
        
        user_create = UserCreate(**user_data)
        result = user_service.register_user(user_create)
        assert result is not None
        
        # Check stored user data
        stored_user = user_service._users["security_test"]
        assert "password" not in stored_user
        assert stored_user["password_hash"] != user_data["password"]
        assert stored_user["password_hash"].startswith("$2b$")  # bcrypt format
    
    def test_user_enumeration_protection(self, user_service):
        """Test protection against user enumeration attacks"""
        # Response time and behavior should be similar for valid and invalid users
        valid_username_result = user_service.authenticate_user("admin", "wrong_password")
        invalid_username_result = user_service.authenticate_user("nonexistent", "wrong_password")
        
        # Both should return None (don't reveal if username exists)
        assert valid_username_result is None
        assert invalid_username_result is None
    
    def test_bcrypt_rounds_security(self, user_service):
        """Test bcrypt is configured with adequate rounds for security"""
        # Test password hashing uses recommended rounds (12)
        password = "TestPassword123!"
        hashed = user_service.pwd_context.hash(password)
        
        # Extract rounds from bcrypt hash
        parts = hashed.split('$')
        rounds = int(parts[2])
        
        # Should use at least 12 rounds for security
        assert rounds >= 12
    
    def test_account_lockout_prevents_brute_force(self, user_service):
        """Test account lockout mechanism prevents brute force attacks"""
        username = "demo"
        
        # Simulate brute force attack
        for i in range(10):  # Try more than lockout threshold
            result = user_service.authenticate_user(username, f"attempt_{i}")
            assert result is None
        
        # Account should be locked
        user_data = user_service._users[username]
        assert user_data["account_locked"] is True
        
        # Even correct password should fail
        result = user_service.authenticate_user(username, "DemoUser123!")
        assert result is None