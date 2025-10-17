"""
JWT Token Utilities for Solace-AI API

Provides secure JWT token generation, validation, and management functionality.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import jwt
from passlib.context import CryptContext
from src.config.security import SecurityConfig, SecurityExceptions
from src.auth.models import TokenData, UserResponse


class JWTManager:
    """JWT token management with security best practices"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = SecurityConfig.SECRET_KEY
        self.algorithm = SecurityConfig.ALGORITHM
        self.access_token_expire_minutes = SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS
        
        # Token blacklist (in production, use Redis or database)
        self.blacklisted_tokens = set()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password with secure settings"""
        return self.pwd_context.hash(password, rounds=SecurityConfig.BCRYPT_ROUNDS)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a new access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),  # JWT ID for token revocation
            "token_type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise SecurityExceptions.InvalidTokenError(f"Failed to create token: {str(e)}")
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a new refresh token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),
            "token_type": "refresh"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            raise SecurityExceptions.InvalidTokenError(f"Failed to create refresh token: {str(e)}")
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode a JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise SecurityExceptions.InvalidTokenError("Token has been revoked")
            
            # Decode the token
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            # Verify token type
            if payload.get("token_type") != token_type:
                raise SecurityExceptions.InvalidTokenError(f"Expected {token_type} token")
            
            # Extract user data
            user_id = payload.get("sub")
            if user_id is None:
                raise SecurityExceptions.InvalidTokenError("Token missing user ID")
            
            return TokenData(
                sub=user_id,
                username=payload.get("username"),
                role=payload.get("role", "user"),
                permissions=payload.get("permissions", []),
                exp=payload.get("exp"),
                iat=payload.get("iat"),
                jti=payload.get("jti")
            )
            
        except jwt.ExpiredSignatureError:
            raise SecurityExceptions.InvalidTokenError("Token has expired")
        except jwt.JWTError as e:
            raise SecurityExceptions.InvalidTokenError(f"Token validation failed: {str(e)}")
        except Exception as e:
            raise SecurityExceptions.InvalidTokenError(f"Unexpected error: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """Generate new access token using refresh token"""
        try:
            # Verify refresh token
            token_data = self.verify_token(refresh_token, token_type="refresh")
            
            # Create new access token with same user data
            access_token_data = {
                "sub": token_data.sub,
                "username": token_data.username,
                "role": token_data.role,
                "permissions": token_data.permissions
            }
            
            new_access_token = self.create_access_token(access_token_data)
            
            # Optionally create new refresh token (token rotation)
            new_refresh_token = self.create_refresh_token({
                "sub": token_data.sub,
                "username": token_data.username
            })
            
            # Blacklist old refresh token
            self.blacklist_token(refresh_token)
            
            return new_access_token, new_refresh_token
            
        except SecurityExceptions.InvalidTokenError:
            raise
        except Exception as e:
            raise SecurityExceptions.InvalidTokenError(f"Failed to refresh token: {str(e)}")
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)
    
    def get_user_permissions(self, role: str) -> List[str]:
        """Get user permissions based on role"""
        return SecurityConfig.USER_ROLES.get(role, SecurityConfig.USER_ROLES["user"])
    
    def create_user_tokens(self, user: UserResponse) -> Dict[str, Any]:
        """Create both access and refresh tokens for a user"""
        permissions = self.get_user_permissions(user.role.value)
        
        token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": permissions
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token({
            "sub": user.id,
            "username": user.username
        })
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "user": user
        }
    
    def extract_token_from_header(self, authorization_header: str) -> str:
        """Extract JWT token from Authorization header"""
        if not authorization_header:
            raise SecurityExceptions.AuthenticationError("Authorization header missing")
        
        try:
            scheme, token = authorization_header.split()
            if scheme.lower() != "bearer":
                raise SecurityExceptions.AuthenticationError("Invalid authentication scheme")
            return token
        except ValueError:
            raise SecurityExceptions.AuthenticationError("Invalid authorization header format")
    
    def validate_token_permissions(self, token_data: TokenData, required_permissions: List[str]) -> bool:
        """Validate if token has required permissions"""
        user_permissions = token_data.permissions
        
        # Admin role has all permissions
        if "admin" in token_data.role:
            return True
        
        # Check if user has all required permissions
        for permission in required_permissions:
            if permission not in user_permissions and "*" not in user_permissions:
                return False
        
        return True


# Global JWT manager instance
jwt_manager = JWTManager()