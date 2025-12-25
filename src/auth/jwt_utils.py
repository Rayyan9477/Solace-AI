"""
JWT Token Utilities for Solace-AI API

Provides secure JWT token generation, validation, and management functionality.
"""

import uuid
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
import jwt
from passlib.context import CryptContext
from src.config.security import SecurityConfig, SecurityExceptions
from src.auth.models import TokenData, UserResponse

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """
    Thread-safe token blacklist with TTL-based cleanup and bounded size.

    Security improvements (SEC-005):
    - Stores JTI (token ID) instead of full tokens to save memory
    - Automatic cleanup of expired entries
    - Maximum size limit to prevent memory exhaustion
    - Thread-safe operations
    """

    MAX_BLACKLIST_SIZE = 10000  # Maximum number of blacklisted tokens

    def __init__(self):
        self._blacklist: Dict[str, datetime] = {}  # jti -> expiry_time
        self._lock = threading.Lock()
        self._last_cleanup = datetime.now(timezone.utc)
        self._cleanup_interval = timedelta(minutes=15)

    def add(self, jti: str, expires_at: datetime) -> None:
        """Add a token JTI to the blacklist with its expiry time."""
        with self._lock:
            # Cleanup if needed
            self._maybe_cleanup()

            # If at max capacity, force cleanup or remove oldest
            if len(self._blacklist) >= self.MAX_BLACKLIST_SIZE:
                self._force_cleanup()

            self._blacklist[jti] = expires_at

    def contains(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted."""
        with self._lock:
            if jti not in self._blacklist:
                return False

            # Check if expired (can be removed)
            expiry = self._blacklist[jti]
            if expiry < datetime.now(timezone.utc):
                del self._blacklist[jti]
                return False

            return True

    def _maybe_cleanup(self) -> None:
        """Cleanup expired entries if cleanup interval has passed."""
        now = datetime.now(timezone.utc)
        if now - self._last_cleanup > self._cleanup_interval:
            self._do_cleanup()
            self._last_cleanup = now

    def _force_cleanup(self) -> None:
        """Force cleanup when at capacity."""
        self._do_cleanup()
        # If still at capacity after cleanup, remove oldest 10%
        if len(self._blacklist) >= self.MAX_BLACKLIST_SIZE:
            sorted_entries = sorted(self._blacklist.items(), key=lambda x: x[1])
            remove_count = self.MAX_BLACKLIST_SIZE // 10
            for jti, _ in sorted_entries[:remove_count]:
                del self._blacklist[jti]
            logger.warning(f"Token blacklist forced removal of {remove_count} entries")

    def _do_cleanup(self) -> None:
        """Remove all expired entries."""
        now = datetime.now(timezone.utc)
        expired = [jti for jti, exp in self._blacklist.items() if exp < now]
        for jti in expired:
            del self._blacklist[jti]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired blacklist entries")

    def size(self) -> int:
        """Return current blacklist size."""
        with self._lock:
            return len(self._blacklist)


class JWTManager:
    """
    JWT token management with security best practices.

    Security measures implemented:
    - SEC-005: Token blacklist with TTL and bounded size
    - SEC-007: Algorithm confusion protection (CVE-2015-2951, CVE-2018-0114)
        - Explicitly defines allowed algorithms (no 'none')
        - Validates token header algorithm before decoding
        - Uses symmetric HS256 by default (configurable)
    """

    # SEC-007: Explicitly allowed algorithms - 'none' is NEVER allowed
    ALLOWED_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = SecurityConfig.SECRET_KEY
        self.algorithm = SecurityConfig.ALGORITHM

        # SEC-007: Validate that configured algorithm is in allowed list
        if self.algorithm not in self.ALLOWED_ALGORITHMS:
            raise ValueError(
                f"Invalid JWT algorithm: {self.algorithm}. "
                f"Allowed algorithms: {self.ALLOWED_ALGORITHMS}"
            )

        self.access_token_expire_minutes = SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS

        # Token blacklist with TTL and bounded size (SEC-005)
        self._token_blacklist = TokenBlacklist()

        # Legacy compatibility - deprecated, use _token_blacklist instead
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
    
    def _validate_token_header(self, token: str) -> None:
        """
        Validate JWT token header algorithm before decoding (SEC-007).

        Explicitly checks for algorithm confusion attacks including:
        - 'none' algorithm attack (CVE-2015-2951)
        - RS256 to HS256 downgrade attack (CVE-2018-0114)

        Args:
            token: The raw JWT token string

        Raises:
            SecurityExceptions.InvalidTokenError: If algorithm is invalid
        """
        try:
            # Extract header without verification
            header = jwt.get_unverified_header(token)
            token_alg = header.get("alg", "").upper()

            # SEC-007: Explicitly reject 'none' algorithm
            if token_alg.lower() == "none" or not token_alg:
                logger.warning(
                    "JWT algorithm confusion attempt detected: 'none' algorithm",
                    extra={"event_type": "jwt_algorithm_attack", "algorithm": token_alg}
                )
                raise SecurityExceptions.InvalidTokenError(
                    "Invalid token: 'none' algorithm not allowed"
                )

            # SEC-007: Reject algorithms not in our allowed list
            if token_alg not in self.ALLOWED_ALGORITHMS:
                logger.warning(
                    f"JWT algorithm mismatch: token uses {token_alg}, expected {self.algorithm}",
                    extra={"event_type": "jwt_algorithm_mismatch", "token_alg": token_alg}
                )
                raise SecurityExceptions.InvalidTokenError(
                    f"Invalid token algorithm: {token_alg}"
                )

        except jwt.exceptions.DecodeError as e:
            raise SecurityExceptions.InvalidTokenError(f"Malformed token header: {str(e)}")

    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """
        Verify and decode a JWT token.

        Security measures:
        - SEC-007: Validates token header algorithm before decoding
        - SEC-005: Checks token blacklist
        """
        try:
            # SEC-007: Validate token header algorithm BEFORE decoding
            self._validate_token_header(token)

            # First decode to get JTI for blacklist check
            # Use options to skip exp verification initially
            unverified_payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # SEC-007: Only allow configured algorithm
                options={"verify_exp": False}
            )

            # Check if token JTI is blacklisted (SEC-005)
            jti = unverified_payload.get("jti")
            if jti and self._token_blacklist.contains(jti):
                raise SecurityExceptions.InvalidTokenError("Token has been revoked")

            # Legacy blacklist check for backward compatibility
            if token in self.blacklisted_tokens:
                raise SecurityExceptions.InvalidTokenError("Token has been revoked")

            # Now fully decode and verify the token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # SEC-007: Only allow configured algorithm
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
        except SecurityExceptions.InvalidTokenError:
            raise
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
        """
        Add token to blacklist.

        Uses the new JTI-based blacklist with TTL for memory efficiency (SEC-005).
        Falls back to legacy full-token blacklist for backward compatibility.
        """
        try:
            # Decode token to get JTI and expiry
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens to be blacklisted
            )
            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                # Convert exp timestamp to datetime
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                self._token_blacklist.add(jti, expires_at)
            else:
                # Fallback to legacy blacklist if no JTI
                self.blacklisted_tokens.add(token)
                logger.warning("Token blacklisted without JTI - using legacy method")

        except jwt.JWTError:
            # If we can't decode, still add to legacy blacklist
            self.blacklisted_tokens.add(token)
            logger.warning("Failed to decode token for blacklisting - using legacy method")
    
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