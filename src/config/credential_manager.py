"""
Secure Credential Management System
Implements encrypted storage and secure access patterns for sensitive data
"""

import os
import json
import base64
import secrets
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class CredentialManager:
    """Secure credential management with encryption"""
    
    def __init__(self, master_key: Optional[str] = None, storage_path: Optional[str] = None):
        """
        Initialize credential manager
        
        Args:
            master_key: Master encryption key (if None, uses environment)
            storage_path: Path to encrypted credential store
        """
        self.storage_path = Path(storage_path or "credentials.enc")
        self._cipher = self._setup_encryption(master_key)
        self._credentials: Dict[str, Any] = {}
        self._load_credentials()
    
    def _setup_encryption(self, master_key: Optional[str] = None) -> Fernet:
        """Setup encryption cipher"""
        if master_key:
            key = master_key.encode()
        else:
            # Use environment variable or generate
            env_key = os.getenv("CREDENTIAL_MASTER_KEY")
            if not env_key:
                raise ValueError(
                    "CREDENTIAL_MASTER_KEY environment variable required. "
                    "Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
                )
            key = env_key.encode()
        
        # Derive key using PBKDF2
        salt = b"solace_ai_salt_2024"  # In production, use random salt stored separately
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key))
        return Fernet(key)
    
    def _load_credentials(self):
        """Load encrypted credentials from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self._cipher.decrypt(encrypted_data)
                self._credentials = json.loads(decrypted_data.decode())
                logger.info("Credentials loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load credentials: {str(e)}")
                self._credentials = {}
        else:
            logger.info("No existing credential store found")
    
    def _save_credentials(self):
        """Save encrypted credentials to storage"""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Encrypt and save
            data = json.dumps(self._credentials, indent=2)
            encrypted_data = self._cipher.encrypt(data.encode())
            
            with open(self.storage_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.storage_path, 0o600)
            logger.info("Credentials saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {str(e)}")
            raise
    
    def store_credential(self, key: str, value: str, category: str = "general") -> bool:
        """
        Store encrypted credential
        
        Args:
            key: Credential identifier
            value: Credential value (will be encrypted)
            category: Category for organization
            
        Returns:
            bool: Success status
        """
        try:
            if category not in self._credentials:
                self._credentials[category] = {}
            
            # Store with metadata
            self._credentials[category][key] = {
                "value": value,
                "created_at": secrets.token_urlsafe(16),  # Timestamp placeholder
                "accessed_count": 0
            }
            
            self._save_credentials()
            logger.info(f"Credential stored: {category}.{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential {key}: {str(e)}")
            return False
    
    def get_credential(self, key: str, category: str = "general") -> Optional[str]:
        """
        Retrieve credential securely
        
        Args:
            key: Credential identifier
            category: Category to search in
            
        Returns:
            Decrypted credential value or None
        """
        try:
            if category in self._credentials and key in self._credentials[category]:
                credential_data = self._credentials[category][key]
                
                # Update access count
                credential_data["accessed_count"] = credential_data.get("accessed_count", 0) + 1
                self._save_credentials()
                
                return credential_data["value"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential {key}: {str(e)}")
            return None
    
    def delete_credential(self, key: str, category: str = "general") -> bool:
        """Delete credential securely"""
        try:
            if category in self._credentials and key in self._credentials[category]:
                del self._credentials[category][key]
                
                # Clean up empty categories
                if not self._credentials[category]:
                    del self._credentials[category]
                
                self._save_credentials()
                logger.info(f"Credential deleted: {category}.{key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete credential {key}: {str(e)}")
            return False
    
    def list_credentials(self) -> Dict[str, list]:
        """List available credentials by category"""
        result = {}
        for category, credentials in self._credentials.items():
            result[category] = [
                {
                    "key": key,
                    "accessed_count": data.get("accessed_count", 0),
                    "created_at": data.get("created_at", "unknown")
                }
                for key, data in credentials.items()
            ]
        return result
    
    def rotate_master_key(self, new_master_key: str) -> bool:
        """Rotate the master encryption key"""
        try:
            # Save current credentials
            old_credentials = self._credentials.copy()
            
            # Setup new cipher
            old_cipher = self._cipher
            self._cipher = self._setup_encryption(new_master_key)
            
            # Re-encrypt with new key
            self._save_credentials()
            
            logger.info("Master key rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate master key: {str(e)}")
            # Restore old cipher
            self._cipher = old_cipher
            self._credentials = old_credentials
            return False

class APIKeyManager:
    """Specialized manager for API keys"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.cm = credential_manager
        self.category = "api_keys"
    
    def store_openai_key(self, api_key: str) -> bool:
        """Store OpenAI API key"""
        return self.cm.store_credential("openai", api_key, self.category)
    
    def store_anthropic_key(self, api_key: str) -> bool:
        """Store Anthropic API key"""
        return self.cm.store_credential("anthropic", api_key, self.category)
    
    def store_google_key(self, api_key: str) -> bool:
        """Store Google/Gemini API key"""
        return self.cm.store_credential("google", api_key, self.category)
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        # Try credential manager first, then environment
        key = self.cm.get_credential("openai", self.category)
        return key or os.getenv("OPENAI_API_KEY")
    
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        key = self.cm.get_credential("anthropic", self.category)
        return key or os.getenv("ANTHROPIC_API_KEY")
    
    def get_google_key(self) -> Optional[str]:
        """Get Google/Gemini API key"""
        key = self.cm.get_credential("google", self.category)
        return key or os.getenv("GOOGLE_API_KEY")

class DatabaseCredentialManager:
    """Specialized manager for database credentials"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.cm = credential_manager
        self.category = "database"
    
    def store_database_url(self, url: str, environment: str = "default") -> bool:
        """Store database URL"""
        return self.cm.store_credential(f"url_{environment}", url, self.category)
    
    def get_database_url(self, environment: str = "default") -> Optional[str]:
        """Get database URL"""
        key = f"url_{environment}"
        url = self.cm.get_credential(key, self.category)
        return url or os.getenv("DATABASE_URL")

# Global instances
try:
    credential_manager = CredentialManager()
    api_key_manager = APIKeyManager(credential_manager)
    db_credential_manager = DatabaseCredentialManager(credential_manager)
    
    logger.info("Credential management system initialized")
except Exception as e:
    logger.warning(f"Credential manager initialization failed: {str(e)}")
    logger.warning("Falling back to environment variables only")
    
    # Fallback to environment-only mode
    credential_manager = None
    api_key_manager = None
    db_credential_manager = None