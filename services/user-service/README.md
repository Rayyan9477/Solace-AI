# User Service - Domain Layer

Production-ready User Service following Clean Architecture and Domain-Driven Design principles.

## Overview

The User Service manages user accounts, authentication, preferences, and consent for the Solace-AI platform. This implementation follows HIPAA/GDPR compliance requirements and implements strict domain boundaries.

## Architecture

### Clean Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│              Domain Layer (Core Business Logic)        │
│  ├── Entities (User, UserPreferences)                  │
│  ├── Value Objects (UserRole, EmailAddress, etc.)     │
│  ├── Domain Services (ConsentService)                  │
│  └── Domain Events                                     │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│         Infrastructure Layer (External Dependencies)   │
│  ├── JWTService (Token generation/verification)        │
│  ├── PasswordService (Argon2/bcrypt hashing)          │
│  ├── TokenService (Email verification/password reset) │
│  └── EncryptionService (Field-level PII/PHI encryption)│
└────────────────────────────────────────────────────────┘
```

### Key Components

#### Domain Layer
- **Entities**: `User`, `UserPreferences` with business rules and invariants
- **Value Objects**: `UserRole`, `AccountStatus`, `EmailAddress`, `ConsentType`, `ConsentRecord`
- **Domain Services**: `ConsentService` for consent management
- **Events**: Domain events for event-driven architecture
- **Config**: Externalized configuration with environment variables

#### Infrastructure Layer
- **JWTService**: JWT token generation, verification, and refresh
- **PasswordService**: Dual algorithm password hashing (Argon2id + bcrypt) with automatic migration
- **TokenService**: Time-limited tokens for email verification and password reset
- **EncryptionService**: Field-level encryption for PII/PHI (HIPAA compliance)

## Features

### User Entity
- Account lifecycle management (create, activate, suspend, delete)
- Email verification workflow
- Login attempt tracking and account lockout
- Soft delete for GDPR compliance
- Profile updates with validation

### User Preferences
- Notification preferences (email, SMS, push)
- Privacy and data sharing settings
- Accessibility options
- Theme and language preferences

### Consent Management
- HIPAA/GDPR compliant consent tracking
- Audit trail for all consent actions
- Required vs. optional consents
- Consent expiry and revocation
- Event publishing for downstream systems

### Infrastructure Services

#### JWT Authentication (JWTService)
- **Token Generation**: Access tokens (15 min) and refresh tokens (30 days)
- **Token Verification**: Type-safe payload extraction with expiry validation
- **Token Refresh**: Generate new access tokens from valid refresh tokens
- **Header Extraction**: Parse Bearer tokens from Authorization headers
- **Algorithms**: Configurable (HS256, RS256, etc.)

**Usage Example:**
```python
from src.infrastructure import create_jwt_service, TokenType

jwt_service = create_jwt_service(secret_key="your-secret-key")

# Generate token pair
token_pair = jwt_service.generate_token_pair(
    user_id=user.id,
    email=user.email.value,
    role=user.role.value
)

# Verify access token
payload = jwt_service.verify_token(token_pair.access_token, TokenType.ACCESS)

# Refresh access token
new_access_token = jwt_service.refresh_access_token(token_pair.refresh_token)
```

#### Password Security (PasswordService)
- **Argon2id Hashing**: Memory-hard algorithm (64MB, time_cost=2) for new passwords
- **bcrypt Support**: Legacy password verification for gradual migration
- **Automatic Migration**: Returns new Argon2 hash when bcrypt password verified
- **Algorithm Detection**: Automatic hash format detection
- **Rehash Checking**: Identifies passwords needing migration

**Usage Example:**
```python
from src.infrastructure import create_password_service

password_service = create_password_service()

# Hash new password (uses Argon2id)
password_hash = password_service.hash_password("SecurePassword123!")

# Verify password with automatic migration
result = password_service.verify_password("SecurePassword123!", password_hash)
if result.is_valid and result.needs_rehash:
    # Update user's password hash with new_hash for migration
    user.password_hash = result.new_hash
```

#### Email Verification & Password Reset (TokenService)
- **Email Verification**: 24-hour tamper-proof tokens
- **Password Reset**: 1-hour secure reset tokens
- **Account Activation**: 7-day activation tokens
- **Fernet Encryption**: AES-128 symmetric encryption with HMAC authentication
- **URL-Safe**: Base64-encoded for email/URL embedding

**Usage Example:**
```python
from src.infrastructure import create_token_service

token_service = create_token_service(encryption_key=b"your-32-byte-key")

# Generate email verification token
token = token_service.generate_email_verification_token(user.id, user.email.value)

# Verify token
user_id, email = token_service.verify_email_verification_token(token)

# Generate password reset token
reset_token = token_service.generate_password_reset_token(user.id, user.email.value)
user_id, email = token_service.verify_password_reset_token(reset_token)
```

#### Field-Level Encryption (EncryptionService)
- **PII/PHI Encryption**: HIPAA-compliant field-level encryption
- **Fernet Encryption**: AES-128 with automatic key rotation support
- **Individual Fields**: Encrypt/decrypt single sensitive fields
- **Batch Operations**: Encrypt multiple dictionary fields at once
- **Key Rotation**: Re-encrypt data with new keys

**Usage Example:**
```python
from src.infrastructure import create_encryption_service

encryption_service = create_encryption_service(encryption_key=b"your-32-byte-key")

# Encrypt sensitive field
encrypted_ssn = encryption_service.encrypt_field("123-45-6789")

# Decrypt field
ssn = encryption_service.decrypt_field(encrypted_ssn)

# Encrypt multiple fields in dictionary
user_data = {"name": "John", "ssn": "123-45-6789", "email": "john@example.com"}
encrypted_data = encryption_service.encrypt_dict_fields(user_data, ["ssn"])

# Key rotation
new_encrypted = encryption_service.rotate_encryption(encrypted_ssn, old_key=old_key)
```

## Business Rules

### User Account
1. Email must be unique and verified before activation
2. Accounts lock after 5 failed login attempts
3. Soft deletes preserve audit trail
4. Status transitions follow strict lifecycle

### Consent
1. Required consents: Terms of Service, Privacy Policy, Data Processing
2. HIPAA consents require audit metadata (IP, user agent)
3. Consent records are immutable (new record for changes)
4. Users can revoke consent at any time

## Installation

```bash
cd services/user-service
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

For coverage report:
```bash
pytest --cov=src --cov-report=html
```

## Environment Variables

### Database
- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: solace_users)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password (required)

### Redis
- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)

### Security
- `SECURITY_JWT_SECRET`: JWT secret key (required, min 32 chars)
- `SECURITY_JWT_EXPIRY_MINUTES`: Access token expiry in minutes (default: 15)
- `SECURITY_JWT_REFRESH_EXPIRY_DAYS`: Refresh token expiry in days (default: 30)
- `SECURITY_ENCRYPTION_KEY`: Fernet encryption key for PII/PHI (required, 32-byte base64)
- `SECURITY_PASSWORD_MIN_LENGTH`: Min password length (default: 8)
- `SECURITY_MAX_LOGIN_ATTEMPTS`: Max login attempts (default: 5)

**Generating Encryption Keys:**
```python
from cryptography.fernet import Fernet

# Generate JWT secret (save to SECURITY_JWT_SECRET)
import secrets
jwt_secret = secrets.token_urlsafe(32)

# Generate encryption key (save to SECURITY_ENCRYPTION_KEY)
encryption_key = Fernet.generate_key()
```

### Kafka
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka servers (default: localhost:9092)
- `KAFKA_TOPIC_USERS`: User events topic (default: solace.users)
- `KAFKA_ENABLE`: Enable Kafka (default: true)

## Domain Events

The service publishes the following domain events:

- `user.created` - User account created
- `user.updated` - User profile updated
- `user.deleted` - User account deleted (soft delete)
- `user.activated` - Account activated after verification
- `user.suspended` - Account suspended
- `user.email_verified` - Email verified
- `user.password_changed` - Password changed
- `user.preferences_updated` - Preferences updated
- `user.consent_granted` - Consent granted
- `user.consent_revoked` - Consent revoked
- `user.login_successful` - Successful login
- `user.login_failed` - Failed login attempt
- `user.account_locked` - Account locked

## Compliance

### HIPAA
- Audit trail for all consent actions
- IP address and user agent logging
- Clinical data sharing consent tracking

### GDPR
- Right to erasure (soft delete)
- Consent management with revocation
- Data processing consent tracking
- Audit logging for all data access

## Code Quality

- **Type Safety**: 100% type hints with Pydantic v2.12+
- **Test Coverage**: 94.30% coverage with 166 comprehensive tests
  - 114 domain layer tests
  - 52 infrastructure layer tests
  - Integration tests for all services
- **Security**:
  - Argon2id password hashing (Password Hashing Competition winner)
  - JWT authentication with access/refresh tokens
  - Field-level encryption for PII/PHI (HIPAA compliance)
  - Production-grade email validation with email-validator
- **Code Organization**:
  - Clean Architecture with strict layer boundaries
  - Immutable value objects and configuration
  - Factory functions for all services
  - Structured logging with structlog
  - Domain events for loose coupling

## License

Copyright © 2025 Solace-AI. All rights reserved.
