# User Service - Domain Layer

Production-ready User Service following Clean Architecture and Domain-Driven Design principles.

## Overview

The User Service manages user accounts, authentication, preferences, and consent for the Solace-AI platform. This implementation follows HIPAA/GDPR compliance requirements and implements strict domain boundaries.

## Architecture

### Clean Architecture Layers

```
┌─────────────────────────────────────┐
│         Domain Layer                │
│  ├── Entities (User, UserPreferences)│
│  ├── Value Objects (UserRole, etc.) │
│  ├── Domain Services (ConsentService)│
│  └── Domain Events                  │
└─────────────────────────────────────┘
```

### Key Components

- **Entities**: `User`, `UserPreferences` with business rules and invariants
- **Value Objects**: `UserRole`, `AccountStatus`, `ConsentType`, `ConsentRecord`
- **Domain Services**: `ConsentService` for consent management
- **Events**: Domain events for event-driven architecture
- **Config**: Externalized configuration with environment variables

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
- `SECURITY_JWT_EXPIRY_MINUTES`: JWT expiry (default: 60)
- `SECURITY_PASSWORD_MIN_LENGTH`: Min password length (default: 8)
- `SECURITY_MAX_LOGIN_ATTEMPTS`: Max login attempts (default: 5)

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

- 100% type hints with Pydantic
- Comprehensive unit tests (80%+ coverage)
- Strict validation at all boundaries
- Structured logging with structlog
- Immutable value objects
- Domain events for loose coupling

## License

Copyright © 2025 Solace-AI. All rights reserved.
