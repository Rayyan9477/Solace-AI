"""
Solace-AI Notification Service.

Multi-channel notification service supporting email, SMS, and push notifications.
Implements clean architecture with domain-driven design principles.

Architecture:
    - Domain Layer: Templates, channels, notification orchestration
    - Interface Layer: REST API endpoints
    - Infrastructure Layer: External service integrations (SMTP, SMS providers)

Usage:
    from src.domain import NotificationService, NotificationChannel
    from src.domain import NotificationTemplate, TemplateEngine
    from src.api import router as notification_router
"""
__version__ = "1.0.0"
