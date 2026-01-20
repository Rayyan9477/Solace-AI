"""
Solace-AI Notification Service - Domain Layer.

Core business logic for notification management including templates,
channels, and orchestration services.
"""
from .templates import (
    TemplateType,
    NotificationTemplate,
    TemplateRegistry,
    TemplateEngine,
    TemplateNotFoundError,
    TemplateRenderError,
)
from .channels import (
    NotificationChannel,
    ChannelType,
    ChannelStatus,
    DeliveryResult,
    EmailChannel,
    SMSChannel,
    PushChannel,
    ChannelRegistry,
    ChannelError,
    DeliveryError,
)
from .service import (
    NotificationService,
    NotificationRequest,
    NotificationResult,
    NotificationRecipient,
    NotificationPriority,
    NotificationStatus,
    create_notification_service,
)

__all__ = [
    # Templates
    "TemplateType",
    "NotificationTemplate",
    "TemplateRegistry",
    "TemplateEngine",
    "TemplateNotFoundError",
    "TemplateRenderError",
    # Channels
    "NotificationChannel",
    "ChannelType",
    "ChannelStatus",
    "DeliveryResult",
    "EmailChannel",
    "SMSChannel",
    "PushChannel",
    "ChannelRegistry",
    "ChannelError",
    "DeliveryError",
    # Service
    "NotificationService",
    "NotificationRequest",
    "NotificationResult",
    "NotificationRecipient",
    "NotificationPriority",
    "NotificationStatus",
    "create_notification_service",
]
