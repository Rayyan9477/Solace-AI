"""
Solace-AI Notification Service - REST API Endpoints.

FastAPI router for notification management including sending, templates, and health checks.

Architecture Layer: Interface/Adapter
Principles: REST, Input Validation, Structured Responses
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, EmailStr
import structlog

# Authentication dependencies from shared security library
try:
    from solace_security.middleware import (
        AuthenticatedUser,
        AuthenticatedService,
        get_current_user,
        get_current_user_optional,
        get_current_service,
        require_roles,
        require_permissions,
    )
    from solace_security import Role, Permission
    _AUTH_AVAILABLE = True
except ImportError:
    # Fallback for testing/development without security library
    from dataclasses import dataclass
    _AUTH_AVAILABLE = False

    @dataclass
    class AuthenticatedUser:
        user_id: UUID
        email: str
        roles: list[str]
        permissions: list[str]

    @dataclass
    class AuthenticatedService:
        service_id: str
        service_name: str
        permissions: list[str]

    async def get_current_user() -> AuthenticatedUser:
        raise HTTPException(status_code=501, detail="Authentication not configured")

    async def get_current_user_optional() -> Optional[AuthenticatedUser]:
        return None

    async def get_current_service() -> AuthenticatedService:
        raise HTTPException(status_code=501, detail="Service auth not configured")

    def require_roles(*roles):
        return get_current_user

    def require_permissions(*perms):
        return get_current_user

    class Role:
        ADMIN = "admin"
        CLINICIAN = "clinician"
        USER = "user"
        SERVICE = "service"

    class Permission:
        SEND_NOTIFICATION = "notification:send"

try:
    from .domain import (
        NotificationService,
        NotificationRequest,
        NotificationResult,
        NotificationRecipient,
        NotificationPriority,
        NotificationStatus,
        TemplateType,
        ChannelType,
        TemplateNotFoundError,
        TemplateRenderError,
    )
except ImportError:
    from domain import (
        NotificationService,
        NotificationRequest,
        NotificationResult,
        NotificationRecipient,
        NotificationPriority,
        NotificationStatus,
        TemplateType,
        ChannelType,
        TemplateNotFoundError,
        TemplateRenderError,
    )

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


class SendNotificationRequest(BaseModel):
    """API request to send a notification."""
    template_type: str = Field(..., description="Template type to use")
    recipients: list[dict[str, Any]] = Field(..., min_length=1, description="List of recipients")
    channels: list[str] = Field(default=["email"], description="Delivery channels")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    priority: str = Field(default="normal", description="Notification priority")
    correlation_id: str | None = Field(default=None, description="Correlation ID for tracing")

    model_config = {"json_schema_extra": {
        "example": {
            "template_type": "welcome",
            "recipients": [{"email": "user@example.com", "name": "John Doe"}],
            "channels": ["email"],
            "variables": {"display_name": "John", "getting_started_link": "https://app.solace-ai.com/start"},
            "priority": "normal",
        }
    }}


class SendEmailRequest(BaseModel):
    """API request to send a single email."""
    to_email: EmailStr = Field(..., description="Recipient email address")
    template_type: str = Field(..., description="Template type to use")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    priority: str = Field(default="normal", description="Notification priority")


class SendSMSRequest(BaseModel):
    """API request to send a single SMS."""
    to_phone: str = Field(..., min_length=10, description="Recipient phone number")
    template_type: str = Field(..., description="Template type to use")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    priority: str = Field(default="normal", description="Notification priority")


class SendPushRequest(BaseModel):
    """API request to send a single push notification."""
    device_token: str = Field(..., min_length=10, description="Device token")
    template_type: str = Field(..., description="Template type to use")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    priority: str = Field(default="normal", description="Notification priority")


class NotificationResponse(BaseModel):
    """API response for notification operations."""
    request_id: str
    status: str
    template_type: str
    total_recipients: int
    successful_deliveries: int
    failed_deliveries: int
    processed_at: str
    processing_time_ms: float
    error_message: str | None = None


class TemplateInfo(BaseModel):
    """Template information response."""
    template_type: str
    name: str
    description: str
    required_variables: list[str]
    is_active: bool


class ChannelInfo(BaseModel):
    """Channel information response."""
    channel_type: str
    status: str
    is_available: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    channels: dict[str, bool]


def _get_notification_service() -> NotificationService:
    """Dependency to get notification service instance."""
    try:
        from .main import get_notification_service
    except ImportError:
        from main import get_notification_service
    return get_notification_service()


def _result_to_response(result: NotificationResult) -> NotificationResponse:
    """Convert NotificationResult to API response."""
    return NotificationResponse(
        request_id=str(result.request_id),
        status=result.status.value,
        template_type=result.template_type.value,
        total_recipients=result.total_recipients,
        successful_deliveries=result.successful_deliveries,
        failed_deliveries=result.failed_deliveries,
        processed_at=result.processed_at.isoformat(),
        processing_time_ms=result.processing_time_ms,
        error_message=result.error_message,
    )


def _parse_template_type(template_type: str) -> TemplateType:
    """Parse and validate template type string."""
    try:
        return TemplateType(template_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid template type: {template_type}. Valid types: {[t.value for t in TemplateType]}",
        )


def _parse_channel_type(channel: str) -> ChannelType:
    """Parse and validate channel type string."""
    try:
        return ChannelType(channel)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid channel type: {channel}. Valid types: {[c.value for c in ChannelType]}",
        )


def _parse_priority(priority: str) -> NotificationPriority:
    """Parse and validate priority string."""
    try:
        return NotificationPriority(priority)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority: {priority}. Valid values: {[p.value for p in NotificationPriority]}",
        )


@router.post("/send", response_model=NotificationResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_notification(
    request: SendNotificationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """
    Send a notification to multiple recipients across channels.

    Supports email, SMS, and push notification channels.
    Requires admin, clinician role, or notification:send permission.
    """
    # Only admins, clinicians, or services can send bulk notifications
    if "admin" not in current_user.roles and "clinician" not in current_user.roles and "service" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to send notifications")
    logger.info("api_send_notification", template=request.template_type,
               recipients=len(request.recipients), authenticated_user=str(current_user.user_id))

    template_type = _parse_template_type(request.template_type)
    channels = [_parse_channel_type(c) for c in request.channels]
    priority = _parse_priority(request.priority)

    recipients = [
        NotificationRecipient(
            user_id=UUID(r["user_id"]) if r.get("user_id") else None,
            email=r.get("email"),
            phone=r.get("phone"),
            device_token=r.get("device_token"),
            name=r.get("name"),
        )
        for r in request.recipients
    ]

    notification_request = NotificationRequest(
        template_type=template_type,
        recipients=recipients,
        channels=channels,
        variables=request.variables,
        priority=priority,
        correlation_id=UUID(request.correlation_id) if request.correlation_id else None,
    )

    result = await service.send_notification(notification_request)
    return _result_to_response(result)


@router.post("/email", response_model=NotificationResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_email(
    request: SendEmailRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single email notification. Requires admin, clinician, or service role."""
    if "admin" not in current_user.roles and "clinician" not in current_user.roles and "service" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to send emails")
    logger.info("api_send_email", to=request.to_email, template=request.template_type,
                authenticated_user=str(current_user.user_id))

    template_type = _parse_template_type(request.template_type)
    priority = _parse_priority(request.priority)

    result = await service.send_email(
        to_email=request.to_email,
        template_type=template_type,
        variables=request.variables,
        priority=priority,
    )
    return _result_to_response(result)


@router.post("/sms", response_model=NotificationResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_sms(
    request: SendSMSRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single SMS notification. Requires admin, clinician, or service role."""
    if "admin" not in current_user.roles and "clinician" not in current_user.roles and "service" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to send SMS")
    logger.info("api_send_sms", to=request.to_phone, template=request.template_type,
                authenticated_user=str(current_user.user_id))

    template_type = _parse_template_type(request.template_type)
    priority = _parse_priority(request.priority)

    result = await service.send_sms(
        to_phone=request.to_phone,
        template_type=template_type,
        variables=request.variables,
        priority=priority,
    )
    return _result_to_response(result)


@router.post("/push", response_model=NotificationResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_push(
    request: SendPushRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single push notification. Requires admin, clinician, or service role."""
    if "admin" not in current_user.roles and "clinician" not in current_user.roles and "service" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions to send push notifications")
    logger.info("api_send_push", template=request.template_type,
                authenticated_user=str(current_user.user_id))

    template_type = _parse_template_type(request.template_type)
    priority = _parse_priority(request.priority)

    result = await service.send_push(
        device_token=request.device_token,
        template_type=template_type,
        variables=request.variables,
        priority=priority,
    )
    return _result_to_response(result)


@router.get("/templates", response_model=list[TemplateInfo])
async def list_templates(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> list[TemplateInfo]:
    """List all available notification templates. Requires authentication."""
    try:
        from .domain import TemplateRegistry
    except ImportError:
        from domain import TemplateRegistry
    registry = TemplateRegistry()
    templates = registry.list_templates()

    return [
        TemplateInfo(
            template_type=t.template_type.value,
            name=t.name,
            description=t.description,
            required_variables=t.required_variables,
            is_active=t.is_active,
        )
        for t in templates
    ]


@router.get("/templates/{template_type}", response_model=TemplateInfo)
async def get_template(
    template_type: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> TemplateInfo:
    """Get details of a specific template. Requires authentication."""
    try:
        from .domain import TemplateRegistry
    except ImportError:
        from domain import TemplateRegistry
    registry = TemplateRegistry()

    parsed_type = _parse_template_type(template_type)
    try:
        template = registry.get_template(parsed_type)
    except TemplateNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    return TemplateInfo(
        template_type=template.template_type.value,
        name=template.name,
        description=template.description,
        required_variables=template.required_variables,
        is_active=template.is_active,
    )


@router.get("/channels", response_model=list[ChannelInfo])
async def list_channels(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: NotificationService = Depends(_get_notification_service),
) -> list[ChannelInfo]:
    """List all configured notification channels. Requires admin role."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required to view channel configuration")
    try:
        from .domain import ChannelRegistry, ChannelStatus
    except ImportError:
        from domain import ChannelRegistry, ChannelStatus
    registry = ChannelRegistry()
    channels = registry.list_channels()

    return [
        ChannelInfo(
            channel_type=channel_type.value,
            status=status.value,
            is_available=status in (ChannelStatus.ACTIVE, ChannelStatus.DEGRADED),
        )
        for channel_type, status in channels
    ]


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: NotificationService = Depends(_get_notification_service),
) -> HealthResponse:
    """Check health of notification service and channels."""
    try:
        from .domain import ChannelRegistry
    except ImportError:
        from domain import ChannelRegistry
    registry = ChannelRegistry()
    channel_health = await registry.health_check_all()

    overall_status = "healthy" if any(channel_health.values()) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        channels={ct.value: healthy for ct, healthy in channel_health.items()},
    )
