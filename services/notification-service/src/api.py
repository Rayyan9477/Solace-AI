"""
Solace-AI Notification Service - REST API Endpoints.

FastAPI router for notification management including sending, templates, and health checks.

Architecture Layer: Interface/Adapter
Principles: REST, Input Validation, Structured Responses
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, EmailStr
import structlog

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
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """
    Send a notification to multiple recipients across channels.

    Supports email, SMS, and push notification channels.
    """
    logger.info("api_send_notification", template=request.template_type,
               recipients=len(request.recipients))

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
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single email notification."""
    logger.info("api_send_email", to=request.to_email, template=request.template_type)

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
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single SMS notification."""
    logger.info("api_send_sms", to=request.to_phone, template=request.template_type)

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
    service: NotificationService = Depends(_get_notification_service),
) -> NotificationResponse:
    """Send a single push notification."""
    logger.info("api_send_push", template=request.template_type)

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
    service: NotificationService = Depends(_get_notification_service),
) -> list[TemplateInfo]:
    """List all available notification templates."""
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
async def get_template(template_type: str) -> TemplateInfo:
    """Get details of a specific template."""
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
    service: NotificationService = Depends(_get_notification_service),
) -> list[ChannelInfo]:
    """List all configured notification channels."""
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
