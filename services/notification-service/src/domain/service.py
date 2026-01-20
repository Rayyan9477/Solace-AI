"""
Solace-AI Notification Service - Notification Orchestration.

Core service that orchestrates notification delivery across multiple channels.
Handles routing, templating, and delivery coordination.

Architecture Layer: Domain
Principles: Facade Pattern, Event-Driven, Async Processing
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
import structlog

from .templates import (
    TemplateType,
    TemplateRegistry,
    RenderedTemplate,
    TemplateNotFoundError,
    TemplateRenderError,
)
from .channels import (
    ChannelType,
    ChannelRegistry,
    DeliveryResult,
    EmailChannel,
    SMSChannel,
    PushChannel,
    EmailConfig,
    SMSConfig,
    PushConfig,
)

logger = structlog.get_logger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    """Notification processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    PARTIALLY_DELIVERED = "partially_delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationRecipient(BaseModel):
    """Recipient information for notification delivery."""
    user_id: UUID | None = Field(default=None, description="User ID if registered user")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    device_token: str | None = Field(default=None, description="Push notification device token")
    name: str | None = Field(default=None, description="Display name")
    preferences: dict[str, Any] = Field(default_factory=dict, description="Channel preferences")

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
        return v

    def get_channel_target(self, channel_type: ChannelType) -> str | None:
        """Get the target address for a specific channel."""
        mapping = {
            ChannelType.EMAIL: self.email,
            ChannelType.SMS: self.phone,
            ChannelType.PUSH: self.device_token,
        }
        return mapping.get(channel_type)


class NotificationRequest(BaseModel):
    """Request to send a notification."""
    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    template_type: TemplateType = Field(..., description="Template to use")
    recipients: list[NotificationRecipient] = Field(..., min_length=1, description="Recipients")
    channels: list[ChannelType] = Field(default_factory=lambda: [ChannelType.EMAIL], description="Channels to use")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL)
    scheduled_at: datetime | None = Field(default=None, description="Scheduled delivery time")
    correlation_id: UUID | None = Field(default=None, description="Correlation ID for tracing")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class NotificationResult(BaseModel):
    """Result of notification processing."""
    request_id: UUID
    status: NotificationStatus
    template_type: TemplateType
    total_recipients: int
    successful_deliveries: int
    failed_deliveries: int
    delivery_results: list[DeliveryResult] = Field(default_factory=list)
    error_message: str | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: float = 0.0


class NotificationService:
    """
    Core notification service orchestrating template rendering and multi-channel delivery.

    Provides a unified interface for sending notifications across email, SMS, and push.
    Handles template resolution, variable substitution, and parallel delivery.
    """
    def __init__(
        self,
        template_registry: TemplateRegistry,
        channel_registry: ChannelRegistry,
    ) -> None:
        self._templates = template_registry
        self._channels = channel_registry
        logger.info("notification_service_initialized")

    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """
        Send a notification to all recipients across specified channels.

        Args:
            request: Notification request with template, recipients, and channels

        Returns:
            NotificationResult with delivery status for each recipient/channel
        """
        start_time = datetime.now(timezone.utc)
        logger.info("notification_send_started",
                   request_id=str(request.request_id),
                   template=request.template_type.value,
                   recipients=len(request.recipients),
                   channels=[c.value for c in request.channels])

        try:
            rendered = self._templates.render_template(
                request.template_type,
                request.variables,
            )
        except (TemplateNotFoundError, TemplateRenderError) as e:
            logger.error("notification_template_error",
                        request_id=str(request.request_id), error=str(e))
            return NotificationResult(
                request_id=request.request_id,
                status=NotificationStatus.FAILED,
                template_type=request.template_type,
                total_recipients=len(request.recipients),
                successful_deliveries=0,
                failed_deliveries=len(request.recipients),
                error_message=str(e),
            )

        delivery_tasks = []
        for recipient in request.recipients:
            for channel_type in request.channels:
                target = recipient.get_channel_target(channel_type)
                if target:
                    delivery_tasks.append(
                        self._deliver_to_channel(
                            channel_type=channel_type,
                            recipient=target,
                            rendered=rendered,
                            metadata={
                                "request_id": str(request.request_id),
                                "user_id": str(recipient.user_id) if recipient.user_id else None,
                                "priority": request.priority.value,
                            },
                        )
                    )

        if not delivery_tasks:
            logger.warning("notification_no_valid_targets",
                          request_id=str(request.request_id))
            return NotificationResult(
                request_id=request.request_id,
                status=NotificationStatus.FAILED,
                template_type=request.template_type,
                total_recipients=len(request.recipients),
                successful_deliveries=0,
                failed_deliveries=len(request.recipients),
                error_message="No valid delivery targets found",
            )

        results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
        delivery_results = []
        successful = 0
        failed = 0

        for result in results:
            if isinstance(result, DeliveryResult):
                delivery_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1
            elif isinstance(result, Exception):
                failed += 1
                logger.error("notification_delivery_exception",
                           request_id=str(request.request_id), error=str(result))

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if successful == 0:
            status = NotificationStatus.FAILED
        elif failed == 0:
            status = NotificationStatus.DELIVERED
        else:
            status = NotificationStatus.PARTIALLY_DELIVERED

        logger.info("notification_send_completed",
                   request_id=str(request.request_id),
                   status=status.value,
                   successful=successful,
                   failed=failed,
                   processing_time_ms=processing_time)

        return NotificationResult(
            request_id=request.request_id,
            status=status,
            template_type=request.template_type,
            total_recipients=len(request.recipients),
            successful_deliveries=successful,
            failed_deliveries=failed,
            delivery_results=delivery_results,
            processing_time_ms=processing_time,
        )

    async def _deliver_to_channel(
        self,
        channel_type: ChannelType,
        recipient: str,
        rendered: RenderedTemplate,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Deliver notification to a specific channel."""
        channel = self._channels.get_channel(channel_type)
        if not channel:
            logger.warning("notification_channel_not_found", channel_type=channel_type.value)
            return DeliveryResult(
                channel_type=channel_type,
                recipient=recipient,
                success=False,
                error_message=f"Channel {channel_type.value} not configured",
            )

        return await channel.send(
            recipient=recipient,
            subject=rendered.subject,
            body=rendered.body,
            html_body=rendered.html_body,
            metadata=metadata,
        )

    async def send_email(
        self,
        to_email: str,
        template_type: TemplateType,
        variables: dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> NotificationResult:
        """Convenience method for sending a single email."""
        request = NotificationRequest(
            template_type=template_type,
            recipients=[NotificationRecipient(email=to_email, name=variables.get("display_name"))],
            channels=[ChannelType.EMAIL],
            variables=variables,
            priority=priority,
        )
        return await self.send_notification(request)

    async def send_sms(
        self,
        to_phone: str,
        template_type: TemplateType,
        variables: dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> NotificationResult:
        """Convenience method for sending a single SMS."""
        request = NotificationRequest(
            template_type=template_type,
            recipients=[NotificationRecipient(phone=to_phone, name=variables.get("display_name"))],
            channels=[ChannelType.SMS],
            variables=variables,
            priority=priority,
        )
        return await self.send_notification(request)

    async def send_push(
        self,
        device_token: str,
        template_type: TemplateType,
        variables: dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> NotificationResult:
        """Convenience method for sending a single push notification."""
        request = NotificationRequest(
            template_type=template_type,
            recipients=[NotificationRecipient(device_token=device_token, name=variables.get("display_name"))],
            channels=[ChannelType.PUSH],
            variables=variables,
            priority=priority,
        )
        return await self.send_notification(request)

    async def send_clinician_alert(
        self,
        clinician_email: str,
        patient_name: str,
        patient_id: str,
        severity: str,
        alert_type: str,
        alert_details: str,
        dashboard_link: str,
    ) -> NotificationResult:
        """Send clinician alert notification."""
        return await self.send_email(
            to_email=clinician_email,
            template_type=TemplateType.CLINICIAN_ALERT,
            variables={
                "display_name": "Clinician",
                "severity": severity,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "alert_type": alert_type,
                "alert_details": alert_details,
                "dashboard_link": dashboard_link,
            },
            priority=NotificationPriority.HIGH,
        )

    async def send_crisis_escalation(
        self,
        clinician_email: str,
        patient_name: str,
        patient_id: str,
        risk_level: str,
        assessment_summary: str,
        dashboard_link: str,
    ) -> NotificationResult:
        """Send crisis escalation notification."""
        return await self.send_email(
            to_email=clinician_email,
            template_type=TemplateType.CRISIS_ESCALATION,
            variables={
                "patient_name": patient_name,
                "patient_id": patient_id,
                "risk_level": risk_level,
                "assessment_summary": assessment_summary,
                "dashboard_link": dashboard_link,
            },
            priority=NotificationPriority.CRITICAL,
        )

    def get_available_channels(self) -> list[ChannelType]:
        """Get list of available channels."""
        return [c.channel_type for c in self._channels.get_available_channels()]

    def get_available_templates(self) -> list[TemplateType]:
        """Get list of available templates."""
        return [t.template_type for t in self._templates.list_templates()]


def create_notification_service(
    email_config: EmailConfig | None = None,
    sms_config: SMSConfig | None = None,
    push_config: PushConfig | None = None,
) -> NotificationService:
    """
    Factory function to create a configured NotificationService.

    Args:
        email_config: Email channel configuration
        sms_config: SMS channel configuration
        push_config: Push notification configuration

    Returns:
        Configured NotificationService instance
    """
    template_registry = TemplateRegistry()
    channel_registry = ChannelRegistry()

    if email_config and email_config.enabled:
        channel_registry.register_channel(EmailChannel(email_config))

    if sms_config and sms_config.enabled:
        channel_registry.register_channel(SMSChannel(sms_config))

    if push_config and push_config.enabled:
        channel_registry.register_channel(PushChannel(push_config))

    return NotificationService(template_registry, channel_registry)
