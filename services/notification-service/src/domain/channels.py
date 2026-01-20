"""
Solace-AI Notification Service - Notification Channels.

Multi-channel notification delivery supporting Email, SMS, and Push.
Implements Strategy Pattern for channel-agnostic notification dispatch.

Architecture Layer: Domain/Infrastructure
Principles: Strategy Pattern, Dependency Inversion, Async I/O
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
from pydantic import BaseModel, Field, EmailStr
import structlog

logger = structlog.get_logger(__name__)


class ChannelType(str, Enum):
    """Supported notification channel types."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class ChannelStatus(str, Enum):
    """Channel operational status."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class ChannelError(Exception):
    """Base exception for channel errors."""
    def __init__(self, channel_type: ChannelType, message: str) -> None:
        self.channel_type = channel_type
        super().__init__(f"[{channel_type.value}] {message}")


class DeliveryError(ChannelError):
    """Raised when notification delivery fails."""
    def __init__(self, channel_type: ChannelType, recipient: str, reason: str) -> None:
        self.recipient = recipient
        self.reason = reason
        super().__init__(channel_type, f"Failed to deliver to {recipient}: {reason}")


class DeliveryResult(BaseModel):
    """Result of a notification delivery attempt."""
    delivery_id: UUID = Field(default_factory=uuid4)
    channel_type: ChannelType
    recipient: str
    success: bool
    message_id: str | None = None
    error_message: str | None = None
    delivered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChannelConfig(BaseModel):
    """Base configuration for notification channels."""
    enabled: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)


class EmailConfig(ChannelConfig):
    """Email channel configuration."""
    smtp_host: str = Field(default="localhost")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = Field(default="")
    smtp_password: str = Field(default="")
    use_tls: bool = Field(default=True)
    from_email: str = Field(default="noreply@solace-ai.com")
    from_name: str = Field(default="Solace-AI")


class SMSConfig(ChannelConfig):
    """SMS channel configuration (Twilio-compatible)."""
    provider_url: str = Field(default="https://api.twilio.com/2010-04-01")
    account_sid: str = Field(default="")
    auth_token: str = Field(default="")
    from_number: str = Field(default="")


class PushConfig(ChannelConfig):
    """Push notification configuration (Firebase-compatible)."""
    firebase_url: str = Field(default="https://fcm.googleapis.com/fcm/send")
    server_key: str = Field(default="")
    project_id: str = Field(default="")


class NotificationChannel(ABC):
    """
    Abstract base class for notification channels.

    Implements Template Method pattern for consistent delivery flow.
    """
    def __init__(self, config: ChannelConfig) -> None:
        self._config = config
        self._status = ChannelStatus.ACTIVE if config.enabled else ChannelStatus.UNAVAILABLE

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type."""

    @property
    def status(self) -> ChannelStatus:
        """Return current channel status."""
        return self._status

    @property
    def is_available(self) -> bool:
        """Check if channel is available for delivery."""
        return self._status in (ChannelStatus.ACTIVE, ChannelStatus.DEGRADED)

    async def send(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """
        Send notification with retry logic.

        Args:
            recipient: Target recipient (email, phone, device token)
            subject: Notification subject/title
            body: Plain text body
            html_body: Optional HTML body (email only)
            metadata: Optional delivery metadata

        Returns:
            DeliveryResult with delivery status
        """
        if not self.is_available:
            return DeliveryResult(
                channel_type=self.channel_type,
                recipient=recipient,
                success=False,
                error_message=f"Channel {self.channel_type.value} is unavailable",
            )

        last_error: str | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                result = await self._deliver(recipient, subject, body, html_body, metadata or {})
                logger.info("notification_delivered",
                           channel=self.channel_type.value, recipient=recipient,
                           attempt=attempt + 1)
                return result
            except Exception as e:
                last_error = str(e)
                logger.warning("notification_delivery_failed",
                             channel=self.channel_type.value, recipient=recipient,
                             attempt=attempt + 1, error=last_error)
                if attempt < self._config.max_retries:
                    await asyncio.sleep(self._config.retry_delay_seconds * (attempt + 1))

        return DeliveryResult(
            channel_type=self.channel_type,
            recipient=recipient,
            success=False,
            error_message=f"Failed after {self._config.max_retries + 1} attempts: {last_error}",
        )

    @abstractmethod
    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Actual delivery implementation."""

    async def health_check(self) -> bool:
        """Check if channel is healthy."""
        return self.is_available


class EmailChannel(NotificationChannel):
    """Email notification channel using SMTP."""
    def __init__(self, config: EmailConfig) -> None:
        super().__init__(config)
        self._email_config = config

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.EMAIL

    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send email via SMTP."""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self._email_config.from_name} <{self._email_config.from_email}>"
        message["To"] = recipient

        message.attach(MIMEText(body, "plain", "utf-8"))
        if html_body:
            message.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            async with aiosmtplib.SMTP(
                hostname=self._email_config.smtp_host,
                port=self._email_config.smtp_port,
                use_tls=self._email_config.use_tls,
                timeout=self._email_config.timeout_seconds,
            ) as smtp:
                if self._email_config.smtp_username:
                    await smtp.login(
                        self._email_config.smtp_username,
                        self._email_config.smtp_password,
                    )
                response = await smtp.send_message(message)
                message_id = response[1] if response else None

            return DeliveryResult(
                channel_type=self.channel_type,
                recipient=recipient,
                success=True,
                message_id=str(message_id) if message_id else None,
                metadata=metadata,
            )
        except Exception as e:
            raise DeliveryError(self.channel_type, recipient, str(e)) from e


class SMSChannel(NotificationChannel):
    """SMS notification channel (Twilio-compatible API)."""
    def __init__(self, config: SMSConfig) -> None:
        super().__init__(config)
        self._sms_config = config

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SMS

    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send SMS via Twilio-compatible API."""
        sms_body = f"{subject}\n\n{body}" if subject else body
        sms_body = sms_body[:1600]  # SMS length limit

        url = f"{self._sms_config.provider_url}/Accounts/{self._sms_config.account_sid}/Messages.json"

        async with httpx.AsyncClient(timeout=self._sms_config.timeout_seconds) as client:
            try:
                response = await client.post(
                    url,
                    auth=(self._sms_config.account_sid, self._sms_config.auth_token),
                    data={
                        "From": self._sms_config.from_number,
                        "To": recipient,
                        "Body": sms_body,
                    },
                )
                response.raise_for_status()
                data = response.json()

                return DeliveryResult(
                    channel_type=self.channel_type,
                    recipient=recipient,
                    success=True,
                    message_id=data.get("sid"),
                    metadata={**metadata, "segments": data.get("num_segments", 1)},
                )
            except httpx.HTTPStatusError as e:
                raise DeliveryError(self.channel_type, recipient, f"HTTP {e.response.status_code}") from e
            except Exception as e:
                raise DeliveryError(self.channel_type, recipient, str(e)) from e


class PushChannel(NotificationChannel):
    """Push notification channel (Firebase-compatible)."""
    def __init__(self, config: PushConfig) -> None:
        super().__init__(config)
        self._push_config = config

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.PUSH

    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send push notification via Firebase-compatible API."""
        payload = {
            "to": recipient,
            "notification": {
                "title": subject,
                "body": body[:4096],  # FCM body limit
            },
            "data": metadata,
        }

        headers = {
            "Authorization": f"key={self._push_config.server_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._push_config.timeout_seconds) as client:
            try:
                response = await client.post(
                    self._push_config.firebase_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                success = data.get("success", 0) > 0
                return DeliveryResult(
                    channel_type=self.channel_type,
                    recipient=recipient,
                    success=success,
                    message_id=data.get("message_id"),
                    error_message=None if success else data.get("results", [{}])[0].get("error"),
                    metadata=metadata,
                )
            except httpx.HTTPStatusError as e:
                raise DeliveryError(self.channel_type, recipient, f"HTTP {e.response.status_code}") from e
            except Exception as e:
                raise DeliveryError(self.channel_type, recipient, str(e)) from e


class ChannelRegistry:
    """
    Registry of notification channels.

    Manages channel lifecycle and provides channel resolution.
    """
    def __init__(self) -> None:
        self._channels: dict[ChannelType, NotificationChannel] = {}
        logger.info("channel_registry_initialized")

    def register_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self._channels[channel.channel_type] = channel
        logger.info("channel_registered", channel_type=channel.channel_type.value,
                   status=channel.status.value)

    def get_channel(self, channel_type: ChannelType) -> NotificationChannel | None:
        """Get a channel by type."""
        return self._channels.get(channel_type)

    def get_available_channels(self) -> list[NotificationChannel]:
        """Get all available channels."""
        return [c for c in self._channels.values() if c.is_available]

    def list_channels(self) -> list[tuple[ChannelType, ChannelStatus]]:
        """List all registered channels with their status."""
        return [(c.channel_type, c.status) for c in self._channels.values()]

    async def health_check_all(self) -> dict[ChannelType, bool]:
        """Check health of all channels."""
        results = {}
        for channel_type, channel in self._channels.items():
            results[channel_type] = await channel.health_check()
        return results
