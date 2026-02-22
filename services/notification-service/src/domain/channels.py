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
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import re
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr
import httpx
from pydantic import BaseModel, Field, EmailStr
import structlog

logger = structlog.get_logger(__name__)


# Pattern for detecting header injection attempts (newlines and control characters)
_HEADER_INJECTION_PATTERN = re.compile(r'[\r\n\x00\x0b\x0c]')
# Pattern for basic email validation
_EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def _sanitize_header(value: str, max_length: int = 998) -> str:
    """
    Sanitize a string for safe use in email headers.

    Prevents email header injection attacks by removing newlines and control characters.

    Args:
        value: The header value to sanitize.
        max_length: Maximum length for the header value (RFC 5322 recommends 998).

    Returns:
        Sanitized string safe for use in email headers.
    """
    if not value:
        return ""
    # Remove newlines and control characters that could enable header injection
    sanitized = _HEADER_INJECTION_PATTERN.sub('', value)
    # Truncate to max length to prevent buffer issues
    return sanitized[:max_length].strip()


def _validate_email_address(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate.

    Returns:
        True if email appears valid, False otherwise.
    """
    if not email or len(email) > 254:  # RFC 5321 max length
        return False
    return _EMAIL_PATTERN.match(email) is not None


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
    """Push notification configuration (Firebase-compatible).

    Supports both legacy API (deprecated) and HTTP v1 API (recommended).
    For HTTP v1 API, provide service_account_file path and project_id.
    For legacy API (deprecated), provide server_key.
    """
    # HTTP v1 API settings (recommended)
    project_id: str = Field(default="", description="Firebase project ID for HTTP v1 API")
    service_account_file: str = Field(
        default="",
        description="Path to service account JSON file for HTTP v1 API authentication"
    )
    use_v1_api: bool = Field(
        default=True,
        description="Use HTTP v1 API (recommended). Set to False for legacy API."
    )

    # Legacy API settings (deprecated - for backward compatibility only)
    firebase_url: str = Field(
        default="https://fcm.googleapis.com/fcm/send",
        description="DEPRECATED: Legacy FCM endpoint"
    )
    server_key: str = Field(
        default="",
        description="DEPRECATED: Server key for legacy API. Use service_account_file instead."
    )


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
        """Send email via SMTP with header injection protection."""
        # Validate recipient email address
        sanitized_recipient = _sanitize_header(recipient)
        if not _validate_email_address(sanitized_recipient):
            raise DeliveryError(
                self.channel_type,
                recipient,
                f"Invalid email address format: {recipient[:50]}..."
            )

        # Sanitize all header values to prevent header injection attacks
        sanitized_subject = _sanitize_header(subject, max_length=200)
        sanitized_from_name = _sanitize_header(self._email_config.from_name, max_length=100)
        sanitized_from_email = _sanitize_header(self._email_config.from_email)

        # Build message with sanitized headers
        message = MIMEMultipart("alternative")
        # Use Header class for proper RFC 2047 encoding of non-ASCII characters
        message["Subject"] = Header(sanitized_subject, "utf-8")
        # Use formataddr to properly format the From header
        message["From"] = formataddr((sanitized_from_name, sanitized_from_email))
        message["To"] = sanitized_recipient

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
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._sms_config.timeout_seconds)
        return self._client

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SMS

    @staticmethod
    def _format_crisis_sms(metadata: dict[str, Any]) -> str:
        """Format crisis SMS to fit within 160 chars.

        Prioritizes safety resources (988 hotline) FIRST, then risk level and
        patient ID. Crisis resources must NEVER be truncated.
        """
        risk_level = metadata.get("risk_level", metadata.get("variables", {}).get("risk_level", ""))
        patient_id = metadata.get("patient_id", metadata.get("variables", {}).get("patient_id", ""))

        # Crisis resources are mandatory and placed first — never truncated
        crisis_resource = "988 Lifeline: call/text 988"
        header = f"CRISIS {risk_level}"
        if patient_id:
            header += f" PT:{patient_id[:8]}"

        # Build message with crisis resource always first
        msg = f"{header} | {crisis_resource}"

        # Only add dashboard link if space remains
        dashboard = metadata.get("dashboard_link", metadata.get("variables", {}).get("dashboard_link", ""))
        if dashboard and len(msg) + len(dashboard) + 1 <= 160:
            msg += " " + dashboard

        return msg[:160]

    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send SMS via Twilio-compatible API.

        For crisis notifications (detected via metadata), formats a compact
        message fitting within 160 chars that prioritizes risk_level, patient_id,
        first trigger keyword, and dashboard link.
        """
        is_crisis = metadata.get("template_type") in ("crisis_alert", "crisis_escalation") or metadata.get("is_crisis")
        if is_crisis:
            sms_body = self._format_crisis_sms(metadata)
        else:
            sms_body = f"{subject}\n\n{body}" if subject else body
            sms_body = sms_body[:1600]  # SMS length limit

        url = f"{self._sms_config.provider_url}/Accounts/{self._sms_config.account_sid}/Messages.json"

        client = await self._get_client()
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
    """
    Push notification channel supporting Firebase Cloud Messaging (FCM).

    Supports both HTTP v1 API (recommended) and legacy API (deprecated).
    HTTP v1 API uses OAuth 2.0 authentication with service account credentials.
    """
    def __init__(self, config: PushConfig) -> None:
        super().__init__(config)
        self._push_config = config
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._client: httpx.AsyncClient | None = None
        self._token_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._push_config.timeout_seconds)
        return self._client

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.PUSH

    async def _get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token for FCM HTTP v1 API.

        Uses Google service account credentials to obtain an access token.
        Tokens are cached until near expiry.
        """
        # Check if we have a valid cached token (fast path, no lock)
        if self._access_token and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                return self._access_token

        async with self._token_lock:
            # Double-check after acquiring lock
            if self._access_token and self._token_expiry:
                if datetime.now(timezone.utc) < self._token_expiry:
                    return self._access_token

            if not self._push_config.service_account_file:
                raise DeliveryError(
                    self.channel_type,
                    "service_account",
                    "service_account_file is required for HTTP v1 API"
                )

            try:
                # Try to use google-auth library for OAuth2
                from google.oauth2 import service_account
                from google.auth.transport.requests import Request

                credentials = service_account.Credentials.from_service_account_file(
                    self._push_config.service_account_file,
                    scopes=['https://www.googleapis.com/auth/firebase.messaging']
                )
                credentials.refresh(Request())

                self._access_token = credentials.token
                # Set expiry to 5 minutes before actual expiry for safety margin
                if credentials.expiry:
                    self._token_expiry = credentials.expiry.replace(tzinfo=timezone.utc) - timedelta(minutes=5)
                else:
                    # Default 55 minute expiry (FCM tokens last 1 hour)
                    self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=55)

                return self._access_token

            except ImportError:
                logger.warning(
                    "google_auth_not_installed",
                    message="google-auth library not installed. Install with: pip install google-auth"
                )
                raise DeliveryError(
                    self.channel_type,
                    "google_auth",
                    "google-auth library required for HTTP v1 API. Install with: pip install google-auth"
                )
            except Exception as e:
                raise DeliveryError(self.channel_type, "auth", f"Failed to get access token: {e}") from e

    async def _deliver(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: str | None,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send push notification via FCM."""
        if self._push_config.use_v1_api:
            return await self._deliver_v1(recipient, subject, body, metadata)
        else:
            return await self._deliver_legacy(recipient, subject, body, metadata)

    async def _deliver_v1(
        self,
        recipient: str,
        subject: str,
        body: str,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """Send push notification via FCM HTTP v1 API (recommended)."""
        if not self._push_config.project_id:
            raise DeliveryError(
                self.channel_type,
                recipient,
                "project_id is required for HTTP v1 API"
            )

        access_token = await self._get_access_token()

        # FCM HTTP v1 API endpoint
        url = f"https://fcm.googleapis.com/v1/projects/{self._push_config.project_id}/messages:send"

        # HTTP v1 API payload format
        payload = {
            "message": {
                "token": recipient,
                "notification": {
                    "title": subject[:100],  # v1 API title limit
                    "body": body[:4096],  # v1 API body limit
                },
                "data": {k: str(v) for k, v in metadata.items()},  # v1 API requires string values
            }
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        client = await self._get_client()
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # v1 API returns message name on success
            message_name = data.get("name", "")
            return DeliveryResult(
                channel_type=self.channel_type,
                recipient=recipient,
                success=True,
                message_id=message_name.split("/")[-1] if message_name else None,
                metadata=metadata,
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            raise DeliveryError(
                self.channel_type,
                recipient,
                f"HTTP {e.response.status_code}: {error_detail}"
            ) from e
        except Exception as e:
            raise DeliveryError(self.channel_type, recipient, str(e)) from e

    async def _deliver_legacy(
        self,
        recipient: str,
        subject: str,
        body: str,
        metadata: dict[str, Any],
    ) -> DeliveryResult:
        """
        Send push notification via legacy FCM API.

        DEPRECATED: This method uses the legacy FCM API which will be removed by Google.
        Migrate to HTTP v1 API by setting use_v1_api=True and providing service_account_file.
        """
        logger.warning(
            "fcm_legacy_api_deprecated",
            message="Using deprecated FCM legacy API. Please migrate to HTTP v1 API."
        )

        if not self._push_config.server_key:
            raise DeliveryError(
                self.channel_type,
                recipient,
                "server_key is required for legacy API"
            )

        payload = {
            "to": recipient,
            "notification": {
                "title": subject,
                "body": body[:4096],
            },
            "data": metadata,
        }

        headers = {
            "Authorization": f"key={self._push_config.server_key}",
            "Content-Type": "application/json",
        }

        client = await self._get_client()
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
