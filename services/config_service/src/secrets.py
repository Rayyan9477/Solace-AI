"""
Solace-AI Secrets Management - Vault and AWS Secrets Manager Integration.
Enterprise-grade secrets management with caching, rotation, and audit logging.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class SecretProvider(str, Enum):
    """Secret storage provider types."""
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    LOCAL = "local"


class SecretMetadata(BaseModel):
    """Metadata for secrets."""
    name: str
    provider: SecretProvider
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1")
    rotation_enabled: bool = Field(default=False)
    rotation_days: int | None = Field(default=None)
    last_rotated: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    tags: dict[str, str] = Field(default_factory=dict)


class CachedSecret(BaseModel):
    """Cached secret with TTL."""
    value: SecretStr
    metadata: SecretMetadata
    cached_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = Field(default=300)

    @property
    def is_expired(self) -> bool:
        elapsed = (datetime.now(timezone.utc) - self.cached_at).total_seconds()
        return elapsed > self.ttl_seconds


class SecretsSettings(BaseSettings):
    """Secrets service configuration."""
    default_provider: SecretProvider = Field(default=SecretProvider.LOCAL)
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=0)
    vault_url: str = Field(default="http://localhost:8200")
    vault_token: SecretStr | None = Field(default=None)
    vault_mount_path: str = Field(default="secret")
    vault_namespace: str | None = Field(default=None)
    vault_tls_verify: bool = Field(default=True)
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_secrets_prefix: str = Field(default="solace/")
    local_secrets_path: str = Field(default="./secrets")
    encryption_key: SecretStr | None = Field(default=None)
    audit_enabled: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="SECRETS_", env_file=".env", extra="ignore"
    )


class SecretAccessEvent(BaseModel):
    """Audit event for secret access."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    secret_name: str
    operation: str
    provider: SecretProvider
    success: bool
    user_id: str | None = None
    service_name: str | None = None
    ip_address: str | None = None
    error_message: str | None = None


class SecretProviderBase(ABC):
    """Abstract base class for secret providers."""

    @abstractmethod
    async def get_secret(self, name: str, version: str | None = None) -> tuple[SecretStr, SecretMetadata]:
        """Retrieve a secret by name."""

    @abstractmethod
    async def set_secret(self, name: str, value: SecretStr, metadata: dict[str, Any] | None = None) -> SecretMetadata:
        """Store or update a secret."""

    @abstractmethod
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""

    @abstractmethod
    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        """List available secrets."""

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Check provider health status."""


class VaultProvider(SecretProviderBase):
    """HashiCorp Vault secrets provider."""

    def __init__(self, settings: SecretsSettings) -> None:
        self._settings = settings
        self._client: Any = None
        self._initialized = False

    async def _ensure_client(self) -> None:
        """Initialize Vault client if needed."""
        if self._initialized:
            return
        try:
            import hvac
            self._client = await asyncio.to_thread(
                hvac.Client,
                url=self._settings.vault_url,
                token=self._settings.vault_token.get_secret_value() if self._settings.vault_token else None,
                verify=self._settings.vault_tls_verify,
                namespace=self._settings.vault_namespace,
            )
            if not await asyncio.to_thread(self._client.is_authenticated):
                raise RuntimeError("Vault authentication failed")
            self._initialized = True
            logger.info("vault_client_initialized", url=self._settings.vault_url)
        except ImportError:
            raise RuntimeError("hvac package not installed. Install with: pip install hvac")

    async def get_secret(self, name: str, version: str | None = None) -> tuple[SecretStr, SecretMetadata]:
        await self._ensure_client()
        path = f"{self._settings.vault_mount_path}/data/{name}"
        response = await asyncio.to_thread(
            self._client.secrets.kv.v2.read_secret_version,
            path=name, mount_point=self._settings.vault_mount_path, version=int(version) if version else None
        )
        data = response["data"]["data"]
        secret_value = data.get("value", json.dumps(data))
        metadata = SecretMetadata(
            name=name, provider=SecretProvider.VAULT,
            version=str(response["data"]["metadata"]["version"]),
            created_at=datetime.fromisoformat(response["data"]["metadata"]["created_time"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(response["data"]["metadata"]["created_time"].replace("Z", "+00:00")),
        )
        return SecretStr(secret_value), metadata

    async def set_secret(self, name: str, value: SecretStr, metadata: dict[str, Any] | None = None) -> SecretMetadata:
        await self._ensure_client()
        data = {"value": value.get_secret_value()}
        if metadata:
            data.update(metadata)
        response = await asyncio.to_thread(
            self._client.secrets.kv.v2.create_or_update_secret,
            path=name, secret=data, mount_point=self._settings.vault_mount_path
        )
        return SecretMetadata(
            name=name, provider=SecretProvider.VAULT,
            version=str(response["data"]["version"]),
            updated_at=datetime.now(timezone.utc),
        )

    async def delete_secret(self, name: str) -> bool:
        await self._ensure_client()
        await asyncio.to_thread(
            self._client.secrets.kv.v2.delete_metadata_and_all_versions,
            path=name, mount_point=self._settings.vault_mount_path
        )
        return True

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        await self._ensure_client()
        path = prefix or ""
        response = await asyncio.to_thread(
            self._client.secrets.kv.v2.list_secrets, path=path, mount_point=self._settings.vault_mount_path)
        secrets: list[SecretMetadata] = []
        for key in response.get("data", {}).get("keys", []):
            secrets.append(SecretMetadata(name=f"{path}/{key}".strip("/"), provider=SecretProvider.VAULT))
        return secrets

    async def check_health(self) -> dict[str, Any]:
        try:
            await self._ensure_client()
            health = await asyncio.to_thread(self._client.sys.read_health_status, method="GET")
            return {"status": "healthy", "provider": "vault", "sealed": health.get("sealed", False)}
        except Exception as e:
            return {"status": "unhealthy", "provider": "vault", "error": str(e)}


class AWSSecretsManagerProvider(SecretProviderBase):
    """AWS Secrets Manager provider."""

    def __init__(self, settings: SecretsSettings) -> None:
        self._settings = settings
        self._client: Any = None
        self._initialized = False

    async def _ensure_client(self) -> None:
        """Initialize AWS Secrets Manager client."""
        if self._initialized:
            return
        try:
            import boto3
            kwargs: dict[str, Any] = {"region_name": self._settings.aws_region}
            if self._settings.aws_access_key_id and self._settings.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self._settings.aws_access_key_id.get_secret_value()
                kwargs["aws_secret_access_key"] = self._settings.aws_secret_access_key.get_secret_value()
            self._client = await asyncio.to_thread(boto3.client, "secretsmanager", **kwargs)
            self._initialized = True
            logger.info("aws_secrets_manager_initialized", region=self._settings.aws_region)
        except ImportError:
            raise RuntimeError("boto3 package not installed. Install with: pip install boto3")

    def _make_name(self, name: str) -> str:
        """Apply prefix to secret name."""
        return f"{self._settings.aws_secrets_prefix}{name}"

    async def get_secret(self, name: str, version: str | None = None) -> tuple[SecretStr, SecretMetadata]:
        await self._ensure_client()
        kwargs: dict[str, Any] = {"SecretId": self._make_name(name)}
        if version:
            kwargs["VersionId"] = version
        response = await asyncio.to_thread(self._client.get_secret_value, **kwargs)
        secret_value = response.get("SecretString", "")
        if not secret_value and "SecretBinary" in response:
            secret_value = base64.b64decode(response["SecretBinary"]).decode("utf-8")
        metadata = SecretMetadata(
            name=name, provider=SecretProvider.AWS_SECRETS_MANAGER,
            version=response.get("VersionId", "AWSCURRENT"),
            created_at=response.get("CreatedDate", datetime.now(timezone.utc)),
        )
        return SecretStr(secret_value), metadata

    async def set_secret(self, name: str, value: SecretStr, metadata: dict[str, Any] | None = None) -> SecretMetadata:
        await self._ensure_client()
        full_name = self._make_name(name)
        try:
            await asyncio.to_thread(self._client.describe_secret, SecretId=full_name)
            response = await asyncio.to_thread(self._client.put_secret_value, SecretId=full_name, SecretString=value.get_secret_value())
        except self._client.exceptions.ResourceNotFoundException:
            kwargs: dict[str, Any] = {"Name": full_name, "SecretString": value.get_secret_value()}
            if metadata and "description" in metadata:
                kwargs["Description"] = metadata["description"]
            if metadata and "tags" in metadata:
                kwargs["Tags"] = [{"Key": k, "Value": v} for k, v in metadata["tags"].items()]
            response = await asyncio.to_thread(self._client.create_secret, **kwargs)
        return SecretMetadata(
            name=name, provider=SecretProvider.AWS_SECRETS_MANAGER,
            version=response.get("VersionId", "AWSCURRENT"), updated_at=datetime.now(timezone.utc),
        )

    async def delete_secret(self, name: str) -> bool:
        await self._ensure_client()
        await asyncio.to_thread(self._client.delete_secret, SecretId=self._make_name(name), ForceDeleteWithoutRecovery=False)
        return True

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        await self._ensure_client()
        full_prefix = self._settings.aws_secrets_prefix + (prefix or "")
        paginator = self._client.get_paginator("list_secrets")
        secrets: list[SecretMetadata] = []
        def _paginate():
            results = []
            for page in paginator.paginate(Filters=[{"Key": "name", "Values": [full_prefix]}]):
                for secret in page.get("SecretList", []):
                    sname = secret["Name"].replace(self._settings.aws_secrets_prefix, "", 1)
                    results.append(SecretMetadata(
                        name=sname, provider=SecretProvider.AWS_SECRETS_MANAGER,
                        created_at=secret.get("CreatedDate", datetime.now(timezone.utc)),
                        updated_at=secret.get("LastChangedDate"),
                        rotation_enabled=secret.get("RotationEnabled", False),
                        tags={t["Key"]: t["Value"] for t in secret.get("Tags", [])},
                    ))
            return results
        return await asyncio.to_thread(_paginate)

    async def check_health(self) -> dict[str, Any]:
        try:
            await self._ensure_client()
            await asyncio.to_thread(self._client.list_secrets, MaxResults=1)
            return {"status": "healthy", "provider": "aws_secrets_manager", "region": self._settings.aws_region}
        except Exception as e:
            return {"status": "unhealthy", "provider": "aws_secrets_manager", "error": str(e)}


class LocalSecretsProvider(SecretProviderBase):
    """Local file-based secrets provider for development."""

    def __init__(self, settings: SecretsSettings) -> None:
        self._settings = settings
        self._secrets: dict[str, tuple[SecretStr, SecretMetadata]] = {}

    async def get_secret(self, name: str, version: str | None = None) -> tuple[SecretStr, SecretMetadata]:
        if name not in self._secrets:
            raise KeyError(f"Secret not found: {name}")
        return self._secrets[name]

    async def set_secret(self, name: str, value: SecretStr, metadata: dict[str, Any] | None = None) -> SecretMetadata:
        meta = SecretMetadata(name=name, provider=SecretProvider.LOCAL, updated_at=datetime.now(timezone.utc))
        self._secrets[name] = (value, meta)
        return meta

    async def delete_secret(self, name: str) -> bool:
        if name in self._secrets:
            del self._secrets[name]
            return True
        return False

    async def list_secrets(self, prefix: str | None = None) -> list[SecretMetadata]:
        result: list[SecretMetadata] = []
        for name, (_, meta) in self._secrets.items():
            if prefix is None or name.startswith(prefix):
                result.append(meta)
        return result

    async def check_health(self) -> dict[str, Any]:
        return {"status": "healthy", "provider": "local", "secret_count": len(self._secrets)}


class SecretsManager:
    """Unified secrets management with multi-provider support."""

    def __init__(self, settings: SecretsSettings | None = None) -> None:
        self._settings = settings or SecretsSettings()
        self._providers: dict[SecretProvider, SecretProviderBase] = {}
        self._cache: dict[str, CachedSecret] = {}
        self._audit_buffer: list[SecretAccessEvent] = []
        self._lock = asyncio.Lock()
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize available secret providers."""
        self._providers[SecretProvider.LOCAL] = LocalSecretsProvider(self._settings)
        if self._settings.vault_token:
            self._providers[SecretProvider.VAULT] = VaultProvider(self._settings)
        if self._settings.aws_access_key_id or self._settings.aws_region:
            self._providers[SecretProvider.AWS_SECRETS_MANAGER] = AWSSecretsManagerProvider(self._settings)

    def _get_provider(self, provider: SecretProvider | None = None) -> SecretProviderBase:
        """Get secret provider instance."""
        target = provider or self._settings.default_provider
        if target not in self._providers:
            raise ValueError(f"Provider not configured: {target}")
        return self._providers[target]

    async def get_secret(self, name: str, provider: SecretProvider | None = None,
                         version: str | None = None, bypass_cache: bool = False) -> SecretStr:
        """Retrieve a secret value."""
        cache_key = f"{provider or self._settings.default_provider}:{name}:{version or 'latest'}"
        if self._settings.cache_enabled and not bypass_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                return cached.value
        try:
            secret_provider = self._get_provider(provider)
            value, metadata = await secret_provider.get_secret(name, version)
            if self._settings.cache_enabled:
                self._cache[cache_key] = CachedSecret(
                    value=value, metadata=metadata, ttl_seconds=self._settings.cache_ttl_seconds
                )
            await self._audit("get", name, provider or self._settings.default_provider, True)
            return value
        except Exception as e:
            await self._audit("get", name, provider or self._settings.default_provider, False, str(e))
            raise

    async def set_secret(self, name: str, value: SecretStr, provider: SecretProvider | None = None,
                         metadata: dict[str, Any] | None = None) -> SecretMetadata:
        """Store or update a secret."""
        try:
            secret_provider = self._get_provider(provider)
            result = await secret_provider.set_secret(name, value, metadata)
            self._invalidate_cache(name, provider)
            await self._audit("set", name, provider or self._settings.default_provider, True)
            return result
        except Exception as e:
            await self._audit("set", name, provider or self._settings.default_provider, False, str(e))
            raise

    async def delete_secret(self, name: str, provider: SecretProvider | None = None) -> bool:
        """Delete a secret."""
        try:
            secret_provider = self._get_provider(provider)
            result = await secret_provider.delete_secret(name)
            self._invalidate_cache(name, provider)
            await self._audit("delete", name, provider or self._settings.default_provider, result)
            return result
        except Exception as e:
            await self._audit("delete", name, provider or self._settings.default_provider, False, str(e))
            raise

    async def list_secrets(self, prefix: str | None = None,
                           provider: SecretProvider | None = None) -> list[SecretMetadata]:
        """List available secrets."""
        secret_provider = self._get_provider(provider)
        return await secret_provider.list_secrets(prefix)

    async def check_health(self) -> dict[str, Any]:
        """Check health of all configured providers."""
        results: dict[str, Any] = {"status": "healthy", "providers": {}}
        for name, provider in self._providers.items():
            health = await provider.check_health()
            results["providers"][name.value] = health
            if health.get("status") != "healthy":
                results["status"] = "degraded"
        return results

    def _invalidate_cache(self, name: str, provider: SecretProvider | None) -> None:
        """Invalidate cached secret entries."""
        target = provider or self._settings.default_provider
        keys_to_remove = [k for k in self._cache if k.startswith(f"{target}:{name}:")]
        for key in keys_to_remove:
            del self._cache[key]

    async def _audit(self, operation: str, secret_name: str, provider: SecretProvider,
                     success: bool, error: str | None = None) -> None:
        """Record audit event."""
        if not self._settings.audit_enabled:
            return
        event = SecretAccessEvent(
            secret_name=secret_name, operation=operation, provider=provider, success=success, error_message=error
        )
        self._audit_buffer.append(event)
        if len(self._audit_buffer) > 10000:
            self._audit_buffer = self._audit_buffer[-5000:]
        log_method = logger.info if success else logger.warning
        log_method("secret_access", operation=operation, secret=secret_name, provider=provider.value, success=success)


def create_secrets_manager(settings: SecretsSettings | None = None) -> SecretsManager:
    """Factory function to create secrets manager."""
    return SecretsManager(settings)
