"""
Solace-AI Configuration Service API Endpoints.
RESTful API for configuration, secrets, and feature flag management.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from pydantic import BaseModel, Field, SecretStr
import structlog

from .settings import (
    ConfigurationManager, get_config_manager, ConfigEnvironment, ConfigValue
)
from .secrets import (
    SecretsManager, create_secrets_manager, SecretProvider, SecretMetadata
)
from .feature_flags import (
    FeatureFlagManager, get_feature_flag_manager, FeatureFlag, FlagStatus,
    RolloutStrategy, EvaluationResult, TargetingGroup, TargetingRule
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["configuration"])


class ConfigResponse(BaseModel):
    """Configuration response model."""
    key: str
    value: Any
    environment: str
    cached: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfigSectionResponse(BaseModel):
    """Configuration section response."""
    section: str
    data: dict[str, Any]
    environment: str


class SecretRequest(BaseModel):
    """Request model for setting secrets."""
    value: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class SecretResponse(BaseModel):
    """Secret metadata response (value never exposed)."""
    name: str
    provider: str
    version: str
    created_at: datetime
    updated_at: datetime | None = None


class FeatureFlagRequest(BaseModel):
    """Request model for creating/updating feature flags."""
    key: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    status: FlagStatus = Field(default=FlagStatus.DISABLED)
    strategy: RolloutStrategy = Field(default=RolloutStrategy.NONE)
    default_value: bool = Field(default=False)
    percentage: int = Field(default=0, ge=0, le=100)
    allowed_users: list[str] = Field(default_factory=list)
    blocked_users: list[str] = Field(default_factory=list)
    targeting_groups: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    expires_at: datetime | None = None


class FeatureFlagResponse(BaseModel):
    """Feature flag response model."""
    key: str
    name: str
    description: str
    status: FlagStatus
    strategy: RolloutStrategy
    default_value: bool
    percentage: int
    created_at: datetime
    updated_at: datetime
    owner: str | None
    tags: list[str]


class FlagEvaluationRequest(BaseModel):
    """Request model for flag evaluation."""
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class BulkEvaluationRequest(BaseModel):
    """Request model for bulk flag evaluation."""
    keys: list[str] = Field(..., min_length=1, max_length=100)
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    environment: str
    timestamp: datetime
    components: dict[str, Any]


def _get_config_manager() -> ConfigurationManager:
    """Dependency for configuration manager."""
    return get_config_manager()


def _get_feature_flags() -> FeatureFlagManager:
    """Dependency for feature flag manager."""
    return get_feature_flag_manager()


def _get_secrets_manager() -> SecretsManager:
    """Dependency for secrets manager."""
    return create_secrets_manager()


async def _verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Verify API key for protected endpoints."""
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
    return x_api_key


@router.get("/health", response_model=HealthResponse)
async def health_check(
    config: ConfigurationManager = Depends(_get_config_manager),
    secrets: SecretsManager = Depends(_get_secrets_manager),
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> HealthResponse:
    """Check service health and component status."""
    secrets_health = await secrets.check_health()
    return HealthResponse(
        status="healthy",
        service="config-service",
        environment=config.settings.environment.value,
        timestamp=datetime.now(timezone.utc),
        components={
            "configuration": {"status": "healthy", "loaded": config._initialized},
            "secrets": secrets_health,
            "feature_flags": {"status": "healthy", "flag_count": len(flags.flags)},
        },
    )


@router.get("/config/{key:path}", response_model=ConfigResponse)
async def get_config_value(
    key: str,
    config: ConfigurationManager = Depends(_get_config_manager),
) -> ConfigResponse:
    """Get configuration value by dot-notation key."""
    try:
        value = config.get(key)
        if value is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config key not found: {key}")
        return ConfigResponse(
            key=key, value=value, environment=config.environment.value, cached=True
        )
    except Exception as e:
        logger.error("config_get_error", key=key, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/config/section/{section}", response_model=ConfigSectionResponse)
async def get_config_section(
    section: str,
    config: ConfigurationManager = Depends(_get_config_manager),
) -> ConfigSectionResponse:
    """Get entire configuration section."""
    data = config.get_section(section)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Section not found: {section}")
    return ConfigSectionResponse(
        section=section, data=data, environment=config.environment.value
    )


@router.post("/config/reload")
async def reload_configuration(
    config: ConfigurationManager = Depends(_get_config_manager),
    _: str = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Reload configuration from all sources."""
    changed = await config.reload()
    return {"reloaded": True, "changed": changed, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/secrets", response_model=list[SecretResponse])
async def list_secrets(
    prefix: str | None = Query(None),
    provider: SecretProvider | None = Query(None),
    secrets: SecretsManager = Depends(_get_secrets_manager),
    _: str = Depends(_verify_api_key),
) -> list[SecretResponse]:
    """List available secrets (metadata only)."""
    items = await secrets.list_secrets(prefix, provider)
    return [
        SecretResponse(
            name=s.name, provider=s.provider.value, version=s.version,
            created_at=s.created_at, updated_at=s.updated_at
        ) for s in items
    ]


@router.get("/secrets/{name:path}", response_model=SecretResponse)
async def get_secret_metadata(
    name: str,
    provider: SecretProvider | None = Query(None),
    secrets: SecretsManager = Depends(_get_secrets_manager),
    _: str = Depends(_verify_api_key),
) -> SecretResponse:
    """Get secret metadata (value never returned via API)."""
    try:
        items = await secrets.list_secrets(name, provider)
        secret_meta = next((s for s in items if s.name == name), None)
        if not secret_meta:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Secret not found: {name}")
        return SecretResponse(
            name=secret_meta.name, provider=secret_meta.provider.value,
            version=secret_meta.version, created_at=secret_meta.created_at,
            updated_at=secret_meta.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("secret_get_error", name=name, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/secrets/{name:path}", response_model=SecretResponse)
async def set_secret(
    name: str,
    request: SecretRequest,
    provider: SecretProvider | None = Query(None),
    secrets: SecretsManager = Depends(_get_secrets_manager),
    _: str = Depends(_verify_api_key),
) -> SecretResponse:
    """Create or update a secret."""
    try:
        metadata = await secrets.set_secret(name, SecretStr(request.value), provider, request.metadata)
        return SecretResponse(
            name=metadata.name, provider=metadata.provider.value,
            version=metadata.version, created_at=metadata.created_at,
            updated_at=metadata.updated_at
        )
    except Exception as e:
        logger.error("secret_set_error", name=name, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/secrets/{name:path}")
async def delete_secret(
    name: str,
    provider: SecretProvider | None = Query(None),
    secrets: SecretsManager = Depends(_get_secrets_manager),
    _: str = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Delete a secret."""
    try:
        deleted = await secrets.delete_secret(name, provider)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Secret not found: {name}")
        return {"deleted": True, "name": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/flags", response_model=list[FeatureFlagResponse])
async def list_feature_flags(
    status_filter: FlagStatus | None = Query(None, alias="status"),
    tags: list[str] | None = Query(None),
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> list[FeatureFlagResponse]:
    """List all feature flags with optional filtering."""
    items = await flags.list_flags(tags, status_filter)
    return [
        FeatureFlagResponse(
            key=f.key, name=f.name, description=f.description, status=f.status,
            strategy=f.strategy, default_value=f.default_value, percentage=f.percentage,
            created_at=f.created_at, updated_at=f.updated_at, owner=f.owner, tags=f.tags
        ) for f in items
    ]


@router.get("/flags/{key}", response_model=FeatureFlagResponse)
async def get_feature_flag(
    key: str,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> FeatureFlagResponse:
    """Get feature flag by key."""
    flag = await flags.get_flag(key)
    if not flag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Flag not found: {key}")
    return FeatureFlagResponse(
        key=flag.key, name=flag.name, description=flag.description, status=flag.status,
        strategy=flag.strategy, default_value=flag.default_value, percentage=flag.percentage,
        created_at=flag.created_at, updated_at=flag.updated_at, owner=flag.owner, tags=flag.tags
    )


@router.post("/flags", response_model=FeatureFlagResponse, status_code=status.HTTP_201_CREATED)
async def create_feature_flag(
    request: FeatureFlagRequest,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
    _: str = Depends(_verify_api_key),
) -> FeatureFlagResponse:
    """Create a new feature flag."""
    existing = await flags.get_flag(request.key)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Flag already exists: {request.key}")
    targeting_groups = [TargetingGroup(**tg) for tg in request.targeting_groups]
    flag = FeatureFlag(
        key=request.key, name=request.name, description=request.description,
        status=request.status, strategy=request.strategy, default_value=request.default_value,
        percentage=request.percentage, allowed_users=request.allowed_users,
        blocked_users=request.blocked_users, targeting_groups=targeting_groups,
        metadata=request.metadata, tags=request.tags, owner=request.owner, expires_at=request.expires_at,
    )
    await flags.register_flag(flag)
    return FeatureFlagResponse(
        key=flag.key, name=flag.name, description=flag.description, status=flag.status,
        strategy=flag.strategy, default_value=flag.default_value, percentage=flag.percentage,
        created_at=flag.created_at, updated_at=flag.updated_at, owner=flag.owner, tags=flag.tags
    )


@router.put("/flags/{key}", response_model=FeatureFlagResponse)
async def update_feature_flag(
    key: str,
    request: FeatureFlagRequest,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
    _: str = Depends(_verify_api_key),
) -> FeatureFlagResponse:
    """Update an existing feature flag."""
    updates = request.model_dump(exclude_unset=True, exclude={"key"})
    if "targeting_groups" in updates:
        updates["targeting_groups"] = [TargetingGroup(**tg) for tg in updates["targeting_groups"]]
    flag = await flags.update_flag(key, updates)
    if not flag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Flag not found: {key}")
    return FeatureFlagResponse(
        key=flag.key, name=flag.name, description=flag.description, status=flag.status,
        strategy=flag.strategy, default_value=flag.default_value, percentage=flag.percentage,
        created_at=flag.created_at, updated_at=flag.updated_at, owner=flag.owner, tags=flag.tags
    )


@router.delete("/flags/{key}")
async def delete_feature_flag(
    key: str,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
    _: str = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Delete a feature flag."""
    deleted = await flags.delete_flag(key)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Flag not found: {key}")
    return {"deleted": True, "key": key}


@router.post("/flags/{key}/evaluate", response_model=EvaluationResult)
async def evaluate_feature_flag(
    key: str,
    request: FlagEvaluationRequest,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> EvaluationResult:
    """Evaluate a feature flag for specific user/context."""
    return await flags.evaluate(key, request.user_id, request.context)


@router.post("/flags/evaluate/bulk")
async def bulk_evaluate_flags(
    request: BulkEvaluationRequest,
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> dict[str, bool]:
    """Evaluate multiple feature flags at once."""
    return await flags.bulk_evaluate(request.keys, request.user_id, request.context)


@router.get("/flags/{key}/enabled")
async def check_flag_enabled(
    key: str,
    user_id: str | None = Query(None),
    flags: FeatureFlagManager = Depends(_get_feature_flags),
) -> dict[str, Any]:
    """Quick check if flag is enabled for user."""
    enabled = flags.is_enabled(key, user_id)
    return {"key": key, "enabled": enabled, "user_id": user_id}
