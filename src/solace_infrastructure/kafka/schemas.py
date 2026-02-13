"""Solace-AI Schema Registry Management - Schema validation and evolution."""
from __future__ import annotations

import collections
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class SchemaType(str, Enum):
    AVRO, JSON, PROTOBUF = "AVRO", "JSON", "PROTOBUF"


class CompatibilityLevel(str, Enum):
    NONE, BACKWARD, BACKWARD_TRANSITIVE = "NONE", "BACKWARD", "BACKWARD_TRANSITIVE"
    FORWARD, FORWARD_TRANSITIVE, FULL, FULL_TRANSITIVE = "FORWARD", "FORWARD_TRANSITIVE", "FULL", "FULL_TRANSITIVE"


class SchemaFormat(str, Enum):
    STRING, BINARY = "string", "binary"


@dataclass
class SchemaVersion:
    schema_id: int
    version: int
    schema_str: str
    schema_type: SchemaType
    fingerprint: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SchemaMetadata:
    subject: str
    compatibility: CompatibilityLevel
    versions: list[int]
    latest_version: int
    schema_type: SchemaType


@dataclass
class CompatibilityResult:
    is_compatible: bool
    messages: list[str] = field(default_factory=list)
    breaking_changes: list[str] = field(default_factory=list)


class SchemaRegistrySettings(BaseSettings):
    url: str = Field(default="http://localhost:8081")
    username: str | None = Field(default=None)
    password: SecretStr | None = Field(default=None)
    ssl_ca_location: str | None = Field(default=None)
    ssl_certificate_location: str | None = Field(default=None)
    ssl_key_location: str | None = Field(default=None)
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    cache_capacity: int = Field(default=1000, ge=0)
    default_compatibility: CompatibilityLevel = Field(default=CompatibilityLevel.BACKWARD)
    model_config = SettingsConfigDict(env_prefix="SCHEMA_REGISTRY_", env_file=".env", extra="ignore")


class SchemaDefinition(BaseModel):
    subject: str = Field(..., min_length=1)
    schema_str: str = Field(..., min_length=2)
    schema_type: SchemaType = Field(default=SchemaType.JSON)
    references: list[dict[str, str]] = Field(default_factory=list)
    compatibility: CompatibilityLevel | None = Field(default=None)

    @property
    def fingerprint(self) -> str:
        normalized = json.dumps(json.loads(self.schema_str), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def to_registry_format(self) -> dict[str, Any]:
        return {"schema": self.schema_str, "schemaType": self.schema_type.value, "references": self.references}


class SchemaValidator(ABC):
    @abstractmethod
    def validate(self, schema_str: str) -> tuple[bool, str | None]: ...
    @abstractmethod
    def check_compatibility(self, new_schema: str, old_schema: str) -> CompatibilityResult: ...


class JsonSchemaValidator(SchemaValidator):
    def validate(self, schema_str: str) -> tuple[bool, str | None]:
        try:
            schema = json.loads(schema_str)
            if not isinstance(schema, dict):
                return False, "Schema must be a JSON object"
            if "type" not in schema and "$schema" not in schema and "properties" not in schema:
                return False, "Schema missing type definition"
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def check_compatibility(self, new_schema: str, old_schema: str) -> CompatibilityResult:
        try:
            new_obj, old_obj = json.loads(new_schema), json.loads(old_schema)
            breaking, messages = [], []
            old_props, new_props = old_obj.get("properties", {}), new_obj.get("properties", {})
            old_required, new_required = set(old_obj.get("required", [])), set(new_obj.get("required", []))
            for prop in old_props:
                if prop not in new_props:
                    breaking.append(f"Removed property: {prop}")
            for prop in new_required - old_required:
                if prop not in old_props:
                    breaking.append(f"New required property without default: {prop}")
            for prop in new_props:
                if prop not in old_props:
                    messages.append(f"Added property: {prop}")
            return CompatibilityResult(len(breaking) == 0, messages, breaking)
        except (json.JSONDecodeError, TypeError) as e:
            return CompatibilityResult(False, [], [f"Parse error: {e}"])


class AvroSchemaValidator(SchemaValidator):
    def validate(self, schema_str: str) -> tuple[bool, str | None]:
        try:
            schema = json.loads(schema_str)
            if "type" not in schema:
                return False, "Avro schema must have 'type' field"
            if schema.get("type") == "record":
                if "name" not in schema:
                    return False, "Avro record must have 'name' field"
                if "fields" not in schema or not isinstance(schema.get("fields"), list):
                    return False, "Avro record must have 'fields' array"
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def check_compatibility(self, new_schema: str, old_schema: str) -> CompatibilityResult:
        try:
            new_obj, old_obj = json.loads(new_schema), json.loads(old_schema)
            breaking, messages = [], []
            if new_obj.get("type") == "record" and old_obj.get("type") == "record":
                old_fields = {f["name"]: f for f in old_obj.get("fields", [])}
                new_fields = {f["name"]: f for f in new_obj.get("fields", [])}
                for name, old_f in old_fields.items():
                    if name not in new_fields and "default" not in old_f:
                        breaking.append(f"Removed field without default: {name}")
                for name, new_f in new_fields.items():
                    if name not in old_fields:
                        if "default" not in new_f:
                            breaking.append(f"New field without default: {name}")
                        else:
                            messages.append(f"Added optional field: {name}")
            return CompatibilityResult(len(breaking) == 0, messages, breaking)
        except (json.JSONDecodeError, TypeError) as e:
            return CompatibilityResult(False, [], [f"Parse error: {e}"])


class SchemaCache(Generic[T]):
    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._cache: collections.OrderedDict[str, tuple[T, datetime]] = collections.OrderedDict()

    def get(self, key: str) -> T | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        return None

    def put(self, key: str, value: T) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)  # O(1) LRU eviction
        self._cache[key] = (value, datetime.now(timezone.utc))

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


class SchemaRegistryAdapter:
    def __init__(self, settings: SchemaRegistrySettings) -> None:
        self._settings = settings
        self._session: Any = None
        self._cache: SchemaCache[SchemaVersion] = SchemaCache(settings.cache_capacity)

    async def connect(self) -> None:
        try:
            import aiohttp
            auth = aiohttp.BasicAuth(self._settings.username, self._settings.password.get_secret_value()) if self._settings.username and self._settings.password else None
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._settings.timeout_seconds), auth=auth)
            logger.info("schema_registry_connected", url=self._settings.url)
        except ImportError:
            logger.warning("aiohttp_not_available", fallback="mock_mode")
            self._session = None

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            logger.info("schema_registry_disconnected")

    async def register_schema(self, definition: SchemaDefinition) -> int:
        cached = self._cache.get(f"{definition.subject}:{definition.fingerprint}")
        if cached:
            return cached.schema_id
        if not self._session:
            schema_id = abs(hash(definition.fingerprint)) % 100000
            self._cache.put(f"{definition.subject}:{definition.fingerprint}",
                          SchemaVersion(schema_id, 1, definition.schema_str, definition.schema_type, definition.fingerprint))
            return schema_id
        try:
            url = f"{self._settings.url}/subjects/{definition.subject}/versions"
            async with self._session.post(url, json=definition.to_registry_format()) as resp:
                if resp.status not in (200, 201):
                    raise RuntimeError(f"Schema registration failed: {await resp.text()}")
                data = await resp.json()
                self._cache.put(f"{definition.subject}:{definition.fingerprint}",
                              SchemaVersion(data["id"], data.get("version", 1), definition.schema_str, definition.schema_type, definition.fingerprint))
                logger.info("schema_registered", subject=definition.subject, id=data["id"])
                return data["id"]
        except Exception as e:
            logger.error("schema_registration_failed", subject=definition.subject, error=str(e))
            raise

    async def get_schema_by_id(self, schema_id: int) -> SchemaVersion | None:
        for key, (cached, _) in list(self._cache._cache.items()):
            if cached.schema_id == schema_id:
                return cached
        if not self._session:
            return None
        try:
            async with self._session.get(f"{self._settings.url}/schemas/ids/{schema_id}") as resp:
                if resp.status == 404:
                    return None
                data = await resp.json()
                return SchemaVersion(schema_id, 0, data["schema"], SchemaType(data.get("schemaType", "JSON")),
                                    hashlib.sha256(data["schema"].encode()).hexdigest()[:16])
        except Exception as e:
            logger.error("schema_fetch_failed", id=schema_id, error=str(e))
            return None

    async def get_latest_schema(self, subject: str) -> SchemaVersion | None:
        if not self._session:
            return None
        try:
            async with self._session.get(f"{self._settings.url}/subjects/{subject}/versions/latest") as resp:
                if resp.status == 404:
                    return None
                data = await resp.json()
                return SchemaVersion(data["id"], data["version"], data["schema"], SchemaType(data.get("schemaType", "JSON")),
                                    hashlib.sha256(data["schema"].encode()).hexdigest()[:16])
        except Exception as e:
            logger.error("schema_fetch_failed", subject=subject, error=str(e))
            return None

    async def check_compatibility(self, subject: str, schema_str: str) -> bool:
        if not self._session:
            logger.debug("schema_registry_mock_mode", operation="check_compatibility", subject=subject)
            return True
        try:
            async with self._session.post(f"{self._settings.url}/compatibility/subjects/{subject}/versions/latest",
                                          json={"schema": schema_str}) as resp:
                return (await resp.json()).get("is_compatible", False)
        except Exception as e:
            logger.error("compatibility_check_failed", subject=subject, error=str(e))
            return False

    async def set_compatibility(self, subject: str, level: CompatibilityLevel) -> bool:
        if not self._session:
            logger.warning("schema_registry_unavailable", operation="set_compatibility", subject=subject)
            return False
        try:
            async with self._session.put(f"{self._settings.url}/config/{subject}", json={"compatibility": level.value}) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error("set_compatibility_failed", subject=subject, error=str(e))
            return False

    async def list_subjects(self) -> list[str]:
        if not self._session:
            logger.warning("schema_registry_unavailable", operation="list_subjects")
            return []
        try:
            async with self._session.get(f"{self._settings.url}/subjects") as resp:
                return await resp.json()
        except Exception as e:
            logger.error("list_subjects_failed", error=str(e))
            return []

    async def delete_subject(self, subject: str, permanent: bool = False) -> bool:
        if not self._session:
            logger.warning("schema_registry_unavailable", operation="delete_subject", subject=subject)
            return False
        try:
            url = f"{self._settings.url}/subjects/{subject}" + ("?permanent=true" if permanent else "")
            async with self._session.delete(url) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.error("delete_subject_failed", subject=subject, error=str(e))
            return False


class SchemaManager:
    def __init__(self, settings: SchemaRegistrySettings | None = None) -> None:
        self._settings = settings or SchemaRegistrySettings()
        self._adapter = SchemaRegistryAdapter(self._settings)
        self._validators: dict[SchemaType, SchemaValidator] = {SchemaType.JSON: JsonSchemaValidator(), SchemaType.AVRO: AvroSchemaValidator()}
        self._registered: dict[str, SchemaDefinition] = {}

    async def connect(self) -> None:
        await self._adapter.connect()

    async def close(self) -> None:
        await self._adapter.close()

    def get_validator(self, schema_type: SchemaType) -> SchemaValidator | None:
        return self._validators.get(schema_type)

    async def register(self, definition: SchemaDefinition) -> int:
        validator = self._validators.get(definition.schema_type)
        if validator:
            valid, error = validator.validate(definition.schema_str)
            if not valid:
                raise ValueError(f"Invalid schema: {error}")
        if definition.compatibility:
            await self._adapter.set_compatibility(definition.subject, definition.compatibility)
        schema_id = await self._adapter.register_schema(definition)
        self._registered[definition.subject] = definition
        logger.info("schema_registered", subject=definition.subject, id=schema_id)
        return schema_id

    async def validate_evolution(self, subject: str, new_schema: str, schema_type: SchemaType = SchemaType.JSON) -> CompatibilityResult:
        latest = await self._adapter.get_latest_schema(subject)
        if not latest:
            return CompatibilityResult(True, ["No existing schema"])
        validator = self._validators.get(schema_type)
        return validator.check_compatibility(new_schema, latest.schema_str) if validator else CompatibilityResult(True, ["No validator"])

    async def get_schema(self, schema_id: int) -> SchemaVersion | None:
        return await self._adapter.get_schema_by_id(schema_id)

    async def get_latest(self, subject: str) -> SchemaVersion | None:
        return await self._adapter.get_latest_schema(subject)

    async def list_subjects(self) -> list[str]:
        return await self._adapter.list_subjects()


async def create_schema_manager(settings: SchemaRegistrySettings | None = None) -> SchemaManager:
    manager = SchemaManager(settings)
    await manager.connect()
    return manager
