"""Solace-AI Seed Data Loader - Initial data population with dependency resolution."""
from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import re
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T")

# Pattern for valid SQL identifiers (letters, digits, underscore, starting with letter/underscore)
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, identifier_type: str = "identifier") -> None:
    """Validate that a string is a safe SQL identifier.

    Prevents SQL injection by ensuring names only contain alphanumeric
    characters and underscores, and start with a letter or underscore.

    Args:
        name: The identifier to validate.
        identifier_type: Description of what's being validated (for error message).

    Raises:
        ValueError: If the identifier contains invalid characters.
    """
    if not name:
        raise ValueError(f"Empty {identifier_type} is not allowed")
    if len(name) > 128:
        raise ValueError(f"{identifier_type} exceeds maximum length: {name}")
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Invalid {identifier_type}: {name}. Must contain only letters, digits, and underscores, starting with a letter or underscore.")


class Environment(str, Enum):
    """Deployment environments for seed data selection."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class SeedCategory(str, Enum):
    """Categories of seed data."""
    SYSTEM = "system"
    CLINICAL = "clinical"
    REFERENCE = "reference"
    TEST = "test"


class SeedSettings(BaseSettings):
    """Seed data configuration from environment."""
    database_url: str = Field(default="postgresql+asyncpg://solace:solace@localhost:5432/solace")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    force_reseed: bool = Field(default=False)
    validate_after_seed: bool = Field(default=True)
    batch_size: int = Field(default=100, ge=1, le=1000)
    model_config = SettingsConfigDict(env_prefix="SEED_", env_file=".env", extra="ignore")


@dataclass
class SeedResult:
    """Result of a seed operation."""
    category: SeedCategory
    table_name: str
    records_created: int
    records_skipped: int
    duration_ms: float
    success: bool
    error: str | None = None


@dataclass
class SeedBatch:
    """A batch of seed data for a single table."""
    table_name: str
    category: SeedCategory
    data: list[dict[str, Any]]
    unique_keys: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


class BaseSeedProvider(ABC, Generic[T]):
    """Abstract base class for seed data providers."""

    @property
    @abstractmethod
    def category(self) -> SeedCategory:
        pass

    @property
    @abstractmethod
    def table_name(self) -> str:
        pass

    @property
    def unique_keys(self) -> list[str]:
        return ["id"]

    @property
    def dependencies(self) -> list[str]:
        return []

    @abstractmethod
    def get_data(self, environment: Environment) -> list[dict[str, Any]]:
        pass


def _make_config(key: str, value: dict, desc: str) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "key": key, "value": value, "description": desc, "is_active": True}


class SystemConfigSeedProvider(BaseSeedProvider[dict[str, Any]]):
    """Provides system configuration seed data."""

    @property
    def category(self) -> SeedCategory:
        return SeedCategory.SYSTEM

    @property
    def table_name(self) -> str:
        return "system_configurations"

    @property
    def unique_keys(self) -> list[str]:
        return ["key"]

    def get_data(self, environment: Environment) -> list[dict[str, Any]]:
        configs = [
            _make_config("safety.crisis_detection_enabled", {"enabled": True, "threshold": 0.85}, "Crisis detection config"),
            _make_config("memory.consolidation_interval_hours", {"hours": 24, "batch_size": 100}, "Memory consolidation"),
            _make_config("therapy.session_timeout_minutes", {"timeout": 60, "warning_at": 55}, "Session timeout"),
            _make_config("llm.default_provider", {"provider": "anthropic", "model": "claude-3-sonnet"}, "Default LLM"),
        ]
        if environment == Environment.DEVELOPMENT:
            configs.append(_make_config("debug.verbose_logging", {"enabled": True, "level": "DEBUG"}, "Debug logging"))
        return configs


def _make_clinical(code: str, name: str, cat: str, desc: str) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "code": code, "name": name, "category": cat,
            "description": desc, "metadata": {"icd_10": code, "dsm_5_tr": True}}


class ClinicalReferenceSeedProvider(BaseSeedProvider[dict[str, Any]]):
    """Provides clinical reference data (DSM-5-TR codes)."""

    @property
    def category(self) -> SeedCategory:
        return SeedCategory.CLINICAL

    @property
    def table_name(self) -> str:
        return "clinical_references"

    @property
    def unique_keys(self) -> list[str]:
        return ["code"]

    def get_data(self, environment: Environment) -> list[dict[str, Any]]:
        return [
            _make_clinical("F32.0", "Major Depressive Disorder, Single Episode, Mild", "mood_disorders", "MDD mild"),
            _make_clinical("F32.1", "Major Depressive Disorder, Single Episode, Moderate", "mood_disorders", "MDD moderate"),
            _make_clinical("F41.1", "Generalized Anxiety Disorder", "anxiety_disorders", "GAD"),
            _make_clinical("F43.10", "Post-Traumatic Stress Disorder", "trauma_disorders", "PTSD"),
        ]


def _make_technique(code: str, name: str, modality: str, desc: str, contra: list, level: str) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "technique_code": code, "name": name, "modality": modality,
            "description": desc, "contraindications": contra, "evidence_level": level}


class TherapyTechniqueSeedProvider(BaseSeedProvider[dict[str, Any]]):
    """Provides therapy technique reference data."""

    @property
    def category(self) -> SeedCategory:
        return SeedCategory.CLINICAL

    @property
    def table_name(self) -> str:
        return "therapy_techniques"

    @property
    def unique_keys(self) -> list[str]:
        return ["technique_code"]

    def get_data(self, environment: Environment) -> list[dict[str, Any]]:
        return [
            _make_technique("CBT_COGNITIVE_RESTRUCTURING", "Cognitive Restructuring", "CBT",
                          "Identify and challenge negative thought patterns", ["acute_psychosis", "severe_dissociation"], "high"),
            _make_technique("DBT_DISTRESS_TOLERANCE", "Distress Tolerance Skills", "DBT",
                          "TIPP, ACCEPTS, and crisis survival skills", [], "high"),
            _make_technique("ACT_DEFUSION", "Cognitive Defusion", "ACT",
                          "Reduce attachment to unhelpful thoughts", [], "moderate"),
            _make_technique("MI_OPEN_QUESTIONS", "Open-Ended Questions", "MI",
                          "Evoke change talk through strategic questioning", [], "high"),
            _make_technique("MINDFULNESS_BREATHING", "Mindful Breathing", "MINDFULNESS",
                          "Focused attention on breath for grounding", ["hyperventilation_history"], "high"),
        ]


def _make_resource(code: str, name: str, rtype: str, contact: str, url: str, priority: int) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "resource_code": code, "name": name, "type": rtype,
            "contact": contact, "url": url, "available_24_7": True, "priority": priority}


class SafetyResourceSeedProvider(BaseSeedProvider[dict[str, Any]]):
    """Provides crisis resource reference data."""

    @property
    def category(self) -> SeedCategory:
        return SeedCategory.SYSTEM

    @property
    def table_name(self) -> str:
        return "safety_resources"

    @property
    def unique_keys(self) -> list[str]:
        return ["resource_code"]

    def get_data(self, environment: Environment) -> list[dict[str, Any]]:
        return [
            _make_resource("988_LIFELINE", "988 Suicide & Crisis Lifeline", "hotline", "988", "https://988lifeline.org", 1),
            _make_resource("CRISIS_TEXT", "Crisis Text Line", "text", "Text HOME to 741741", "https://www.crisistextline.org", 2),
            _make_resource("SAMHSA", "SAMHSA National Helpline", "hotline", "1-800-662-4357", "https://www.samhsa.gov/find-help/national-helpline", 3),
        ]


class SeedDataLoader:
    """Orchestrates seed data loading with dependency resolution."""

    def __init__(self, settings: SeedSettings | None = None, engine: AsyncEngine | None = None) -> None:
        self._settings = settings or SeedSettings()
        self._engine = engine
        self._session_factory: sessionmaker | None = None
        self._providers: list[BaseSeedProvider[Any]] = []

    async def initialize(self) -> None:
        if self._engine is None:
            self._engine = create_async_engine(self._settings.database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self._register_default_providers()
        logger.info("seed_loader_initialized", env=self._settings.environment.value)

    def _register_default_providers(self) -> None:
        self._providers = [SystemConfigSeedProvider(), SafetyResourceSeedProvider(),
                          ClinicalReferenceSeedProvider(), TherapyTechniqueSeedProvider()]

    def register_provider(self, provider: BaseSeedProvider[Any]) -> None:
        self._providers.append(provider)

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    def _resolve_dependencies(self) -> list[BaseSeedProvider[Any]]:
        table_to_provider = {p.table_name: p for p in self._providers}
        resolved: list[BaseSeedProvider[Any]] = []
        seen: set[str] = set()

        def visit(provider: BaseSeedProvider[Any]) -> None:
            if provider.table_name in seen:
                return
            seen.add(provider.table_name)
            for dep in provider.dependencies:
                if dep in table_to_provider:
                    visit(table_to_provider[dep])
            resolved.append(provider)

        for provider in self._providers:
            visit(provider)
        return resolved

    async def seed_all(self) -> list[SeedResult]:
        results: list[SeedResult] = []
        for provider in self._resolve_dependencies():
            result = await self._seed_provider(provider)
            results.append(result)
            if not result.success and not self._settings.force_reseed:
                logger.error("seed_provider_failed", table=provider.table_name, error=result.error)
                break
        return results

    async def _seed_provider(self, provider: BaseSeedProvider[Any]) -> SeedResult:
        start_time = asyncio.get_event_loop().time()
        data = provider.get_data(self._settings.environment)
        try:
            async with self._session_factory() as session:
                created, skipped = 0, 0
                for record in data:
                    if await self._record_exists(session, provider.table_name, provider.unique_keys, record):
                        if not self._settings.force_reseed:
                            skipped += 1
                            continue
                    await self._upsert_record(session, provider.table_name, provider.unique_keys, record)
                    created += 1
                await session.commit()
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info("seed_provider_completed", table=provider.table_name, created=created, skipped=skipped)
            return SeedResult(provider.category, provider.table_name, created, skipped, duration, True)
        except Exception as e:
            return SeedResult(provider.category, provider.table_name, 0, 0, 0, False, str(e))

    async def _record_exists(self, session: AsyncSession, table: str, keys: list[str], record: dict) -> bool:
        # Validate table and column names to prevent SQL injection
        _validate_identifier(table, "table name")
        for key in keys:
            _validate_identifier(key, "column name")

        conditions = " AND ".join(f"{k} = :{k}" for k in keys)
        result = await session.execute(text(f"SELECT 1 FROM {table} WHERE {conditions} LIMIT 1"),
                                       {k: record.get(k) for k in keys})
        return result.scalar() is not None

    async def _upsert_record(self, session: AsyncSession, table: str, keys: list[str], record: dict) -> None:
        # Validate table and column names to prevent SQL injection
        _validate_identifier(table, "table name")
        for key in keys:
            _validate_identifier(key, "column name")
        for col in record.keys():
            _validate_identifier(col, "column name")

        cols = ", ".join(record.keys())
        vals = ", ".join(f":{k}" for k in record.keys())
        conflict = ", ".join(keys)
        updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in record.keys() if k not in keys)
        await session.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({vals}) ON CONFLICT ({conflict}) DO UPDATE SET {updates}"), record)


async def create_seed_loader(settings: SeedSettings | None = None) -> SeedDataLoader:
    loader = SeedDataLoader(settings)
    await loader.initialize()
    return loader
