"""
Solace-AI Contraindication Database - PostgreSQL repository for contraindication rules.
Provides async database operations with connection pooling and caching.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

logger = structlog.get_logger(__name__)


class DatabaseConfig(BaseSettings):
    """Configuration for PostgreSQL database connection."""

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    database: str = Field(default="solace", description="Database name")
    user: str = Field(default="solace", description="Database user")
    password: SecretStr = Field(default="", description="Database password")
    min_pool_size: int = Field(default=2, ge=1, le=20, description="Minimum connection pool size")
    max_pool_size: int = Field(default=10, ge=1, le=100, description="Maximum connection pool size")
    connection_timeout: int = Field(default=30, ge=5, le=120, description="Connection timeout seconds")
    command_timeout: int = Field(default=60, ge=5, le=300, description="Command timeout seconds")
    enable_cache: bool = Field(default=True, description="Enable in-memory rule caching")
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")

    model_config = SettingsConfigDict(
        env_prefix="SAFETY_DB_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class ContraindicationRuleRecord:
    """Database record for contraindication rule."""

    id: UUID
    technique: str
    condition: str
    contraindication_type: str
    severity: Decimal
    rationale: str
    is_active: bool
    alternatives: list[str]
    prerequisites: list[str]
    created_at: datetime
    updated_at: datetime
    version: int


class ContraindicationRepository:
    """
    PostgreSQL repository for contraindication rules.
    Provides async operations with connection pooling and optional caching.
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        """Initialize database repository."""
        self._config = config or DatabaseConfig()
        self._pool: asyncpg.Pool | None = None
        self._cache: dict[str, list[ContraindicationRuleRecord]] = {}
        self._cache_timestamp: datetime | None = None
        self._initialized = False

        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg_unavailable", message="PostgreSQL operations will be unavailable")

    async def connect(self) -> bool:
        """
        Initialize database connection pool.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not ASYNCPG_AVAILABLE:
            logger.error("asyncpg_not_installed", message="Install asyncpg: pip install asyncpg")
            return False

        try:
            self._pool = await asyncpg.create_pool(
                host=self._config.host,
                port=self._config.port,
                database=self._config.database,
                user=self._config.user,
                password=self._config.password.get_secret_value(),
                min_size=self._config.min_pool_size,
                max_size=self._config.max_pool_size,
                command_timeout=self._config.command_timeout,
                timeout=self._config.connection_timeout,
            )
            self._initialized = True
            logger.info(
                "database_connected",
                host=self._config.host,
                database=self._config.database,
                pool_size=f"{self._config.min_pool_size}-{self._config.max_pool_size}"
            )
            return True

        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            self._initialized = False
            return False

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("database_disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if database is connected and ready."""
        return self._initialized and self._pool is not None

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._config.enable_cache or not self._cache_timestamp:
            return False

        elapsed = (datetime.now(timezone.utc) - self._cache_timestamp).total_seconds()
        return elapsed < self._config.cache_ttl_seconds

    def _invalidate_cache(self) -> None:
        """Invalidate the rule cache."""
        self._cache.clear()
        self._cache_timestamp = None

    async def find_by_technique(
        self,
        technique: str,
        use_cache: bool = True
    ) -> list[ContraindicationRuleRecord]:
        """
        Find all contraindication rules for a specific technique.

        Args:
            technique: Therapy technique name
            use_cache: Whether to use cached results

        Returns:
            List of contraindication rules for the technique
        """
        cache_key = f"technique:{technique}"

        if use_cache and self._is_cache_valid() and cache_key in self._cache:
            logger.debug("cache_hit", key=cache_key)
            return self._cache[cache_key]

        if not self.is_connected:
            logger.warning("database_not_connected", operation="find_by_technique")
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        r.id, r.technique, r.condition, r.contraindication_type,
                        r.severity, r.rationale, r.is_active, r.created_at,
                        r.updated_at, r.version,
                        COALESCE(
                            array_agg(DISTINCT a.alternative_technique) FILTER (WHERE a.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as alternatives,
                        COALESCE(
                            array_agg(DISTINCT p.prerequisite) FILTER (WHERE p.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as prerequisites
                    FROM contraindication_rules r
                    LEFT JOIN rule_alternatives a ON r.id = a.rule_id
                    LEFT JOIN rule_prerequisites p ON r.id = p.rule_id
                    WHERE r.technique = $1 AND r.is_active = TRUE
                    GROUP BY r.id
                    ORDER BY r.severity DESC
                    """,
                    technique
                )

                rules = [self._map_row_to_record(row) for row in rows]

                if use_cache:
                    self._cache[cache_key] = rules
                    self._cache_timestamp = datetime.now(timezone.utc)

                logger.debug("rules_fetched", technique=technique, count=len(rules))
                return rules

        except Exception as e:
            logger.error("find_by_technique_failed", technique=technique, error=str(e))
            return []

    async def find_by_condition(
        self,
        condition: str,
        use_cache: bool = True
    ) -> list[ContraindicationRuleRecord]:
        """
        Find all contraindication rules for a specific condition.

        Args:
            condition: Mental health condition name
            use_cache: Whether to use cached results

        Returns:
            List of contraindication rules for the condition
        """
        cache_key = f"condition:{condition}"

        if use_cache and self._is_cache_valid() and cache_key in self._cache:
            logger.debug("cache_hit", key=cache_key)
            return self._cache[cache_key]

        if not self.is_connected:
            logger.warning("database_not_connected", operation="find_by_condition")
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        r.id, r.technique, r.condition, r.contraindication_type,
                        r.severity, r.rationale, r.is_active, r.created_at,
                        r.updated_at, r.version,
                        COALESCE(
                            array_agg(DISTINCT a.alternative_technique) FILTER (WHERE a.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as alternatives,
                        COALESCE(
                            array_agg(DISTINCT p.prerequisite) FILTER (WHERE p.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as prerequisites
                    FROM contraindication_rules r
                    LEFT JOIN rule_alternatives a ON r.id = a.rule_id
                    LEFT JOIN rule_prerequisites p ON r.id = p.rule_id
                    WHERE r.condition = $1 AND r.is_active = TRUE
                    GROUP BY r.id
                    ORDER BY r.severity DESC
                    """,
                    condition
                )

                rules = [self._map_row_to_record(row) for row in rows]

                if use_cache:
                    self._cache[cache_key] = rules
                    self._cache_timestamp = datetime.now(timezone.utc)

                logger.debug("rules_fetched", condition=condition, count=len(rules))
                return rules

        except Exception as e:
            logger.error("find_by_condition_failed", condition=condition, error=str(e))
            return []

    async def find_one(
        self,
        technique: str,
        condition: str
    ) -> ContraindicationRuleRecord | None:
        """
        Find a specific contraindication rule by technique and condition.

        Args:
            technique: Therapy technique name
            condition: Mental health condition name

        Returns:
            The contraindication rule if found, None otherwise
        """
        if not self.is_connected:
            logger.warning("database_not_connected", operation="find_one")
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        r.id, r.technique, r.condition, r.contraindication_type,
                        r.severity, r.rationale, r.is_active, r.created_at,
                        r.updated_at, r.version,
                        COALESCE(
                            array_agg(DISTINCT a.alternative_technique) FILTER (WHERE a.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as alternatives,
                        COALESCE(
                            array_agg(DISTINCT p.prerequisite) FILTER (WHERE p.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as prerequisites
                    FROM contraindication_rules r
                    LEFT JOIN rule_alternatives a ON r.id = a.rule_id
                    LEFT JOIN rule_prerequisites p ON r.id = p.rule_id
                    WHERE r.technique = $1 AND r.condition = $2 AND r.is_active = TRUE
                    GROUP BY r.id
                    """,
                    technique, condition
                )

                if row:
                    return self._map_row_to_record(row)
                return None

        except Exception as e:
            logger.error("find_one_failed", technique=technique, condition=condition, error=str(e))
            return None

    async def find_all_active(self, use_cache: bool = True) -> list[ContraindicationRuleRecord]:
        """
        Find all active contraindication rules.

        Args:
            use_cache: Whether to use cached results

        Returns:
            List of all active contraindication rules
        """
        cache_key = "all_active"

        if use_cache and self._is_cache_valid() and cache_key in self._cache:
            logger.debug("cache_hit", key=cache_key)
            return self._cache[cache_key]

        if not self.is_connected:
            logger.warning("database_not_connected", operation="find_all_active")
            return []

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        r.id, r.technique, r.condition, r.contraindication_type,
                        r.severity, r.rationale, r.is_active, r.created_at,
                        r.updated_at, r.version,
                        COALESCE(
                            array_agg(DISTINCT a.alternative_technique) FILTER (WHERE a.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as alternatives,
                        COALESCE(
                            array_agg(DISTINCT p.prerequisite) FILTER (WHERE p.id IS NOT NULL),
                            ARRAY[]::text[]
                        ) as prerequisites
                    FROM contraindication_rules r
                    LEFT JOIN rule_alternatives a ON r.id = a.rule_id
                    LEFT JOIN rule_prerequisites p ON r.id = p.rule_id
                    WHERE r.is_active = TRUE
                    GROUP BY r.id
                    ORDER BY r.technique, r.severity DESC
                    """
                )

                rules = [self._map_row_to_record(row) for row in rows]

                if use_cache:
                    self._cache[cache_key] = rules
                    self._cache_timestamp = datetime.now(timezone.utc)

                logger.info("all_rules_fetched", count=len(rules))
                return rules

        except Exception as e:
            logger.error("find_all_active_failed", error=str(e))
            return []

    async def save(
        self,
        technique: str,
        condition: str,
        contraindication_type: str,
        severity: Decimal,
        rationale: str,
        alternatives: list[str] | None = None,
        prerequisites: list[str] | None = None,
        created_by: str | None = None
    ) -> ContraindicationRuleRecord | None:
        """
        Save a new contraindication rule.

        Args:
            technique: Therapy technique name
            condition: Mental health condition name
            contraindication_type: Type of contraindication
            severity: Severity score (0.0 to 1.0)
            rationale: Clinical rationale
            alternatives: List of alternative techniques
            prerequisites: List of prerequisites
            created_by: User creating the rule

        Returns:
            The created rule record if successful, None otherwise
        """
        if not self.is_connected:
            logger.warning("database_not_connected", operation="save")
            return None

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        INSERT INTO contraindication_rules
                            (technique, condition, contraindication_type, severity, rationale, created_by)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id, technique, condition, contraindication_type, severity,
                                  rationale, is_active, created_at, updated_at, version
                        """,
                        technique, condition, contraindication_type, severity, rationale, created_by
                    )

                    rule_id = row["id"]

                    if alternatives:
                        for i, alt in enumerate(alternatives):
                            await conn.execute(
                                """
                                INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
                                VALUES ($1, $2, $3)
                                """,
                                rule_id, alt, i
                            )

                    if prerequisites:
                        for i, prereq in enumerate(prerequisites):
                            await conn.execute(
                                """
                                INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
                                VALUES ($1, $2, $3)
                                """,
                                rule_id, prereq, i
                            )

                    self._invalidate_cache()

                    logger.info(
                        "rule_saved",
                        rule_id=str(rule_id),
                        technique=technique,
                        condition=condition
                    )

                    return ContraindicationRuleRecord(
                        id=row["id"],
                        technique=row["technique"],
                        condition=row["condition"],
                        contraindication_type=row["contraindication_type"],
                        severity=Decimal(str(row["severity"])),
                        rationale=row["rationale"],
                        is_active=row["is_active"],
                        alternatives=alternatives or [],
                        prerequisites=prerequisites or [],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        version=row["version"]
                    )

        except Exception as e:
            logger.error(
                "save_failed",
                technique=technique,
                condition=condition,
                error=str(e)
            )
            return None

    async def update(
        self,
        rule_id: UUID,
        severity: Decimal | None = None,
        rationale: str | None = None,
        is_active: bool | None = None,
        updated_by: str | None = None
    ) -> bool:
        """
        Update an existing contraindication rule.

        Args:
            rule_id: UUID of the rule to update
            severity: New severity score
            rationale: New rationale
            is_active: Whether rule is active
            updated_by: User updating the rule

        Returns:
            True if update successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("database_not_connected", operation="update")
            return False

        try:
            updates = []
            params = []
            param_count = 0

            if severity is not None:
                param_count += 1
                updates.append(f"severity = ${param_count}")
                params.append(severity)

            if rationale is not None:
                param_count += 1
                updates.append(f"rationale = ${param_count}")
                params.append(rationale)

            if is_active is not None:
                param_count += 1
                updates.append(f"is_active = ${param_count}")
                params.append(is_active)

            if updated_by is not None:
                param_count += 1
                updates.append(f"updated_by = ${param_count}")
                params.append(updated_by)

            if not updates:
                return True

            param_count += 1
            params.append(rule_id)

            query = f"""
                UPDATE contraindication_rules
                SET {', '.join(updates)}
                WHERE id = ${param_count}
            """

            async with self._pool.acquire() as conn:
                result = await conn.execute(query, *params)

                if result == "UPDATE 1":
                    self._invalidate_cache()
                    logger.info("rule_updated", rule_id=str(rule_id))
                    return True

                logger.warning("rule_not_found", rule_id=str(rule_id))
                return False

        except Exception as e:
            logger.error("update_failed", rule_id=str(rule_id), error=str(e))
            return False

    async def deactivate(self, rule_id: UUID, deactivated_by: str | None = None) -> bool:
        """
        Deactivate a contraindication rule (soft delete).

        Args:
            rule_id: UUID of the rule to deactivate
            deactivated_by: User deactivating the rule

        Returns:
            True if deactivation successful, False otherwise
        """
        return await self.update(rule_id, is_active=False, updated_by=deactivated_by)

    def _map_row_to_record(self, row: asyncpg.Record) -> ContraindicationRuleRecord:
        """Map database row to record object."""
        return ContraindicationRuleRecord(
            id=row["id"],
            technique=row["technique"],
            condition=row["condition"],
            contraindication_type=row["contraindication_type"],
            severity=Decimal(str(row["severity"])),
            rationale=row["rationale"],
            is_active=row["is_active"],
            alternatives=list(row["alternatives"]) if row["alternatives"] else [],
            prerequisites=list(row["prerequisites"]) if row["prerequisites"] else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"]
        )


# Singleton instance management
_repository_instance: ContraindicationRepository | None = None


async def get_contraindication_repository(
    config: DatabaseConfig | None = None
) -> ContraindicationRepository:
    """
    Get or create the contraindication repository singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The repository instance
    """
    global _repository_instance

    if _repository_instance is None:
        _repository_instance = ContraindicationRepository(config)
        await _repository_instance.connect()

    return _repository_instance


async def close_contraindication_repository() -> None:
    """Close the contraindication repository singleton."""
    global _repository_instance

    if _repository_instance is not None:
        await _repository_instance.disconnect()
        _repository_instance = None


# Backward compatibility aliases
ContraindicationDBConfig = DatabaseConfig
ContraindicationRuleDTO = ContraindicationRuleRecord
ContraindicationDatabase = ContraindicationRepository
get_contraindication_db = get_contraindication_repository
close_contraindication_db = close_contraindication_repository
