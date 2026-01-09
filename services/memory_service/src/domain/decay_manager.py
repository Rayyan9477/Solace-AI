"""
Solace-AI Memory Service - Decay Manager.
Implements Ebbinghaus decay model with safety override for memory retention.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class DecaySettings(BaseSettings):
    """Configuration for Ebbinghaus decay model."""
    base_decay_rate: Decimal = Field(default=Decimal("0.1"), description="Lambda base decay per day")
    short_term_rate: Decimal = Field(default=Decimal("0.15"), description="Short-term decay rate")
    medium_term_rate: Decimal = Field(default=Decimal("0.05"), description="Medium-term decay rate")
    long_term_rate: Decimal = Field(default=Decimal("0.02"), description="Long-term decay rate")
    reinforcement_multiplier: Decimal = Field(default=Decimal("1.5"), description="Stability boost on access")
    archive_threshold: Decimal = Field(default=Decimal("0.3"), description="Archive below this strength")
    delete_threshold: Decimal = Field(default=Decimal("0.1"), description="Delete below this strength")
    max_stability: Decimal = Field(default=Decimal("10.0"), description="Maximum stability value")
    emotional_decay_modifier: Decimal = Field(default=Decimal("0.7"), description="High emotion decay modifier")
    clinical_decay_modifier: Decimal = Field(default=Decimal("0.5"), description="Clinical info decay modifier")
    decay_interval_hours: int = Field(default=24, description="Hours between decay cycles")
    model_config = SettingsConfigDict(env_prefix="DECAY_", env_file=".env", extra="ignore")


class RetentionCategory(str, Enum):
    """Categories affecting decay rate."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"


class DecayAction(str, Enum):
    """Actions resulting from decay evaluation."""
    KEEP = "keep"
    REINFORCE = "reinforce"
    ARCHIVE = "archive"
    DELETE = "delete"


class DecayableItem(Protocol):
    """Protocol for items that can decay."""
    @property
    def retention_category(self) -> str: ...
    @property
    def retention_strength(self) -> Decimal: ...
    @retention_strength.setter
    def retention_strength(self, value: Decimal) -> None: ...
    @property
    def created_at(self) -> datetime: ...
    @property
    def accessed_at(self) -> datetime: ...


@dataclass
class DecayResult:
    """Result of decay calculation for a single item."""
    item_id: UUID
    original_strength: Decimal
    new_strength: Decimal
    decay_applied: Decimal
    stability_factor: Decimal
    action: DecayAction
    retention_category: RetentionCategory


class DecayBatchResult(BaseModel):
    """Result from batch decay processing."""
    batch_id: UUID = Field(default_factory=uuid4)
    items_processed: int = Field(default=0)
    items_kept: int = Field(default=0)
    items_reinforced: int = Field(default=0)
    items_archived: int = Field(default=0)
    items_deleted: int = Field(default=0)
    total_decay_applied: float = Field(default=0.0)
    processing_time_ms: int = Field(default=0)


@dataclass
class StabilityRecord:
    """Tracks stability (reinforcement history) for an item."""
    item_id: UUID = field(default_factory=uuid4)
    stability: Decimal = Decimal("1.0")
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reinforcement_history: list[datetime] = field(default_factory=list)


class DecayManager:
    """Manages Ebbinghaus decay model with safety override."""

    def __init__(self, settings: DecaySettings | None = None) -> None:
        self._settings = settings or DecaySettings()
        self._stability_records: dict[UUID, StabilityRecord] = {}
        self._permanent_items: set[UUID] = set()
        self._last_decay_cycle: datetime | None = None
        self._stats = {"decay_cycles": 0, "items_processed": 0, "items_archived": 0,
                       "items_deleted": 0, "reinforcements": 0}

    def calculate_retention(self, time_elapsed_days: float, stability: Decimal,
                            decay_rate: Decimal) -> Decimal:
        """Calculate retention using Ebbinghaus formula: R(t) = e^(-λt) × S"""
        if stability <= 0:
            return Decimal("0")
        lambda_t = float(decay_rate) * time_elapsed_days
        base_retention = Decimal(str(math.exp(-lambda_t)))
        return min(Decimal("1.0"), base_retention * stability)

    def get_decay_rate(self, category: RetentionCategory, is_emotional: bool = False,
                       is_clinical: bool = False) -> Decimal:
        """Get decay rate for retention category with modifiers."""
        if category == RetentionCategory.PERMANENT:
            return Decimal("0")
        rate_map = {
            RetentionCategory.LONG_TERM: self._settings.long_term_rate,
            RetentionCategory.MEDIUM_TERM: self._settings.medium_term_rate,
            RetentionCategory.SHORT_TERM: self._settings.short_term_rate,
        }
        rate = rate_map.get(category, self._settings.base_decay_rate)
        if is_emotional:
            rate = rate * self._settings.emotional_decay_modifier
        if is_clinical:
            rate = rate * self._settings.clinical_decay_modifier
        return rate

    def apply_decay(self, item_id: UUID, current_strength: Decimal,
                    retention_category: str, created_at: datetime,
                    accessed_at: datetime | None = None,
                    is_emotional: bool = False,
                    is_clinical: bool = False) -> DecayResult:
        """Apply decay to a single item."""
        self._stats["items_processed"] += 1
        category = RetentionCategory(retention_category) if retention_category in [e.value for e in RetentionCategory] else RetentionCategory.MEDIUM_TERM
        if category == RetentionCategory.PERMANENT or item_id in self._permanent_items:
            return DecayResult(
                item_id=item_id, original_strength=current_strength,
                new_strength=current_strength, decay_applied=Decimal("0"),
                stability_factor=Decimal("1.0"), action=DecayAction.KEEP,
                retention_category=RetentionCategory.PERMANENT,
            )
        stability_record = self._stability_records.get(item_id)
        stability = stability_record.stability if stability_record else Decimal("1.0")
        decay_rate = self.get_decay_rate(category, is_emotional, is_clinical)
        reference_time = accessed_at or created_at
        days_elapsed = (datetime.now(timezone.utc) - reference_time).total_seconds() / 86400
        new_strength = self.calculate_retention(days_elapsed, stability, decay_rate)
        decay_applied = max(Decimal("0"), current_strength - new_strength)
        action = self._determine_action(new_strength)
        if action == DecayAction.ARCHIVE:
            self._stats["items_archived"] += 1
        elif action == DecayAction.DELETE:
            self._stats["items_deleted"] += 1
        return DecayResult(
            item_id=item_id, original_strength=current_strength,
            new_strength=new_strength, decay_applied=decay_applied,
            stability_factor=stability, action=action, retention_category=category,
        )

    def reinforce(self, item_id: UUID) -> Decimal:
        """Reinforce an item (boost stability on access)."""
        self._stats["reinforcements"] += 1
        record = self._stability_records.get(item_id)
        if not record:
            record = StabilityRecord(item_id=item_id)
            self._stability_records[item_id] = record
        record.access_count += 1
        record.last_accessed = datetime.now(timezone.utc)
        record.reinforcement_history.append(datetime.now(timezone.utc))
        if len(record.reinforcement_history) > 10:
            record.reinforcement_history = record.reinforcement_history[-10:]
        record.stability = min(
            self._settings.max_stability,
            record.stability * self._settings.reinforcement_multiplier
        )
        logger.debug("item_reinforced", item_id=str(item_id), new_stability=float(record.stability))
        return record.stability

    def mark_permanent(self, item_id: UUID) -> None:
        """Mark an item as permanent (safety override)."""
        self._permanent_items.add(item_id)
        logger.info("item_marked_permanent", item_id=str(item_id))

    def unmark_permanent(self, item_id: UUID) -> bool:
        """Remove permanent status from item."""
        if item_id in self._permanent_items:
            self._permanent_items.discard(item_id)
            return True
        return False

    def is_permanent(self, item_id: UUID) -> bool:
        """Check if item is marked as permanent."""
        return item_id in self._permanent_items

    def process_batch(self, items: list[tuple[UUID, Decimal, str, datetime, datetime | None]],
                      apply_safety_override: bool = True) -> DecayBatchResult:
        """Process decay for a batch of items."""
        import time
        start_time = time.perf_counter()
        self._stats["decay_cycles"] += 1
        result = DecayBatchResult()
        total_decay = Decimal("0")
        for item_id, strength, category, created, accessed in items:
            if apply_safety_override and self._is_safety_critical(category):
                self.mark_permanent(item_id)
            decay_result = self.apply_decay(item_id, strength, category, created, accessed)
            total_decay += decay_result.decay_applied
            result.items_processed += 1
            if decay_result.action == DecayAction.KEEP:
                result.items_kept += 1
            elif decay_result.action == DecayAction.REINFORCE:
                result.items_reinforced += 1
            elif decay_result.action == DecayAction.ARCHIVE:
                result.items_archived += 1
            elif decay_result.action == DecayAction.DELETE:
                result.items_deleted += 1
        result.total_decay_applied = float(total_decay)
        result.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        self._last_decay_cycle = datetime.now(timezone.utc)
        logger.info("decay_batch_processed", items=result.items_processed,
                    archived=result.items_archived, deleted=result.items_deleted)
        return result

    def get_stability(self, item_id: UUID) -> Decimal:
        """Get current stability for an item."""
        record = self._stability_records.get(item_id)
        return record.stability if record else Decimal("1.0")

    def set_stability(self, item_id: UUID, stability: Decimal) -> None:
        """Set stability for an item."""
        record = self._stability_records.get(item_id)
        if not record:
            record = StabilityRecord(item_id=item_id)
            self._stability_records[item_id] = record
        record.stability = min(self._settings.max_stability, max(Decimal("0.1"), stability))

    def should_run_decay_cycle(self) -> bool:
        """Check if enough time has passed for next decay cycle."""
        if not self._last_decay_cycle:
            return True
        elapsed = datetime.now(timezone.utc) - self._last_decay_cycle
        return elapsed >= timedelta(hours=self._settings.decay_interval_hours)

    def estimate_retention_at(self, item_id: UUID, current_strength: Decimal,
                              retention_category: str, created_at: datetime,
                              target_date: datetime) -> Decimal:
        """Estimate retention strength at a future date."""
        category = RetentionCategory(retention_category) if retention_category in [e.value for e in RetentionCategory] else RetentionCategory.MEDIUM_TERM
        if category == RetentionCategory.PERMANENT or item_id in self._permanent_items:
            return current_strength
        stability = self.get_stability(item_id)
        decay_rate = self.get_decay_rate(category)
        days_from_creation = (target_date - created_at).total_seconds() / 86400
        return self.calculate_retention(days_from_creation, stability, decay_rate)

    def get_retention_forecast(self, item_id: UUID, current_strength: Decimal,
                               retention_category: str, created_at: datetime,
                               days_ahead: int = 30) -> list[tuple[int, float]]:
        """Get retention forecast for next N days."""
        forecast = []
        for day in range(0, days_ahead + 1, max(1, days_ahead // 10)):
            target = datetime.now(timezone.utc) + timedelta(days=day)
            retention = self.estimate_retention_at(item_id, current_strength, retention_category, created_at, target)
            forecast.append((day, float(retention)))
        return forecast

    def cleanup_stability_records(self, max_age_days: int = 365) -> int:
        """Cleanup old stability records."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed = 0
        for item_id in list(self._stability_records.keys()):
            record = self._stability_records[item_id]
            if record.last_accessed < cutoff:
                del self._stability_records[item_id]
                removed += 1
        if removed:
            logger.info("stability_records_cleaned", removed=removed)
        return removed

    def get_statistics(self) -> dict[str, Any]:
        """Get decay manager statistics."""
        return {
            **self._stats,
            "permanent_items": len(self._permanent_items),
            "stability_records": len(self._stability_records),
            "last_decay_cycle": self._last_decay_cycle.isoformat() if self._last_decay_cycle else None,
            "settings": {
                "base_rate": float(self._settings.base_decay_rate),
                "archive_threshold": float(self._settings.archive_threshold),
                "delete_threshold": float(self._settings.delete_threshold),
            },
        }

    def _determine_action(self, strength: Decimal) -> DecayAction:
        """Determine action based on retention strength."""
        if strength >= Decimal("0.7"):
            return DecayAction.KEEP
        if strength >= self._settings.archive_threshold:
            return DecayAction.KEEP
        if strength >= self._settings.delete_threshold:
            return DecayAction.ARCHIVE
        return DecayAction.DELETE

    def _is_safety_critical(self, category: str) -> bool:
        """Check if category indicates safety-critical content."""
        safety_categories = ["permanent", "safety", "crisis", "emergency"]
        return category.lower() in safety_categories
