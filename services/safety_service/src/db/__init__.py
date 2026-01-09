"""
Solace-AI Safety Service Database Module.
Provides database access layers for safety service data.
"""
from safety_service.src.db.contraindication_db import (
    ContraindicationDatabase,
    ContraindicationDBConfig,
    ContraindicationRuleDTO,
    get_contraindication_db,
    close_contraindication_db,
)

__all__ = [
    "ContraindicationDatabase",
    "ContraindicationDBConfig",
    "ContraindicationRuleDTO",
    "get_contraindication_db",
    "close_contraindication_db",
]
