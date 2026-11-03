"""IF-21: the registry of MVP service images. Stdlib only."""
from __future__ import annotations

import secrets
from dataclasses import dataclass

_RATE_LIMITING = "infrastructure.api_gateway.rate_limiting"


@dataclass(frozen=True)
class Image:
    name: str
    dockerfile: str
    module: str
    port: int | None
    probes: tuple[str, ...] = ()
    merged_src: bool = False


IMAGES: tuple[Image, ...] = (
    Image("orchestrator-service", "services/orchestrator_service/Dockerfile",
          "services.orchestrator_service.src.main", 8000, (_RATE_LIMITING,)),
    Image("safety-service", "services/safety_service/Dockerfile",
          "services.safety_service.src.main", 8002),
    Image("therapy-service", "services/therapy_service/Dockerfile",
          "services.therapy_service.src.main", 8006),
    Image("diagnosis-service", "services/diagnosis_service/Dockerfile",
          "services.diagnosis_service.src.main", 8004),
    Image("memory-service", "services/memory_service/Dockerfile",
          "services.memory_service.src.main", 8005),
    Image("personality-service", "services/personality_service/Dockerfile",
          "services.personality_service.src.main", 8007),
    Image("user-service", "services/user-service/Dockerfile",
          "src.main", 8001, (_RATE_LIMITING,), merged_src=True),
    Image("notification-service", "services/notification-service/Dockerfile",
          "src.main", 8003, ("src.api", "src.domain.channels"), merged_src=True),
    Image("analytics-service", "services/analytics-service/Dockerfile",
          "main", 8009),
    Image("migrate", "docker/migrate.Dockerfile", "", None,
          ("scripts.db.migrate", "solace_infrastructure.database.schema_check")),
)


def probe_env() -> dict[str, str]:
    """Fresh random throwaway values for the variables a module import requires (never literals)."""
    return {
        "POSTGRES_PASSWORD": secrets.token_urlsafe(24),
        "SECRET_KEY": secrets.token_urlsafe(48),
        "ENCRYPTION_KEY": secrets.token_urlsafe(32),
        "AUTH_SECRET_KEY": secrets.token_urlsafe(48),
        "JWT_SECRET_KEY": secrets.token_urlsafe(48),
    }
