"""Solace-AI Authorization - RBAC/ABAC policy enforcement."""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CHAT_SEND = "chat:send"
    CHAT_READ = "chat:read"
    SESSION_CREATE = "session:create"
    SESSION_READ = "session:read"
    SESSION_DELETE = "session:delete"
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    ASSESSMENT_READ = "assessment:read"
    ASSESSMENT_WRITE = "assessment:write"
    AUDIT_READ = "audit:read"
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"
    PHI_ACCESS = "phi:access"
    PHI_EXPORT = "phi:export"


class Role(str, Enum):
    """System roles with hierarchical permissions."""
    ANONYMOUS = "anonymous"
    USER = "user"
    THERAPIST = "therapist"
    CLINICIAN = "clinician"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"
    SERVICE = "service"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ANONYMOUS: set(),
    Role.USER: {
        Permission.CHAT_SEND, Permission.CHAT_READ, Permission.SESSION_CREATE,
        Permission.SESSION_READ, Permission.USER_READ, Permission.ASSESSMENT_READ,
        Permission.ASSESSMENT_WRITE, Permission.SETTINGS_READ, Permission.SETTINGS_WRITE,
    },
    Role.THERAPIST: {
        Permission.CHAT_SEND, Permission.CHAT_READ, Permission.SESSION_CREATE,
        Permission.SESSION_READ, Permission.SESSION_DELETE, Permission.USER_READ,
        Permission.ASSESSMENT_READ, Permission.ASSESSMENT_WRITE, Permission.PHI_ACCESS,
        Permission.SETTINGS_READ, Permission.SETTINGS_WRITE,
    },
    Role.CLINICIAN: {
        Permission.CHAT_SEND, Permission.CHAT_READ, Permission.SESSION_CREATE,
        Permission.SESSION_READ, Permission.SESSION_DELETE, Permission.USER_READ,
        Permission.USER_WRITE, Permission.ASSESSMENT_READ, Permission.ASSESSMENT_WRITE,
        Permission.PHI_ACCESS, Permission.PHI_EXPORT, Permission.AUDIT_READ,
        Permission.SETTINGS_READ, Permission.SETTINGS_WRITE,
    },
    Role.ADMIN: {
        Permission.READ, Permission.WRITE, Permission.DELETE,
        Permission.CHAT_SEND, Permission.CHAT_READ, Permission.SESSION_CREATE,
        Permission.SESSION_READ, Permission.SESSION_DELETE, Permission.USER_READ,
        Permission.USER_WRITE, Permission.USER_DELETE, Permission.ASSESSMENT_READ,
        Permission.ASSESSMENT_WRITE, Permission.PHI_ACCESS, Permission.PHI_EXPORT,
        Permission.AUDIT_READ, Permission.SETTINGS_READ, Permission.SETTINGS_WRITE,
    },
    Role.SUPERADMIN: {Permission.ADMIN} | {p for p in Permission},
    Role.SERVICE: {
        Permission.READ, Permission.WRITE, Permission.PHI_ACCESS,
        Permission.SESSION_CREATE, Permission.SESSION_READ,
    },
}


class ResourceType(str, Enum):
    """Types of resources in the system."""
    USER = "user"
    SESSION = "session"
    MESSAGE = "message"
    ASSESSMENT = "assessment"
    TREATMENT_PLAN = "treatment_plan"
    AUDIT_LOG = "audit_log"
    SETTINGS = "settings"
    API_KEY = "api_key"


class AuthorizationContext(BaseModel):
    """Context for authorization decisions."""
    user_id: str
    roles: list[Role] = Field(default_factory=list)
    permissions: list[Permission] = Field(default_factory=list)
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: dict[str, Any] = Field(default_factory=dict)

    def has_role(self, role: Role) -> bool:
        return role in self.roles

    def has_permission(self, permission: Permission) -> bool:
        if permission in self.permissions:
            return True
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False

    def get_all_permissions(self) -> set[Permission]:
        all_perms = set(self.permissions)
        for role in self.roles:
            all_perms |= ROLE_PERMISSIONS.get(role, set())
        return all_perms


class Resource(BaseModel):
    """Resource being accessed."""
    resource_type: ResourceType
    resource_id: str
    owner_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class AuthorizationDecision(BaseModel):
    """Result of authorization check."""
    allowed: bool
    reason: str
    policy_name: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def allow(cls, reason: str, policy: str | None = None) -> AuthorizationDecision:
        return cls(allowed=True, reason=reason, policy_name=policy)

    @classmethod
    def deny(cls, reason: str, policy: str | None = None) -> AuthorizationDecision:
        return cls(allowed=False, reason=reason, policy_name=policy)


class Policy(ABC):
    """Abstract base for authorization policies."""
    name: str = "base_policy"
    priority: int = 0

    @abstractmethod
    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        """Evaluate policy. Return None to abstain."""
        pass


class RBACPolicy(Policy):
    """Role-Based Access Control policy."""
    name = "rbac_policy"
    priority = 10

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        if context.has_permission(action):
            return AuthorizationDecision.allow(
                f"Permission {action.value} granted via role", self.name)
        return None


class OwnershipPolicy(Policy):
    """Resource ownership policy."""
    name = "ownership_policy"
    priority = 20

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        if resource.owner_id and resource.owner_id == context.user_id:
            allowed_actions = {Permission.READ, Permission.WRITE, Permission.DELETE}
            if action in allowed_actions or action.value.endswith(":read") or action.value.endswith(":write"):
                return AuthorizationDecision.allow("Owner has access to own resource", self.name)
        return None


class ResourceTypePolicy(Policy):
    """Policy based on resource type restrictions."""
    name = "resource_type_policy"
    priority = 30
    _resource_permissions: dict[ResourceType, set[Permission]] = {
        ResourceType.USER: {Permission.USER_READ, Permission.USER_WRITE, Permission.USER_DELETE},
        ResourceType.SESSION: {Permission.SESSION_CREATE, Permission.SESSION_READ, Permission.SESSION_DELETE},
        ResourceType.MESSAGE: {Permission.CHAT_SEND, Permission.CHAT_READ},
        ResourceType.ASSESSMENT: {Permission.ASSESSMENT_READ, Permission.ASSESSMENT_WRITE},
        ResourceType.AUDIT_LOG: {Permission.AUDIT_READ},
        ResourceType.SETTINGS: {Permission.SETTINGS_READ, Permission.SETTINGS_WRITE},
    }

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        required_perms = self._resource_permissions.get(resource.resource_type)
        if required_perms and action in required_perms and context.has_permission(action):
            return AuthorizationDecision.allow(
                f"Action {action.value} allowed on {resource.resource_type.value}", self.name)
        return None


class TimeBasedPolicy(Policy):
    """Policy restricting access by time."""
    name = "time_policy"
    priority = 5

    def __init__(self, allowed_hours: tuple[int, int] = (0, 24)) -> None:
        self._start_hour, self._end_hour = allowed_hours

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        current_hour = context.request_time.hour
        if not (self._start_hour <= current_hour < self._end_hour):
            return AuthorizationDecision.deny(
                f"Access restricted outside hours {self._start_hour}-{self._end_hour}", self.name)
        return None


class IPRestrictionPolicy(Policy):
    """Policy restricting access by IP address."""
    name = "ip_restriction_policy"
    priority = 1

    def __init__(self, allowed_patterns: list[str] | None = None,
                 blocked_patterns: list[str] | None = None) -> None:
        self._allowed = [re.compile(p) for p in (allowed_patterns or [])]
        self._blocked = [re.compile(p) for p in (blocked_patterns or [])]

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        if not context.ip_address:
            return None
        for pattern in self._blocked:
            if pattern.match(context.ip_address):
                return AuthorizationDecision.deny(f"IP {context.ip_address} is blocked", self.name)
        if self._allowed:
            for pattern in self._allowed:
                if pattern.match(context.ip_address):
                    return None
            return AuthorizationDecision.deny(f"IP {context.ip_address} not in allowed list", self.name)
        return None


class AttributeBasedPolicy(Policy):
    """Attribute-Based Access Control (ABAC) policy."""
    name = "abac_policy"
    priority = 15

    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self._rules = rules or []

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision | None:
        for rule in self._rules:
            if self._matches_rule(rule, context, resource, action):
                if rule.get("effect") == "deny":
                    return AuthorizationDecision.deny(rule.get("reason", "ABAC rule denied"), self.name)
                return AuthorizationDecision.allow(rule.get("reason", "ABAC rule allowed"), self.name)
        return None

    def _matches_rule(self, rule: dict[str, Any], context: AuthorizationContext,
                      resource: Resource, action: Permission) -> bool:
        if "action" in rule and rule["action"] != action.value:
            return False
        if "resource_type" in rule and rule["resource_type"] != resource.resource_type.value:
            return False
        if "user_attribute" in rule:
            attr_name, attr_value = rule["user_attribute"]
            if context.attributes.get(attr_name) != attr_value:
                return False
        if "resource_attribute" in rule:
            attr_name, attr_value = rule["resource_attribute"]
            if resource.attributes.get(attr_name) != attr_value:
                return False
        return True


class PolicyEngine:
    """Engine for evaluating authorization policies."""

    def __init__(self) -> None:
        self._policies: list[Policy] = []

    def add_policy(self, policy: Policy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority)

    def remove_policy(self, policy_name: str) -> bool:
        original_len = len(self._policies)
        self._policies = [p for p in self._policies if p.name != policy_name]
        return len(self._policies) < original_len

    def evaluate(self, context: AuthorizationContext, resource: Resource,
                 action: Permission) -> AuthorizationDecision:
        for policy in self._policies:
            decision = policy.evaluate(context, resource, action)
            if decision is not None:
                logger.debug("policy_decision", policy=policy.name,
                             allowed=decision.allowed, action=action.value)
                return decision
        return AuthorizationDecision.deny("No policy granted access")

    def is_authorized(self, context: AuthorizationContext, resource: Resource,
                      action: Permission) -> bool:
        decision = self.evaluate(context, resource, action)
        if not decision.allowed:
            logger.warning("authorization_denied", user_id=context.user_id,
                           action=action.value, resource_type=resource.resource_type.value,
                           resource_id=resource.resource_id, reason=decision.reason)
        return decision.allowed


class Authorizer:
    """High-level authorization interface."""

    def __init__(self, engine: PolicyEngine | None = None) -> None:
        self._engine = engine or PolicyEngine()

    def check(self, context: AuthorizationContext, resource: Resource,
              action: Permission) -> AuthorizationDecision:
        return self._engine.evaluate(context, resource, action)

    def require(self, context: AuthorizationContext, resource: Resource,
                action: Permission) -> None:
        decision = self.check(context, resource, action)
        if not decision.allowed:
            raise PermissionError(f"Authorization denied: {decision.reason}")

    def can_access(self, context: AuthorizationContext, resource: Resource,
                   action: Permission) -> bool:
        return self._engine.is_authorized(context, resource, action)


def create_default_policy_engine() -> PolicyEngine:
    """Create policy engine with default policies."""
    engine = PolicyEngine()
    engine.add_policy(RBACPolicy())
    engine.add_policy(OwnershipPolicy())
    engine.add_policy(ResourceTypePolicy())
    return engine


def create_authorizer(engine: PolicyEngine | None = None) -> Authorizer:
    """Factory function to create authorizer."""
    return Authorizer(engine or create_default_policy_engine())
