"""Unit tests for authorization module."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from solace_security.authorization import (
    Permission,
    Role,
    ROLE_PERMISSIONS,
    ResourceType,
    AuthorizationContext,
    Resource,
    AuthorizationDecision,
    RBACPolicy,
    OwnershipPolicy,
    ResourceTypePolicy,
    TimeBasedPolicy,
    IPRestrictionPolicy,
    AttributeBasedPolicy,
    PolicyEngine,
    Authorizer,
    create_default_policy_engine,
    create_authorizer,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.CHAT_SEND.value == "chat:send"
        assert Permission.PHI_ACCESS.value == "phi:access"


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        assert Role.USER.value == "user"
        assert Role.ADMIN.value == "admin"
        assert Role.CLINICIAN.value == "clinician"


class TestRolePermissions:
    """Tests for role-permission mappings."""

    def test_user_permissions(self):
        perms = ROLE_PERMISSIONS[Role.USER]
        assert Permission.CHAT_SEND in perms
        assert Permission.SESSION_CREATE in perms
        assert Permission.PHI_ACCESS not in perms

    def test_clinician_permissions(self):
        perms = ROLE_PERMISSIONS[Role.CLINICIAN]
        assert Permission.PHI_ACCESS in perms
        assert Permission.PHI_EXPORT in perms

    def test_admin_permissions(self):
        perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.USER_DELETE in perms
        assert Permission.ADMIN not in perms

    def test_superadmin_has_all(self):
        perms = ROLE_PERMISSIONS[Role.SUPERADMIN]
        assert Permission.ADMIN in perms


class TestAuthorizationContext:
    """Tests for AuthorizationContext."""

    def test_create_context(self):
        ctx = AuthorizationContext(
            user_id="user123",
            roles=[Role.USER],
            permissions=[Permission.READ]
        )
        assert ctx.user_id == "user123"
        assert Role.USER in ctx.roles

    def test_has_role(self):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER, Role.THERAPIST])
        assert ctx.has_role(Role.USER)
        assert ctx.has_role(Role.THERAPIST)
        assert not ctx.has_role(Role.ADMIN)

    def test_has_permission_direct(self):
        ctx = AuthorizationContext(user_id="u1", permissions=[Permission.CHAT_SEND])
        assert ctx.has_permission(Permission.CHAT_SEND)

    def test_has_permission_via_role(self):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        assert ctx.has_permission(Permission.CHAT_SEND)
        assert not ctx.has_permission(Permission.PHI_ACCESS)

    def test_get_all_permissions(self):
        ctx = AuthorizationContext(
            user_id="u1", roles=[Role.USER],
            permissions=[Permission.PHI_ACCESS]
        )
        all_perms = ctx.get_all_permissions()
        assert Permission.CHAT_SEND in all_perms
        assert Permission.PHI_ACCESS in all_perms


class TestResource:
    """Tests for Resource model."""

    def test_create_resource(self):
        resource = Resource(
            resource_type=ResourceType.USER,
            resource_id="user123",
            owner_id="user123"
        )
        assert resource.resource_type == ResourceType.USER
        assert resource.owner_id == "user123"


class TestAuthorizationDecision:
    """Tests for AuthorizationDecision."""

    def test_allow_decision(self):
        decision = AuthorizationDecision.allow("Permission granted", "test_policy")
        assert decision.allowed
        assert decision.reason == "Permission granted"
        assert decision.policy_name == "test_policy"

    def test_deny_decision(self):
        decision = AuthorizationDecision.deny("Access denied", "test_policy")
        assert not decision.allowed
        assert decision.reason == "Access denied"


class TestRBACPolicy:
    """Tests for RBACPolicy."""

    @pytest.fixture
    def policy(self):
        return RBACPolicy()

    def test_allows_with_permission(self, policy):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.SESSION, resource_id="s1")
        decision = policy.evaluate(ctx, resource, Permission.CHAT_SEND)
        assert decision is not None
        assert decision.allowed

    def test_abstains_without_permission(self, policy):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.USER, resource_id="u2")
        decision = policy.evaluate(ctx, resource, Permission.PHI_ACCESS)
        assert decision is None


class TestOwnershipPolicy:
    """Tests for OwnershipPolicy."""

    @pytest.fixture
    def policy(self):
        return OwnershipPolicy()

    def test_allows_owner_access(self, policy):
        ctx = AuthorizationContext(user_id="user123")
        resource = Resource(
            resource_type=ResourceType.USER,
            resource_id="user123",
            owner_id="user123"
        )
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is not None
        assert decision.allowed

    def test_abstains_non_owner(self, policy):
        ctx = AuthorizationContext(user_id="user123")
        resource = Resource(
            resource_type=ResourceType.USER,
            resource_id="user456",
            owner_id="user456"
        )
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is None


class TestResourceTypePolicy:
    """Tests for ResourceTypePolicy."""

    @pytest.fixture
    def policy(self):
        return ResourceTypePolicy()

    def test_allows_matching_permission(self, policy):
        ctx = AuthorizationContext(user_id="u1", permissions=[Permission.USER_READ])
        resource = Resource(resource_type=ResourceType.USER, resource_id="u2")
        decision = policy.evaluate(ctx, resource, Permission.USER_READ)
        assert decision is not None
        assert decision.allowed


class TestTimeBasedPolicy:
    """Tests for TimeBasedPolicy."""

    def test_allows_during_hours(self):
        policy = TimeBasedPolicy(allowed_hours=(0, 24))
        ctx = AuthorizationContext(user_id="u1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is None

    def test_denies_outside_hours(self):
        current_hour = datetime.now(timezone.utc).hour
        blocked_start = (current_hour + 1) % 24
        blocked_end = (current_hour + 2) % 24
        if blocked_start >= blocked_end:
            blocked_end = blocked_start + 1
        policy = TimeBasedPolicy(allowed_hours=(blocked_start, blocked_end))
        ctx = AuthorizationContext(user_id="u1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is not None
        assert not decision.allowed


class TestIPRestrictionPolicy:
    """Tests for IPRestrictionPolicy."""

    def test_blocks_ip(self):
        policy = IPRestrictionPolicy(blocked_patterns=[r"192\.168\..*"])
        ctx = AuthorizationContext(user_id="u1", ip_address="192.168.1.1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is not None
        assert not decision.allowed

    def test_allows_non_blocked_ip(self):
        policy = IPRestrictionPolicy(blocked_patterns=[r"192\.168\..*"])
        ctx = AuthorizationContext(user_id="u1", ip_address="10.0.0.1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is None

    def test_abstains_no_ip(self):
        policy = IPRestrictionPolicy(blocked_patterns=[r".*"])
        ctx = AuthorizationContext(user_id="u1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is None


class TestAttributeBasedPolicy:
    """Tests for AttributeBasedPolicy."""

    def test_matches_rule(self):
        rules = [{"action": "read", "resource_type": "user", "effect": "allow", "reason": "ABAC allowed"}]
        policy = AttributeBasedPolicy(rules=rules)
        ctx = AuthorizationContext(user_id="u1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.READ)
        assert decision is not None
        assert decision.allowed

    def test_denies_rule(self):
        rules = [{"action": "write", "resource_type": "user", "effect": "deny", "reason": "Write denied"}]
        policy = AttributeBasedPolicy(rules=rules)
        ctx = AuthorizationContext(user_id="u1")
        resource = Resource(resource_type=ResourceType.USER, resource_id="u1")
        decision = policy.evaluate(ctx, resource, Permission.WRITE)
        assert decision is not None
        assert not decision.allowed


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    @pytest.fixture
    def engine(self):
        engine = PolicyEngine()
        engine.add_policy(RBACPolicy())
        engine.add_policy(OwnershipPolicy())
        return engine

    def test_evaluate_allows(self, engine):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.SESSION, resource_id="s1")
        decision = engine.evaluate(ctx, resource, Permission.CHAT_SEND)
        assert decision.allowed

    def test_evaluate_denies(self, engine):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.ANONYMOUS])
        resource = Resource(resource_type=ResourceType.USER, resource_id="u2")
        decision = engine.evaluate(ctx, resource, Permission.ADMIN)
        assert not decision.allowed

    def test_remove_policy(self, engine):
        assert engine.remove_policy("rbac_policy")
        assert not engine.remove_policy("nonexistent")

    def test_is_authorized(self, engine):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.SESSION, resource_id="s1")
        assert engine.is_authorized(ctx, resource, Permission.CHAT_SEND)


class TestAuthorizer:
    """Tests for Authorizer."""

    @pytest.fixture
    def authorizer(self):
        return create_authorizer()

    def test_check_allowed(self, authorizer):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.SESSION, resource_id="s1")
        decision = authorizer.check(ctx, resource, Permission.CHAT_SEND)
        assert decision.allowed

    def test_can_access(self, authorizer):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.USER])
        resource = Resource(resource_type=ResourceType.SESSION, resource_id="s1")
        assert authorizer.can_access(ctx, resource, Permission.CHAT_SEND)

    def test_require_raises(self, authorizer):
        ctx = AuthorizationContext(user_id="u1", roles=[Role.ANONYMOUS])
        resource = Resource(resource_type=ResourceType.USER, resource_id="u2")
        with pytest.raises(PermissionError):
            authorizer.require(ctx, resource, Permission.ADMIN)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_policy_engine(self):
        engine = create_default_policy_engine()
        assert isinstance(engine, PolicyEngine)

    def test_create_authorizer(self):
        authorizer = create_authorizer()
        assert isinstance(authorizer, Authorizer)
