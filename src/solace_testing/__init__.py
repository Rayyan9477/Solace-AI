"""
Solace-AI Testing Library.

Enterprise-grade testing infrastructure for Solace-AI microservices:
- Common pytest fixtures for databases, caches, and services
- Mock implementations of PostgreSQL, Redis, Weaviate, LLM clients
- Test data factories using builder pattern
- Integration test utilities with service orchestration
- Contract testing helpers for API and event verification
"""

from solace_testing.fixtures import (
    DatabaseFixture,
    FixtureConfig,
    FixtureContext,
    HTTPClientFixture,
    KafkaFixture,
    MockPostgresConnection,
    ObservabilityFixture,
    PostgresFixture,
    RedisFixture,
    WeaviateFixture,
    fixture_scope,
    pytest_fixture_factory,
    sync_fixture_scope,
)
from solace_testing.mocks import (
    MockEventPublisher,
    MockHTTPClient,
    MockLLMClient,
    MockLLMResponse,
    MockPostgresClient,
    MockQueryResult,
    MockRedisClient,
    MockWeaviateClient,
)
from solace_testing.factories import (
    BaseFactory,
    DiagnosisFactory,
    EntityFactory,
    EntityIdFactory,
    EntityMetadataDict,
    EventFactory,
    FactoryConfig,
    FactoryRegistry,
    FactorySequence,
    LLMResponseFactory,
    MessageFactory,
    SafetyAssessmentFactory,
    SessionFactory,
    ToolCallFactory,
    UserFactory,
    VectorFactory,
    get_factory_registry,
)
from solace_testing.integration import (
    APITestClient,
    DataSeeder,
    HealthWaiter,
    IntegrationTestRunner,
    ServiceConfig,
    ServiceContainer,
    ServiceState,
    ServiceStatus,
    ServiceType,
    IntegrationEnvironment,
    create_test_environment,
)
from solace_testing.contracts import (
    ConsumerContractTest,
    ContractDefinition,
    ContractRegistry,
    ContractRequest,
    ContractResponse,
    ContractStatus,
    ContractVerifier,
    EventContractDefinition,
    FieldMatcher,
    HttpMethod,
    ProviderContractTest,
    SchemaType,
    VerificationResult,
)

__version__ = "0.1.0"

__all__ = [
    # Fixtures
    "DatabaseFixture",
    "FixtureConfig",
    "FixtureContext",
    "HTTPClientFixture",
    "KafkaFixture",
    "MockPostgresConnection",
    "ObservabilityFixture",
    "PostgresFixture",
    "RedisFixture",
    "WeaviateFixture",
    "fixture_scope",
    "pytest_fixture_factory",
    "sync_fixture_scope",
    # Mocks
    "MockEventPublisher",
    "MockHTTPClient",
    "MockLLMClient",
    "MockLLMResponse",
    "MockPostgresClient",
    "MockQueryResult",
    "MockRedisClient",
    "MockWeaviateClient",
    # Factories
    "BaseFactory",
    "DiagnosisFactory",
    "EntityFactory",
    "EntityIdFactory",
    "EntityMetadataDict",
    "EventFactory",
    "FactoryConfig",
    "FactoryRegistry",
    "FactorySequence",
    "LLMResponseFactory",
    "MessageFactory",
    "SafetyAssessmentFactory",
    "SessionFactory",
    "ToolCallFactory",
    "UserFactory",
    "VectorFactory",
    "get_factory_registry",
    # Integration
    "APITestClient",
    "DataSeeder",
    "HealthWaiter",
    "IntegrationTestRunner",
    "ServiceConfig",
    "ServiceContainer",
    "ServiceState",
    "ServiceStatus",
    "ServiceType",
    "IntegrationEnvironment",
    "create_test_environment",
    # Contracts
    "ConsumerContractTest",
    "ContractDefinition",
    "ContractRegistry",
    "ContractRequest",
    "ContractResponse",
    "ContractStatus",
    "ContractVerifier",
    "EventContractDefinition",
    "FieldMatcher",
    "HttpMethod",
    "ProviderContractTest",
    "SchemaType",
    "VerificationResult",
]
