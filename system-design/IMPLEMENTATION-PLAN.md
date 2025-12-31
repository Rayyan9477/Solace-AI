# Solace-AI: Complete Implementation Plan

> **Version**: 3.0
> **Date**: January 1, 2026
> **Author**: Principal Backend & AI Systems Engineer
> **Status**: Implementation Blueprint (Reviewed, Enhanced & Version-Verified)
> **Architecture**: Microservices + Event-Driven
> **Alignment**: Full alignment with system-design/*.md architecture documents
> **Technology Stack**: Verified via Context7 Documentation API (January 2025)

---

## Executive Summary

This document provides the complete implementation plan for Solace-AI, a mental health AI platform built on **microservices architecture** with **event-driven communication**. The system is decomposed into independently deployable services, each owning its domain logic and data.

### Architecture Principles

| Principle | Implementation |
|-----------|----------------|
| **Microservices** | Each domain module is an independent service |
| **Event-Driven** | Kafka for async communication between services |
| **API Gateway** | Kong/Istio for routing, auth, rate limiting |
| **Service Mesh** | Istio for mTLS, traffic management |
| **Domain-Driven** | Each service owns its bounded context |
| **Clean Architecture** | Hexagonal/Ports-Adapters within each service |

---

## Table of Contents

1. [Microservices Architecture Overview](#1-microservices-architecture-overview)
2. [Service Catalog](#2-service-catalog)
3. [Shared Libraries](#3-shared-libraries)
4. [Phase-by-Phase Implementation](#4-phase-by-phase-implementation)
5. [Directory Structure](#5-directory-structure)
6. [Deployment Architecture](#6-deployment-architecture)
7. [Implementation Execution Order](#7-implementation-execution-order)
8. [Quality Gates & Acceptance Criteria](#8-quality-gates--acceptance-criteria)
9. [Architecture Alignment: Critical Components](#9-architecture-alignment-critical-components)
10. [Event Schemas & API Contracts](#10-event-schemas--api-contracts)
11. [LangGraph Agent Priority Hierarchy](#11-langgraph-agent-priority-hierarchy)
12. [Technology Stack: Latest Versions & Patterns (2025)](#12-technology-stack-latest-versions--patterns-2025)

---

## 1. Microservices Architecture Overview

### 1.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              SOLACE-AI PLATFORM                                      │
│                         Microservices Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          API GATEWAY (Kong/Istio)                              │ │
│  │     JWT Auth │ Rate Limiting │ Request Routing │ TLS Termination │ CORS       │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                    ORCHESTRATION SERVICE (LangGraph)                           │ │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │   │  Supervisor  │  │    Safety    │  │    Router    │  │    State     │      │ │
│  │   │    Agent     │  │    Agent     │  │              │  │   Manager    │      │ │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                            │
│     ┌──────────────┬──────────────┬────┴────┬──────────────┬──────────────┐        │
│     │              │              │         │              │              │        │
│     ▼              ▼              ▼         ▼              ▼              ▼        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ SAFETY   │ │ MEMORY   │ │DIAGNOSIS │ │ THERAPY  │ │PERSONALITY│ │   USER   │   │
│  │ SERVICE  │ │ SERVICE  │ │ SERVICE  │ │ SERVICE  │ │ SERVICE  │ │ SERVICE  │   │
│  │   🛡️    │ │   🧠    │ │   🔍    │ │   💆    │ │   🎭    │ │   👤    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       │            │            │            │            │            │          │
│  ┌────┴────────────┴────────────┴────────────┴────────────┴────────────┴────┐    │
│  │                         KAFKA EVENT BUS                                   │    │
│  │   solace.safety │ solace.memory │ solace.diagnosis │ solace.therapy │ ...│    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                           DATA LAYER                                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │  │
│  │  │  Redis   │  │ Weaviate │  │ Postgres │  │  Kafka   │  │    S3    │    │  │
│  │  │  Cluster │  │  Vector  │  │    DB    │  │  Streams │  │ Archive  │    │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                      OBSERVABILITY LAYER                                   │  │
│  │     Prometheus │ Grafana │ Jaeger │ ELK Stack │ AlertManager              │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Service Communication Patterns

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     SERVICE COMMUNICATION PATTERNS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SYNCHRONOUS (REST/gRPC)                  ASYNCHRONOUS (Kafka Events)           │
│  ─────────────────────────                ──────────────────────────────        │
│  • Request/Response                       • Event Publishing                     │
│  • Real-time queries                      • Event Sourcing                       │
│  • Health checks                          • Saga Orchestration                   │
│  • User-facing API                        • Audit Trail                          │
│                                                                                  │
│  ┌─────────────┐     REST      ┌─────────────┐                                 │
│  │ Orchestrator│──────────────▶│   Memory    │                                 │
│  │   Service   │◀──────────────│   Service   │                                 │
│  └──────┬──────┘               └─────────────┘                                 │
│         │                                                                        │
│         │ Kafka Event                                                            │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        KAFKA EVENT BUS                                   │   │
│  │                                                                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │  Safety   │  │  Memory   │  │ Diagnosis │  │  Therapy  │            │   │
│  │  │  Events   │  │  Events   │  │  Events   │  │  Events   │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Service Mesh Architecture (Istio)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ISTIO SERVICE MESH                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      CONTROL PLANE (Istiod)                              │   │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │   │
│  │   │   Pilot   │  │  Citadel  │  │   Galley  │  │   Mixer   │           │   │
│  │   │  (Config) │  │ (Security)│  │(Validate) │  │ (Telemetry)│           │   │
│  │   └───────────┘  └───────────┘  └───────────┘  └───────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│  ┌─────────────────────────────────────┴───────────────────────────────────┐   │
│  │                         DATA PLANE                                       │   │
│  │                                                                          │   │
│  │  ┌─────────────────┐      mTLS       ┌─────────────────┐                │   │
│  │  │  Service Pod    │◀──────────────▶│  Service Pod    │                │   │
│  │  │ ┌─────────────┐ │                 │ ┌─────────────┐ │                │   │
│  │  │ │   Service   │ │                 │ │   Service   │ │                │   │
│  │  │ └─────────────┘ │                 │ └─────────────┘ │                │   │
│  │  │ ┌─────────────┐ │                 │ ┌─────────────┐ │                │   │
│  │  │ │Envoy Sidecar│ │                 │ │Envoy Sidecar│ │                │   │
│  │  │ └─────────────┘ │                 │ └─────────────┘ │                │   │
│  │  └─────────────────┘                 └─────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  FEATURES: mTLS │ Traffic Management │ Circuit Breaking │ Observability        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Service Catalog

### 2.1 Core Services Overview

| Service | Port | Database | Events Published | Events Consumed |
|---------|------|----------|------------------|-----------------|
| **api-gateway** | 8000 | - | - | - |
| **orchestrator-service** | 8001 | Redis | `session.*`, `response.*` | All events |
| **safety-service** | 8002 | PostgreSQL, Redis | `safety.*`, `crisis.*` | `session.*`, `message.*` |
| **memory-service** | 8003 | PostgreSQL, Weaviate, Redis | `memory.*` | `session.*`, `assessment.*` |
| **diagnosis-service** | 8004 | PostgreSQL | `diagnosis.*`, `assessment.*` | `session.*`, `memory.*` |
| **therapy-service** | 8005 | PostgreSQL | `therapy.*`, `intervention.*` | `diagnosis.*`, `memory.*` |
| **personality-service** | 8006 | PostgreSQL, Weaviate | `personality.*` | `session.*`, `memory.*` |
| **user-service** | 8007 | PostgreSQL | `user.*` | - |
| **notification-service** | 8008 | PostgreSQL | `notification.*` | `crisis.*`, `session.*` |
| **analytics-service** | 8009 | PostgreSQL, ClickHouse | - | All events |

### 2.2 Service Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SERVICE RESPONSIBILITIES                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  🛡️ SAFETY SERVICE                        🧠 MEMORY SERVICE                     │
│  ─────────────────                        ────────────────                       │
│  • Crisis detection (multi-layer)         • 5-tier memory hierarchy             │
│  • Risk assessment                        • Context assembly                     │
│  • Escalation management                  • Working memory (Redis)              │
│  • Safety plan storage                    • Episodic memory (PostgreSQL)        │
│  • Contraindication checking              • Semantic memory (Weaviate)          │
│  • Response filtering                     • Memory consolidation                 │
│                                           • Ebbinghaus decay                     │
│                                                                                  │
│  🔍 DIAGNOSIS SERVICE                     💆 THERAPY SERVICE                    │
│  ───────────────────                      ─────────────────                      │
│  • AMIE-inspired 4-step reasoning         • Stepped care routing                │
│  • Symptom extraction                     • CBT/DBT/ACT/MI techniques           │
│  • Differential generation                • Session state management            │
│  • DSM-5-TR/HiTOP mapping                 • Treatment planning                  │
│  • Confidence calibration                 • Homework management                 │
│  • Anti-sycophancy (Devil's Advocate)     • Progress tracking                   │
│                                                                                  │
│  🎭 PERSONALITY SERVICE                   🎼 ORCHESTRATOR SERVICE               │
│  ─────────────────────                    ───────────────────────               │
│  • Big Five (OCEAN) detection             • LangGraph multi-agent               │
│  • Ensemble ML (RoBERTa + LLM)            • Supervisor agent                    │
│  • Style adaptation                       • Request routing                     │
│  • MoEL empathy generation                • State management                    │
│  • Cultural adaptation                    • Response aggregation                │
│  • Profile management                     • Safety coordination                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Shared Libraries

### 3.1 Library Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SHARED LIBRARIES                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  solace-common/                    solace-events/                               │
│  ──────────────                    ───────────────                              │
│  • Base entities                   • Event schemas (Avro/JSON)                  │
│  • Value objects                   • Event publisher                            │
│  • Domain primitives               • Event consumer                             │
│  • Exceptions                      • Dead letter handling                       │
│  • Utilities                       • Kafka configuration                        │
│                                                                                  │
│  solace-infrastructure/            solace-security/                             │
│  ──────────────────────            ─────────────────                            │
│  • Database clients                • JWT authentication                         │
│  • Redis client                    • Authorization (RBAC/ABAC)                  │
│  • Weaviate client                 • Encryption utilities                       │
│  • Health checks                   • Audit logging                              │
│  • Observability                   • PHI protection                             │
│                                                                                  │
│  solace-ml/                        solace-testing/                              │
│  ──────────                        ───────────────                              │
│  • LLM client abstraction          • Test fixtures                              │
│  • Embedding models                • Mock services                              │
│  • Feature extraction              • Integration test utils                     │
│  • Model inference                 • Contract testing                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase-by-Phase Implementation

### Phase Overview

| Phase | Name | Services/Libraries | Batches | Files |
|-------|------|-------------------|---------|-------|
| **1** | Shared Libraries | 6 libraries | 6 | 30 |
| **2** | Infrastructure Services | 2 services | 4 | 20 |
| **3** | Safety Service | 1 service | 3 | 15 |
| **4** | Memory Service | 1 service | 4 | 20 |
| **5** | Diagnosis Service | 1 service | 3 | 15 |
| **6** | Therapy Service | 1 service | 3 | 15 |
| **7** | Personality Service | 1 service | 3 | 15 |
| **8** | Orchestrator Service | 1 service | 4 | 20 |
| **9** | API Gateway & User Service | 2 services | 3 | 15 |
| **10** | Supporting Services | 2 services | 3 | 15 |
| **TOTAL** | | **18 deployables** | **36** | **180** |

---

### PHASE 1: SHARED LIBRARIES

**Purpose**: Build reusable libraries shared across all microservices.

#### Batch 1.1: solace-common (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entity.py` | `libs/solace-common/src/domain/entity.py` | ~200 | Base Entity with identity, timestamps, versioning |
| `value_object.py` | `libs/solace-common/src/domain/value_object.py` | ~180 | Immutable value objects with validation |
| `aggregate.py` | `libs/solace-common/src/domain/aggregate.py` | ~220 | Aggregate root with domain events |
| `exceptions.py` | `libs/solace-common/src/exceptions.py` | ~350 | Exception hierarchy (Domain/Application/Infrastructure) |
| `utils.py` | `libs/solace-common/src/utils.py` | ~300 | Common utilities (datetime, crypto, validation) |

#### Batch 1.2: solace-events (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `schemas.py` | `libs/solace-events/src/schemas.py` | ~400 | All event schemas with Pydantic validation |
| `publisher.py` | `libs/solace-events/src/publisher.py` | ~350 | Transactional event publisher with outbox |
| `consumer.py` | `libs/solace-events/src/consumer.py` | ~380 | Consumer group management, offset tracking |
| `dead_letter.py` | `libs/solace-events/src/dead_letter.py` | ~220 | DLQ handling with retry policies |
| `config.py` | `libs/solace-events/src/config.py` | ~150 | Kafka configuration and topic management |

#### Batch 1.3: solace-infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `postgres.py` | `libs/solace-infrastructure/src/postgres.py` | ~350 | Async PostgreSQL client with connection pooling |
| `redis.py` | `libs/solace-infrastructure/src/redis.py` | ~300 | Redis cluster client with pub/sub |
| `weaviate.py` | `libs/solace-infrastructure/src/weaviate.py` | ~350 | Weaviate client with schema management |
| `health.py` | `libs/solace-infrastructure/src/health.py` | ~200 | Health check utilities for all backends |
| `observability.py` | `libs/solace-infrastructure/src/observability.py` | ~300 | Logging, metrics, tracing utilities |

#### Batch 1.4: solace-security (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `auth.py` | `libs/solace-security/src/auth.py` | ~350 | JWT validation, token management |
| `authorization.py` | `libs/solace-security/src/authorization.py` | ~300 | RBAC/ABAC policy enforcement |
| `encryption.py` | `libs/solace-security/src/encryption.py` | ~250 | AES-256 encryption for PHI |
| `audit.py` | `libs/solace-security/src/audit.py` | ~280 | Audit logging with immutability |
| `phi_protection.py` | `libs/solace-security/src/phi_protection.py` | ~220 | PHI detection and masking |

#### Batch 1.5: solace-ml (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `llm_client.py` | `libs/solace-ml/src/llm_client.py` | ~380 | Abstract LLM client with provider switching |
| `anthropic.py` | `libs/solace-ml/src/anthropic.py` | ~300 | Claude adapter with streaming |
| `openai.py` | `libs/solace-ml/src/openai.py` | ~280 | OpenAI adapter with function calling |
| `embeddings.py` | `libs/solace-ml/src/embeddings.py` | ~320 | Text embedding service |
| `inference.py` | `libs/solace-ml/src/inference.py` | ~300 | Model inference utilities |

#### Batch 1.6: solace-testing (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `fixtures.py` | `libs/solace-testing/src/fixtures.py` | ~350 | Common pytest fixtures |
| `mocks.py` | `libs/solace-testing/src/mocks.py` | ~300 | Mock services and clients |
| `factories.py` | `libs/solace-testing/src/factories.py` | ~280 | Test data factories |
| `integration.py` | `libs/solace-testing/src/integration.py` | ~250 | Integration test utilities |
| `contracts.py` | `libs/solace-testing/src/contracts.py` | ~200 | Contract testing helpers |

---

### PHASE 2: INFRASTRUCTURE SERVICES

**Purpose**: Deploy foundational infrastructure services.

#### Batch 2.1: Configuration Service (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `settings.py` | `services/config-service/src/settings.py` | ~350 | Centralized configuration management |
| `secrets.py` | `services/config-service/src/secrets.py` | ~250 | Secrets management (Vault/AWS SM) |
| `feature_flags.py` | `services/config-service/src/feature_flags.py` | ~200 | Feature flag management |
| `api.py` | `services/config-service/src/api.py` | ~280 | Configuration API endpoints |
| `main.py` | `services/config-service/src/main.py` | ~150 | FastAPI application entry |

#### Batch 2.2: Event Bus Setup (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `topics.py` | `infrastructure/kafka/topics.py` | ~300 | Topic definitions and configurations |
| `schemas.py` | `infrastructure/kafka/schemas.py` | ~400 | Schema registry management |
| `partitioning.py` | `infrastructure/kafka/partitioning.py` | ~200 | Partitioning strategies |
| `retention.py` | `infrastructure/kafka/retention.py` | ~180 | Retention policies |
| `monitoring.py` | `infrastructure/kafka/monitoring.py` | ~220 | Kafka monitoring setup |

#### Batch 2.3: Database Migrations (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `base_models.py` | `infrastructure/database/base_models.py` | ~300 | Base SQLAlchemy models |
| `migrations_runner.py` | `infrastructure/database/migrations_runner.py` | ~250 | Alembic migration runner |
| `seed_data.py` | `infrastructure/database/seed_data.py` | ~350 | Initial seed data |
| `weaviate_schema.py` | `infrastructure/database/weaviate_schema.py` | ~300 | Weaviate collections setup |
| `redis_setup.py` | `infrastructure/database/redis_setup.py` | ~150 | Redis cluster configuration |

#### Batch 2.4: Observability Stack (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `prometheus_config.py` | `infrastructure/observability/prometheus_config.py` | ~250 | Prometheus configuration |
| `grafana_dashboards.py` | `infrastructure/observability/grafana_dashboards.py` | ~300 | Dashboard definitions |
| `jaeger_config.py` | `infrastructure/observability/jaeger_config.py` | ~180 | Distributed tracing setup |
| `alerting_rules.py` | `infrastructure/observability/alerting_rules.py` | ~280 | AlertManager rules |
| `log_aggregation.py` | `infrastructure/observability/log_aggregation.py` | ~200 | ELK stack configuration |

---

### PHASE 3: SAFETY SERVICE

**Purpose**: Implement the always-active safety monitoring service (CRITICAL - First Domain Service).

#### Batch 3.1: Safety Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/safety-service/src/main.py` | ~200 | FastAPI application with middleware |
| `api.py` | `services/safety-service/src/api.py` | ~350 | Safety check endpoints |
| `service.py` | `services/safety-service/src/domain/service.py` | ~400 | Main safety orchestration |
| `crisis_detector.py` | `services/safety-service/src/domain/crisis_detector.py` | ~380 | Multi-layer crisis detection |
| `escalation.py` | `services/safety-service/src/domain/escalation.py` | ~350 | Escalation workflow management |

#### Batch 3.2: Safety Service - Domain (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/safety-service/src/domain/entities.py` | ~300 | SafetyAssessment, SafetyPlan entities |
| `value_objects.py` | `services/safety-service/src/domain/value_objects.py` | ~250 | RiskFactor, CrisisLevel value objects |
| `repository.py` | `services/safety-service/src/infrastructure/repository.py` | ~320 | Safety data persistence |
| `events.py` | `services/safety-service/src/events.py` | ~200 | Safety event publishers/consumers |
| `config.py` | `services/safety-service/src/config.py` | ~150 | Service configuration |

#### Batch 3.3: Safety Service - ML Components (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `keyword_detector.py` | `services/safety-service/src/ml/keyword_detector.py` | ~300 | Fast keyword-based crisis detection |
| `sentiment_analyzer.py` | `services/safety-service/src/ml/sentiment_analyzer.py` | ~280 | Sentiment analysis for risk |
| `pattern_matcher.py` | `services/safety-service/src/ml/pattern_matcher.py` | ~320 | Pattern-based risk detection |
| `llm_assessor.py` | `services/safety-service/src/ml/llm_assessor.py` | ~350 | LLM-based deep risk assessment |
| `contraindication.py` | `services/safety-service/src/ml/contraindication.py` | ~280 | Technique contraindication checker |

---

### PHASE 4: MEMORY SERVICE

**Purpose**: Implement the 5-tier memory hierarchy service.

#### Batch 4.1: Memory Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/memory-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/memory-service/src/api.py` | ~380 | Memory CRUD and query endpoints |
| `service.py` | `services/memory-service/src/domain/service.py` | ~400 | Main memory orchestration |
| `context_assembler.py` | `services/memory-service/src/domain/context_assembler.py` | ~350 | LLM context assembly |
| `consolidation.py` | `services/memory-service/src/domain/consolidation.py` | ~380 | Memory consolidation pipeline |

#### Batch 4.2: Memory Service - Tiers (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `working_memory.py` | `services/memory-service/src/domain/working_memory.py` | ~350 | Tier 1-2: Input buffer, working memory |
| `session_memory.py` | `services/memory-service/src/domain/session_memory.py` | ~320 | Tier 3: Session memory |
| `episodic_memory.py` | `services/memory-service/src/domain/episodic_memory.py` | ~350 | Tier 4: Past sessions, events |
| `semantic_memory.py` | `services/memory-service/src/domain/semantic_memory.py` | ~350 | Tier 5: Facts, knowledge graph |
| `decay_manager.py` | `services/memory-service/src/domain/decay_manager.py` | ~280 | Ebbinghaus decay implementation |

#### Batch 4.3: Memory Service - Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `postgres_repo.py` | `services/memory-service/src/infrastructure/postgres_repo.py` | ~350 | PostgreSQL repository |
| `weaviate_repo.py` | `services/memory-service/src/infrastructure/weaviate_repo.py` | ~380 | Weaviate vector repository |
| `redis_cache.py` | `services/memory-service/src/infrastructure/redis_cache.py` | ~300 | Redis working memory cache |
| `hybrid_search.py` | `services/memory-service/src/infrastructure/hybrid_search.py` | ~320 | BM25 + semantic hybrid search |
| `rag_pipeline.py` | `services/memory-service/src/infrastructure/rag_pipeline.py` | ~350 | Agentic Corrective RAG |

#### Batch 4.4: Memory Service - Domain (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/memory-service/src/domain/entities.py` | ~320 | MemoryRecord, UserProfile, SessionSummary |
| `value_objects.py` | `services/memory-service/src/domain/value_objects.py` | ~250 | RetentionPolicy, MemoryTier |
| `events.py` | `services/memory-service/src/events.py` | ~220 | Memory event publishers/consumers |
| `config.py` | `services/memory-service/src/config.py` | ~150 | Service configuration |
| `knowledge_graph.py` | `services/memory-service/src/domain/knowledge_graph.py` | ~350 | Triple extraction and graph queries |

---

### PHASE 5: DIAGNOSIS SERVICE

**Purpose**: Implement AMIE-inspired diagnostic assessment service.

#### Batch 5.1: Diagnosis Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/diagnosis-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/diagnosis-service/src/api.py` | ~350 | Diagnosis and assessment endpoints |
| `service.py` | `services/diagnosis-service/src/domain/service.py` | ~400 | 4-step Chain-of-Reasoning orchestration |
| `symptom_extractor.py` | `services/diagnosis-service/src/domain/symptom_extractor.py` | ~350 | Symptom extraction from conversation |
| `differential.py` | `services/diagnosis-service/src/domain/differential.py` | ~380 | DSM-5-TR/HiTOP differential generation |

#### Batch 5.2: Diagnosis Service - Anti-Sycophancy (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `advocate.py` | `services/diagnosis-service/src/domain/advocate.py` | ~350 | Devil's Advocate challenger |
| `confidence.py` | `services/diagnosis-service/src/domain/confidence.py` | ~300 | Sample consistency calibration |
| `clinical_codes.py` | `services/diagnosis-service/src/domain/clinical_codes.py` | ~280 | DSM-5-TR/ICD-11 code mapping |
| `severity.py` | `services/diagnosis-service/src/domain/severity.py` | ~250 | PHQ-9/GAD-7 severity assessment |
| `evidence.py` | `services/diagnosis-service/src/domain/evidence.py` | ~320 | Evidence-based hypothesis support |

#### Batch 5.3: Diagnosis Service - Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/diagnosis-service/src/domain/entities.py` | ~300 | Diagnosis, Symptom entities |
| `value_objects.py` | `services/diagnosis-service/src/domain/value_objects.py` | ~250 | ClinicalHypothesis, SeverityLevel |
| `repository.py` | `services/diagnosis-service/src/infrastructure/repository.py` | ~320 | Diagnosis persistence |
| `events.py` | `services/diagnosis-service/src/events.py` | ~200 | Diagnosis events |
| `config.py` | `services/diagnosis-service/src/config.py` | ~150 | Service configuration |

---

### PHASE 6: THERAPY SERVICE

**Purpose**: Implement evidence-based therapeutic intervention service.

#### Batch 6.1: Therapy Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/therapy-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/therapy-service/src/api.py` | ~350 | Therapy session and intervention endpoints |
| `service.py` | `services/therapy-service/src/domain/service.py` | ~400 | Hybrid rules+LLM therapy orchestration |
| `technique_selector.py` | `services/therapy-service/src/domain/technique_selector.py` | ~380 | CBT/DBT/ACT/MI technique selection |
| `session_manager.py` | `services/therapy-service/src/domain/session_manager.py` | ~350 | Session state machine |

#### Batch 6.2: Therapy Service - Treatment (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `treatment_planner.py` | `services/therapy-service/src/domain/treatment_planner.py` | ~380 | Stepped care treatment planning |
| `homework.py` | `services/therapy-service/src/domain/homework.py` | ~300 | Homework assignment and tracking |
| `progress.py` | `services/therapy-service/src/domain/progress.py` | ~320 | Progress tracking and outcomes |
| `modalities.py` | `services/therapy-service/src/domain/modalities.py` | ~350 | CBT/DBT/ACT/MI modality implementations |
| `interventions.py` | `services/therapy-service/src/domain/interventions.py` | ~350 | Intervention delivery |

#### Batch 6.3: Therapy Service - Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/therapy-service/src/domain/entities.py` | ~320 | TreatmentPlan, TherapySession |
| `value_objects.py` | `services/therapy-service/src/domain/value_objects.py` | ~280 | Technique, OutcomeMeasure |
| `repository.py` | `services/therapy-service/src/infrastructure/repository.py` | ~320 | Therapy persistence |
| `events.py` | `services/therapy-service/src/events.py` | ~200 | Therapy events |
| `config.py` | `services/therapy-service/src/config.py` | ~150 | Service configuration |

---

### PHASE 7: PERSONALITY SERVICE

**Purpose**: Implement Big Five personality detection and adaptation service.

#### Batch 7.1: Personality Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/personality-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/personality-service/src/api.py` | ~350 | Personality detection endpoints |
| `service.py` | `services/personality-service/src/domain/service.py` | ~380 | Personality detection orchestration |
| `trait_detector.py` | `services/personality-service/src/domain/trait_detector.py` | ~350 | OCEAN trait ensemble detection |
| `style_adapter.py` | `services/personality-service/src/domain/style_adapter.py` | ~320 | Communication style mapping |

#### Batch 7.2: Personality Service - ML (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `roberta_model.py` | `services/personality-service/src/ml/roberta_model.py` | ~350 | Fine-tuned RoBERTa classifier |
| `llm_detector.py` | `services/personality-service/src/ml/llm_detector.py` | ~300 | Zero-shot LLM personality detection |
| `liwc_features.py` | `services/personality-service/src/ml/liwc_features.py` | ~320 | LIWC feature extraction |
| `multimodal.py` | `services/personality-service/src/ml/multimodal.py` | ~350 | Late fusion multimodal analysis |
| `empathy.py` | `services/personality-service/src/ml/empathy.py` | ~350 | MoEL empathy generation |

#### Batch 7.3: Personality Service - Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/personality-service/src/domain/entities.py` | ~300 | PersonalityProfile, TraitAssessment |
| `value_objects.py` | `services/personality-service/src/domain/value_objects.py` | ~250 | OceanScores, CommunicationStyle |
| `repository.py` | `services/personality-service/src/infrastructure/repository.py` | ~320 | Personality persistence |
| `events.py` | `services/personality-service/src/events.py` | ~200 | Personality events |
| `config.py` | `services/personality-service/src/config.py` | ~150 | Service configuration |

---

### PHASE 8: ORCHESTRATOR SERVICE

**Purpose**: Implement LangGraph multi-agent orchestration service.

#### Batch 8.1: Orchestrator Service - Core (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/orchestrator-service/src/main.py` | ~200 | FastAPI application with WebSocket |
| `api.py` | `services/orchestrator-service/src/api.py` | ~380 | Chat endpoints, WebSocket handler |
| `graph_builder.py` | `services/orchestrator-service/src/langgraph/graph_builder.py` | ~400 | LangGraph state graph construction |
| `state_schema.py` | `services/orchestrator-service/src/langgraph/state_schema.py` | ~300 | Typed state with checkpointing |
| `supervisor.py` | `services/orchestrator-service/src/langgraph/supervisor.py` | ~350 | Supervisor agent node |

#### Batch 8.2: Orchestrator Service - Agents (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `safety_agent.py` | `services/orchestrator-service/src/agents/safety_agent.py` | ~350 | Safety monitoring agent |
| `diagnosis_agent.py` | `services/orchestrator-service/src/agents/diagnosis_agent.py` | ~320 | Diagnosis coordination agent |
| `therapy_agent.py` | `services/orchestrator-service/src/agents/therapy_agent.py` | ~320 | Therapy coordination agent |
| `personality_agent.py` | `services/orchestrator-service/src/agents/personality_agent.py` | ~300 | Personality adaptation agent |
| `chat_agent.py` | `services/orchestrator-service/src/agents/chat_agent.py` | ~280 | General conversation agent |

#### Batch 8.3: Orchestrator Service - Response (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `router.py` | `services/orchestrator-service/src/langgraph/router.py` | ~300 | Intent classification and routing |
| `aggregator.py` | `services/orchestrator-service/src/langgraph/aggregator.py` | ~320 | Response aggregation |
| `response_generator.py` | `services/orchestrator-service/src/response/generator.py` | ~380 | Final response generation |
| `style_applicator.py` | `services/orchestrator-service/src/response/style_applicator.py` | ~300 | Personality style application |
| `safety_wrapper.py` | `services/orchestrator-service/src/response/safety_wrapper.py` | ~280 | Safety wrapping and resources |

#### Batch 8.4: Orchestrator Service - Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `service_clients.py` | `services/orchestrator-service/src/infrastructure/clients.py` | ~350 | Service-to-service HTTP clients |
| `state_persistence.py` | `services/orchestrator-service/src/infrastructure/state.py` | ~300 | LangGraph state persistence |
| `events.py` | `services/orchestrator-service/src/events.py` | ~220 | Orchestrator events |
| `config.py` | `services/orchestrator-service/src/config.py` | ~150 | Service configuration |
| `websocket.py` | `services/orchestrator-service/src/websocket.py` | ~280 | WebSocket connection management |

---

### PHASE 9: API GATEWAY & USER SERVICE

**Purpose**: Implement API Gateway configuration and User Service.

#### Batch 9.1: API Gateway (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `kong_config.py` | `infrastructure/api-gateway/kong_config.py` | ~350 | Kong gateway configuration |
| `routes.py` | `infrastructure/api-gateway/routes.py` | ~300 | Route definitions |
| `rate_limiting.py` | `infrastructure/api-gateway/rate_limiting.py` | ~200 | Rate limiting policies |
| `auth_plugin.py` | `infrastructure/api-gateway/auth_plugin.py` | ~280 | JWT authentication plugin |
| `cors.py` | `infrastructure/api-gateway/cors.py` | ~150 | CORS configuration |

#### Batch 9.2: User Service (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/user-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/user-service/src/api.py` | ~350 | User CRUD endpoints |
| `service.py` | `services/user-service/src/domain/service.py` | ~350 | User domain service |
| `repository.py` | `services/user-service/src/infrastructure/repository.py` | ~300 | User persistence |
| `auth.py` | `services/user-service/src/auth.py` | ~320 | Authentication and sessions |

#### Batch 9.3: User Service - Domain (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `entities.py` | `services/user-service/src/domain/entities.py` | ~280 | User, UserPreferences entities |
| `value_objects.py` | `services/user-service/src/domain/value_objects.py` | ~200 | UserRole, Consent value objects |
| `events.py` | `services/user-service/src/events.py` | ~180 | User events |
| `config.py` | `services/user-service/src/config.py` | ~150 | Service configuration |
| `consent.py` | `services/user-service/src/domain/consent.py` | ~250 | Consent management |

---

### PHASE 10: SUPPORTING SERVICES

**Purpose**: Implement notification and analytics services.

#### Batch 10.1: Notification Service (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/notification-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/notification-service/src/api.py` | ~280 | Notification endpoints |
| `service.py` | `services/notification-service/src/domain/service.py` | ~350 | Notification orchestration |
| `channels.py` | `services/notification-service/src/domain/channels.py` | ~300 | Email, SMS, Push channels |
| `templates.py` | `services/notification-service/src/domain/templates.py` | ~250 | Notification templates |

#### Batch 10.2: Analytics Service (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `main.py` | `services/analytics-service/src/main.py` | ~200 | FastAPI application |
| `api.py` | `services/analytics-service/src/api.py` | ~300 | Analytics query endpoints |
| `consumer.py` | `services/analytics-service/src/consumer.py` | ~350 | Event consumer for analytics |
| `aggregations.py` | `services/analytics-service/src/aggregations.py` | ~320 | Metrics aggregation |
| `reports.py` | `services/analytics-service/src/reports.py` | ~280 | Report generation |

#### Batch 10.3: Supporting Infrastructure (5 files)

| File | Path | LOC | Responsibility |
|------|------|-----|----------------|
| `notification_entities.py` | `services/notification-service/src/domain/entities.py` | ~200 | Notification entities |
| `notification_events.py` | `services/notification-service/src/events.py` | ~180 | Notification events |
| `analytics_repository.py` | `services/analytics-service/src/repository.py` | ~300 | ClickHouse repository |
| `notification_config.py` | `services/notification-service/src/config.py` | ~150 | Notification config |
| `analytics_config.py` | `services/analytics-service/src/config.py` | ~150 | Analytics config |

---

## 5. Directory Structure

```
solace-ai/
├── README.md
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
├── pyproject.toml
│
├── system-design/                    # Architecture documentation
│   ├── README.md
│   ├── IMPLEMENTATION-PLAN.md       # This document
│   ├── 00-system-integration/
│   ├── 01-diagnosis-module/
│   ├── 02-therapy-module/
│   ├── 03-personality-module/
│   └── 04-memory-module/
│
├── libs/                             # Shared libraries
│   ├── solace-common/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── domain/
│   │   │   │   ├── entity.py
│   │   │   │   ├── value_object.py
│   │   │   │   └── aggregate.py
│   │   │   ├── exceptions.py
│   │   │   └── utils.py
│   │   └── tests/
│   │
│   ├── solace-events/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── schemas.py
│   │   │   ├── publisher.py
│   │   │   ├── consumer.py
│   │   │   ├── dead_letter.py
│   │   │   └── config.py
│   │   └── tests/
│   │
│   ├── solace-infrastructure/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── postgres.py
│   │   │   ├── redis.py
│   │   │   ├── weaviate.py
│   │   │   ├── health.py
│   │   │   └── observability.py
│   │   └── tests/
│   │
│   ├── solace-security/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── auth.py
│   │   │   ├── authorization.py
│   │   │   ├── encryption.py
│   │   │   ├── audit.py
│   │   │   └── phi_protection.py
│   │   └── tests/
│   │
│   ├── solace-ml/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── llm_client.py
│   │   │   ├── anthropic.py
│   │   │   ├── openai.py
│   │   │   ├── embeddings.py
│   │   │   └── inference.py
│   │   └── tests/
│   │
│   └── solace-testing/
│       ├── pyproject.toml
│       ├── src/
│       │   ├── fixtures.py
│       │   ├── mocks.py
│       │   ├── factories.py
│       │   ├── integration.py
│       │   └── contracts.py
│       └── tests/
│
├── services/                         # Microservices
│   │
│   ├── orchestrator-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── websocket.py
│   │   │   ├── langgraph/
│   │   │   │   ├── graph_builder.py
│   │   │   │   ├── state_schema.py
│   │   │   │   ├── supervisor.py
│   │   │   │   ├── router.py
│   │   │   │   └── aggregator.py
│   │   │   ├── agents/
│   │   │   │   ├── safety_agent.py
│   │   │   │   ├── diagnosis_agent.py
│   │   │   │   ├── therapy_agent.py
│   │   │   │   ├── personality_agent.py
│   │   │   │   └── chat_agent.py
│   │   │   ├── response/
│   │   │   │   ├── generator.py
│   │   │   │   ├── style_applicator.py
│   │   │   │   └── safety_wrapper.py
│   │   │   └── infrastructure/
│   │   │       ├── clients.py
│   │   │       └── state.py
│   │   └── tests/
│   │
│   ├── safety-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── crisis_detector.py
│   │   │   │   ├── escalation.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   ├── ml/
│   │   │   │   ├── keyword_detector.py
│   │   │   │   ├── sentiment_analyzer.py
│   │   │   │   ├── pattern_matcher.py
│   │   │   │   ├── llm_assessor.py
│   │   │   │   └── contraindication.py
│   │   │   └── infrastructure/
│   │   │       └── repository.py
│   │   └── tests/
│   │
│   ├── memory-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── working_memory.py
│   │   │   │   ├── session_memory.py
│   │   │   │   ├── episodic_memory.py
│   │   │   │   ├── semantic_memory.py
│   │   │   │   ├── context_assembler.py
│   │   │   │   ├── consolidation.py
│   │   │   │   ├── decay_manager.py
│   │   │   │   ├── knowledge_graph.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   └── infrastructure/
│   │   │       ├── postgres_repo.py
│   │   │       ├── weaviate_repo.py
│   │   │       ├── redis_cache.py
│   │   │       ├── hybrid_search.py
│   │   │       └── rag_pipeline.py
│   │   └── tests/
│   │
│   ├── diagnosis-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── symptom_extractor.py
│   │   │   │   ├── differential.py
│   │   │   │   ├── advocate.py
│   │   │   │   ├── confidence.py
│   │   │   │   ├── clinical_codes.py
│   │   │   │   ├── severity.py
│   │   │   │   ├── evidence.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   └── infrastructure/
│   │   │       └── repository.py
│   │   └── tests/
│   │
│   ├── therapy-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── technique_selector.py
│   │   │   │   ├── session_manager.py
│   │   │   │   ├── treatment_planner.py
│   │   │   │   ├── homework.py
│   │   │   │   ├── progress.py
│   │   │   │   ├── modalities.py
│   │   │   │   ├── interventions.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   └── infrastructure/
│   │   │       └── repository.py
│   │   └── tests/
│   │
│   ├── personality-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── trait_detector.py
│   │   │   │   ├── style_adapter.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   ├── ml/
│   │   │   │   ├── roberta_model.py
│   │   │   │   ├── llm_detector.py
│   │   │   │   ├── liwc_features.py
│   │   │   │   ├── multimodal.py
│   │   │   │   └── empathy.py
│   │   │   └── infrastructure/
│   │   │       └── repository.py
│   │   └── tests/
│   │
│   ├── user-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── auth.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── consent.py
│   │   │   │   ├── entities.py
│   │   │   │   └── value_objects.py
│   │   │   └── infrastructure/
│   │   │       └── repository.py
│   │   └── tests/
│   │
│   ├── notification-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api.py
│   │   │   ├── config.py
│   │   │   ├── events.py
│   │   │   ├── domain/
│   │   │   │   ├── service.py
│   │   │   │   ├── channels.py
│   │   │   │   ├── templates.py
│   │   │   │   └── entities.py
│   │   │   └── infrastructure/
│   │   └── tests/
│   │
│   └── analytics-service/
│       ├── Dockerfile
│       ├── pyproject.toml
│       ├── src/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── api.py
│       │   ├── config.py
│       │   ├── consumer.py
│       │   ├── aggregations.py
│       │   ├── reports.py
│       │   └── repository.py
│       └── tests/
│
├── infrastructure/                   # Infrastructure as Code
│   ├── api-gateway/
│   │   ├── kong_config.py
│   │   ├── routes.py
│   │   ├── rate_limiting.py
│   │   ├── auth_plugin.py
│   │   └── cors.py
│   │
│   ├── kafka/
│   │   ├── topics.py
│   │   ├── schemas.py
│   │   ├── partitioning.py
│   │   ├── retention.py
│   │   └── monitoring.py
│   │
│   ├── database/
│   │   ├── base_models.py
│   │   ├── migrations_runner.py
│   │   ├── seed_data.py
│   │   ├── weaviate_schema.py
│   │   └── redis_setup.py
│   │
│   ├── observability/
│   │   ├── prometheus_config.py
│   │   ├── grafana_dashboards.py
│   │   ├── jaeger_config.py
│   │   ├── alerting_rules.py
│   │   └── log_aggregation.py
│   │
│   └── kubernetes/
│       ├── namespaces/
│       ├── deployments/
│       ├── services/
│       ├── configmaps/
│       ├── secrets/
│       ├── istio/
│       └── helm/
│
├── scripts/                          # Development scripts
│   ├── setup.sh
│   ├── dev.sh
│   ├── test.sh
│   ├── lint.sh
│   ├── build.sh
│   └── deploy.sh
│
└── tests/                            # Integration & E2E tests
    ├── integration/
    ├── e2e/
    ├── load/
    └── contracts/
```

---

## 6. Deployment Architecture

### 6.1 Kubernetes Deployment

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      KUBERNETES DEPLOYMENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  NAMESPACE: solace-prod                                                          │
│  ─────────────────────                                                           │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         INGRESS LAYER                                    │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │   │
│  │  │  Ingress Nginx   │  │  Cert Manager    │  │   ExternalDNS    │      │   │
│  │  │  (L7 Routing)    │  │  (TLS Certs)     │  │  (DNS Records)   │      │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       APPLICATION LAYER                                  │   │
│  │                                                                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │  Orchestrator  │  │    Safety      │  │    Memory      │            │   │
│  │  │   Deployment   │  │   Deployment   │  │   Deployment   │            │   │
│  │  │   (3 replicas) │  │   (3 replicas) │  │   (3 replicas) │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  │                                                                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │   Diagnosis    │  │    Therapy     │  │  Personality   │            │   │
│  │  │   Deployment   │  │   Deployment   │  │   Deployment   │            │   │
│  │  │   (2 replicas) │  │   (2 replicas) │  │   (2 replicas) │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  │                                                                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │     User       │  │  Notification  │  │   Analytics    │            │   │
│  │  │   Deployment   │  │   Deployment   │  │   Deployment   │            │   │
│  │  │   (2 replicas) │  │   (2 replicas) │  │   (2 replicas) │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  NAMESPACE: solace-data                                                          │
│  ─────────────────────                                                           │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA LAYER                                      │   │
│  │                                                                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │   PostgreSQL   │  │     Redis      │  │     Kafka      │            │   │
│  │  │  StatefulSet   │  │   StatefulSet  │  │  StatefulSet   │            │   │
│  │  │  (3 replicas)  │  │  (6 replicas)  │  │  (3 replicas)  │            │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Service Scaling Guidelines

| Service | Min Replicas | Max Replicas | CPU Request | Memory Request |
|---------|--------------|--------------|-------------|----------------|
| **orchestrator** | 3 | 10 | 500m | 1Gi |
| **safety** | 3 | 10 | 500m | 1Gi |
| **memory** | 3 | 8 | 500m | 2Gi |
| **diagnosis** | 2 | 6 | 500m | 1Gi |
| **therapy** | 2 | 6 | 500m | 1Gi |
| **personality** | 2 | 6 | 1000m | 2Gi |
| **user** | 2 | 4 | 250m | 512Mi |
| **notification** | 2 | 4 | 250m | 512Mi |
| **analytics** | 2 | 4 | 500m | 1Gi |

---

## 7. Implementation Execution Order

### Execution Timeline

```
PHASE 1: SHARED LIBRARIES (Week 1-2)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 1.1: solace-common        ─────► Foundation entities, exceptions
├── Batch 1.2: solace-events        ─────► Kafka event infrastructure
├── Batch 1.3: solace-infrastructure ────► Database clients
├── Batch 1.4: solace-security      ─────► Auth, encryption, audit
├── Batch 1.5: solace-ml            ─────► LLM client, embeddings
└── Batch 1.6: solace-testing       ─────► Test utilities
│
└── [Gate: All libraries compile, unit tests pass]

PHASE 2: INFRASTRUCTURE SERVICES (Week 3)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 2.1: Configuration Service
├── Batch 2.2: Kafka Event Bus Setup
├── Batch 2.3: Database Migrations
└── Batch 2.4: Observability Stack
│
└── [Gate: Infrastructure healthy, Kafka topics created, DBs migrated]

PHASE 3: SAFETY SERVICE (Week 4) ⚠️ CRITICAL PATH
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 3.1: Safety Service - Core
├── Batch 3.2: Safety Service - Domain
└── Batch 3.3: Safety Service - ML Components
│
└── [Gate: Crisis detection <10ms, escalation works, audit complete]

PHASE 4: MEMORY SERVICE (Week 5)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 4.1: Memory Service - Core
├── Batch 4.2: Memory Service - Tiers
├── Batch 4.3: Memory Service - Infrastructure
└── Batch 4.4: Memory Service - Domain
│
└── [Gate: 5-tier hierarchy works, context assembly <100ms, consolidation runs]

PHASE 5: DIAGNOSIS SERVICE (Week 6)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 5.1: Diagnosis Service - Core
├── Batch 5.2: Diagnosis Service - Anti-Sycophancy
└── Batch 5.3: Diagnosis Service - Infrastructure
│
└── [Gate: 4-step reasoning works, Devil's Advocate challenges, DSM-5-TR mapping]

PHASE 6: THERAPY SERVICE (Week 7)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 6.1: Therapy Service - Core
├── Batch 6.2: Therapy Service - Treatment
└── Batch 6.3: Therapy Service - Infrastructure
│
└── [Gate: Stepped care routing, modalities functional, homework tracking]

PHASE 7: PERSONALITY SERVICE (Week 8)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 7.1: Personality Service - Core
├── Batch 7.2: Personality Service - ML
└── Batch 7.3: Personality Service - Infrastructure
│
└── [Gate: OCEAN detection r>0.85, style adaptation, MoEL empathy]

PHASE 8: ORCHESTRATOR SERVICE (Week 9)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 8.1: Orchestrator Service - Core
├── Batch 8.2: Orchestrator Service - Agents
├── Batch 8.3: Orchestrator Service - Response
└── Batch 8.4: Orchestrator Service - Infrastructure
│
└── [Gate: LangGraph runs, agents coordinate, safety always active]

PHASE 9: API GATEWAY & USER SERVICE (Week 10)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 9.1: API Gateway
├── Batch 9.2: User Service
└── Batch 9.3: User Service - Domain
│
└── [Gate: Auth works, rate limiting active, CORS configured]

PHASE 10: SUPPORTING SERVICES (Week 11)
════════════════════════════════════════════════════════════════════════════════
│
├── Batch 10.1: Notification Service
├── Batch 10.2: Analytics Service
└── Batch 10.3: Supporting Infrastructure
│
└── [Gate: Notifications send, analytics consume events, reports generate]

WEEK 12: INTEGRATION & TESTING
════════════════════════════════════════════════════════════════════════════════
│
├── End-to-end integration testing
├── Load testing
├── Security penetration testing
├── HIPAA compliance verification
└── Production deployment preparation
```

### Critical Path Dependencies

```
                    ┌───────────────────┐
                    │ solace-common     │
                    │ solace-events     │
                    │ solace-infra      │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │  solace-  │  │  solace-  │  │  solace-  │
        │ security  │  │    ml     │  │ testing   │
        └─────┬─────┘  └─────┬─────┘  └───────────┘
              │               │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │    SAFETY     │ ◄── CRITICAL: Must be first domain service
              │    SERVICE    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │    MEMORY     │ ◄── Foundation for all other domains
              │    SERVICE    │
              └───────┬───────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ DIAGNOSIS │  │  THERAPY  │  │PERSONALITY│
│  SERVICE  │  │  SERVICE  │  │  SERVICE  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
              ┌───────▼───────┐
              │ ORCHESTRATOR  │
              │    SERVICE    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  API GATEWAY  │
              │ USER SERVICE  │
              └───────────────┘
```

---

## 8. Quality Gates & Acceptance Criteria

### Per-Batch Quality Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| **Compilation** | Zero errors, zero warnings | `make lint && make build` |
| **Completeness** | No TODOs, no pass statements, no stubs | Code review checklist |
| **Architecture** | Hexagonal boundaries respected | Architecture tests |
| **Complexity** | Cyclomatic < 15, LOC < 400 | Static analysis |
| **Dependencies** | All justified, no deprecated | `pip-audit` |
| **Tests** | Coverage > 80% | `pytest --cov` |
| **Documentation** | Public APIs documented | Docstring check |

### Per-Service Acceptance Criteria

| Service | SLO | Latency | Availability |
|---------|-----|---------|--------------|
| **orchestrator** | 99.9% | p99 < 500ms | Active-Active |
| **safety** | 99.99% | p99 < 100ms | Active-Active |
| **memory** | 99.9% | p99 < 200ms | Active-Active |
| **diagnosis** | 99.9% | p99 < 1000ms | Active-Passive |
| **therapy** | 99.9% | p99 < 500ms | Active-Passive |
| **personality** | 99.9% | p99 < 300ms | Active-Passive |

### HIPAA Compliance Checklist

- [ ] All PHI encrypted at rest (AES-256)
- [ ] All PHI encrypted in transit (TLS 1.3)
- [ ] Access logging enabled for all PHI
- [ ] Audit trails immutable (6-year retention)
- [ ] BAAs signed with all vendors
- [ ] Minimum necessary access enforced
- [ ] Automatic session timeout (15 min)
- [ ] MFA enabled for all admin access

---

## 9. Architecture Alignment: Critical Components

### 9.1 Four-Layer Safety Architecture (CRITICAL)

Per `02-therapy-module/ARCHITECTURE.md`, implement exactly this 4-layer safety system:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        4-LAYER SAFETY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 1: INPUT SAFETY GATE (<10ms)                                         ││
│  │ ─────────────────────────────────                                          ││
│  │ • Crisis keyword detection (regex + ML)                                    ││
│  │ • Sentiment analysis for distress signals                                  ││
│  │ • Context pattern recognition                                              ││
│  │ • Historical risk factor check from Memory                                 ││
│  │                                                                            ││
│  │ Keywords: "suicide", "kill myself", "end it", "no point", "can't go on"   ││
│  │ Action: IMMEDIATE escalation if matched                                    ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                        │                                        │
│                                        ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 2: CONTRAINDICATION CHECK (<100ms)                                   ││
│  │ ─────────────────────────────────────────                                  ││
│  │ • Technique-condition matching matrix                                      ││
│  │ • Severity appropriateness validation                                      ││
│  │ • Prerequisite verification (e.g., DBT requires distress tolerance first) ││
│  │ • Timing appropriateness (not exposure during crisis)                      ││
│  │                                                                            ││
│  │ Contraindication Matrix:                                                   ││
│  │   ABSOLUTE: Exposure therapy + Active psychosis                            ││
│  │   RELATIVE: Cognitive restructuring + Severe depression                    ││
│  │   TECHNIQUE-SPECIFIC: DBT diary card + First session                       ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                        │                                        │
│                                        ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 3: OUTPUT FILTERING (<500ms)                                         ││
│  │ ────────────────────────────────────                                       ││
│  │ • Content validation (no harmful advice)                                   ││
│  │ • Boundary enforcement (scope of practice)                                 ││
│  │ • Compassion check (empathy in responses)                                  ││
│  │ • Resource inclusion (crisis lines when appropriate)                       ││
│  │                                                                            ││
│  │ Forbidden: Medical diagnoses, medication advice, legal advice             ││
│  │ Required: Crisis resources if ANY safety concern detected                  ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                        │                                        │
│                                        ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │ LAYER 4: SESSION MONITORING (Continuous)                                   ││
│  │ ─────────────────────────────────────────                                  ││
│  │ • Engagement tracking (message frequency, length)                          ││
│  │ • Emotional trajectory (sentiment over session)                            ││
│  │ • Duration limits (recommend breaks after 60 min)                          ││
│  │ • Frequency monitoring (alert if >5 sessions/day)                          ││
│  │                                                                            ││
│  │ Deterioration signals: Sentiment drop >30%, Engagement drop >50%           ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Files (Safety Service):**

| File | Responsibility | Layer |
|------|----------------|-------|
| `input_gate.py` | Crisis keyword detection, sentiment, pattern matching | Layer 1 |
| `contraindication_matrix.py` | Technique-condition rules engine | Layer 2 |
| `output_filter.py` | Content validation, boundary enforcement, compassion | Layer 3 |
| `session_monitor.py` | Engagement tracking, trajectory analysis | Layer 4 |
| `crisis_escalation.py` | 3-level escalation protocol (Critical/High/Elevated) | Cross-layer |

### 9.2 Five-Tier Memory Hierarchy (CRITICAL)

Per `04-memory-module/ARCHITECTURE.md`, implement exactly this 5-tier system:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        5-TIER MEMORY HIERARCHY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TIER 1: INPUT BUFFER                    TIER 2: WORKING MEMORY                 │
│  ────────────────────                    ──────────────────────                 │
│  Storage: In-memory only                 Storage: Redis + In-memory             │
│  TTL: Request duration                   TTL: Session duration                  │
│  Access: <1ms                            Access: <10ms                          │
│  Size: Single message                    Size: 4K-8K tokens                     │
│  Purpose: Current processing             Purpose: LLM context window            │
│                                                                                  │
│  TOKEN BUDGET ALLOCATION:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ System Prompt:     500-1000 tokens                                      │   │
│  │ User Profile:      200-400 tokens                                       │   │
│  │ Retrieved Context: 1000-2000 tokens                                     │   │
│  │ Recent Messages:   2000-4000 tokens                                     │   │
│  │ Current Exchange:  Variable                                             │   │
│  │ Response Buffer:   1000-2000 tokens                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  TIER 3: SESSION MEMORY                  TIER 4: EPISODIC MEMORY                │
│  ───────────────────────                 ──────────────────────                 │
│  Storage: Redis + PostgreSQL             Storage: PostgreSQL + Weaviate         │
│  TTL: 24 hours after session             TTL: Decay-based (Ebbinghaus)          │
│  Access: <50ms                           Access: <200ms                         │
│  Size: Full session transcript           Size: Summarized sessions              │
│  Purpose: Current session state          Purpose: Past session retrieval        │
│                                                                                  │
│  TIER 5: SEMANTIC MEMORY                                                         │
│  ───────────────────────                                                         │
│  Storage: Weaviate + PostgreSQL                                                  │
│  TTL: Permanent (with versioning)                                               │
│  Access: <200ms                                                                  │
│  Size: Extracted facts only                                                      │
│  Purpose: User knowledge, facts, relationships                                   │
│                                                                                  │
│  ══════════════════════════════════════════════════════════════════════════════ │
│  EBBINGHAUS DECAY MODEL: R(t) = e^(-λt) × S                                     │
│  ──────────────────────────────────────────                                     │
│  R = Retention strength                                                          │
│  t = Time elapsed                                                                │
│  λ = Decay rate (base: 0.1/day)                                                 │
│  S = Stability (reinforcement multiplier: 1.5x per recall)                      │
│                                                                                  │
│  RETENTION CATEGORIES:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 🔒 PERMANENT (λ=0): Safety plans, crisis history, diagnoses, allergies │   │
│  │ 📚 LONG-TERM (λ=0.02): Treatment plans, milestones, key relationships  │   │
│  │ 📋 MEDIUM-TERM (λ=0.05): Session summaries, homework, patterns         │   │
│  │ 📝 SHORT-TERM (λ=0.15): Casual details, temporary context              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ⚠️  SAFETY-CRITICAL MEMORY NEVER DECAYS (λ=0 always)                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Files (Memory Service):**

| File | Tier | Responsibility |
|------|------|----------------|
| `input_buffer.py` | Tier 1 | Current message processing buffer |
| `working_memory.py` | Tier 2 | Redis-backed context window with token budgeting |
| `session_memory.py` | Tier 3 | Full session transcript storage |
| `episodic_memory.py` | Tier 4 | Past session summaries and retrieval |
| `semantic_memory.py` | Tier 5 | Knowledge graph, fact storage |
| `decay_manager.py` | Cross-tier | Ebbinghaus decay with safety override |
| `rag_pipeline.py` | Cross-tier | Agentic Corrective RAG with document grading |

### 9.3 AMIE 4-Step Chain-of-Reasoning (CRITICAL)

Per `01-diagnosis-module/ARCHITECTURE.md`, implement this exact reasoning flow:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    AMIE-INSPIRED 4-STEP CHAIN-OF-REASONING                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 1: ANALYZE (Initial Symptom Summary)                                       │
│  ─────────────────────────────────────────                                       │
│  Input: User messages, conversation history                                      │
│  Output: Structured symptom list with temporal markers                           │
│                                                                                  │
│  Extract:                                                                         │
│  • Presenting symptoms (what user explicitly states)                             │
│  • Onset information (when symptoms began)                                       │
│  • Duration (how long symptoms have persisted)                                   │
│  • Severity indicators (frequency, intensity, impact)                            │
│  • Triggering factors (contextual associations)                                  │
│  • Protective factors (what helps)                                               │
│                                                                                  │
│                                        │                                        │
│                                        ▼                                        │
│  STEP 2: HYPOTHESIZE (Differential Generation)                                   │
│  ──────────────────────────────────────────────                                  │
│  Input: Extracted symptoms, user history (from Memory)                           │
│  Output: Ordered list of clinical hypotheses with confidence                     │
│                                                                                  │
│  Process:                                                                         │
│  • Match symptoms to DSM-5-TR criteria clusters                                  │
│  • Apply HiTOP dimensional scoring (0-4 scale)                                   │
│  • Cross-reference with user's diagnostic history                                │
│  • Generate ranked differential with confidence intervals                        │
│                                                                                  │
│  Example Output:                                                                  │
│  1. Major Depressive Disorder, Moderate (ICD: F32.1) - 78% [72-84%]             │
│  2. Generalized Anxiety Disorder (ICD: F41.1) - 65% [58-72%]                    │
│  3. Adjustment Disorder with Mixed Anxiety/Depression - 45% [38-52%]            │
│                                                                                  │
│                                        │                                        │
│                                        ▼                                        │
│  STEP 3: CHALLENGE (Devil's Advocate - Anti-Sycophancy)                         │
│  ───────────────────────────────────────────────────────                         │
│  Input: Generated hypotheses                                                      │
│  Output: Challenged hypotheses with alternative interpretations                  │
│                                                                                  │
│  ⚠️  CRITICAL: This step prevents confirmation bias                             │
│                                                                                  │
│  Process:                                                                         │
│  • For EACH hypothesis, generate counter-evidence                                │
│  • Identify symptoms that DON'T fit the hypothesis                               │
│  • Propose alternative explanations                                              │
│  • Flag if insufficient evidence for confident diagnosis                         │
│  • Require minimum 3 supporting data points per hypothesis                       │
│                                                                                  │
│  Challenge Questions:                                                             │
│  "What evidence contradicts this hypothesis?"                                    │
│  "What alternative conditions present similarly?"                                │
│  "What information is missing to confirm this?"                                  │
│                                                                                  │
│                                        │                                        │
│                                        ▼                                        │
│  STEP 4: SYNTHESIZE (Final Assessment with Calibrated Confidence)               │
│  ─────────────────────────────────────────────────────────────────              │
│  Input: Challenged hypotheses, all gathered evidence                             │
│  Output: Final assessment with uncertainty quantification                        │
│                                                                                  │
│  Process:                                                                         │
│  • Integrate surviving hypotheses post-challenge                                 │
│  • Apply Sample Consistency Confidence Calibration                               │
│  • Generate severity assessment (PHQ-9/GAD-7 estimated)                          │
│  • Determine appropriate stepped care level                                      │
│  • Output structured DiagnosisResult                                             │
│                                                                                  │
│  Confidence Calibration:                                                          │
│  • Run N=3 independent LLM samples                                               │
│  • If agreement <60%: Flag as "Uncertain - needs more information"              │
│  • If agreement 60-80%: Report with wide confidence interval                     │
│  • If agreement >80%: Report with narrow confidence interval                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Files (Diagnosis Service):**

| File | Step | Responsibility |
|------|------|----------------|
| `symptom_analyzer.py` | Step 1 | Extract and structure symptoms with temporal markers |
| `differential_generator.py` | Step 2 | DSM-5-TR/HiTOP differential with confidence |
| `devil_advocate.py` | Step 3 | Challenge hypotheses, generate counter-evidence |
| `synthesizer.py` | Step 4 | Final assessment with calibrated confidence |
| `confidence_calibrator.py` | Step 4 | Sample consistency confidence scoring |
| `clinical_codes.py` | Cross-step | DSM-5-TR/ICD-11 code mapping |

### 9.4 Therapy Modality Implementation

Per `02-therapy-module/ARCHITECTURE.md`, implement these modalities:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          THERAPY MODALITIES                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CBT (Cognitive Behavioral Therapy) - 12-Session Protocol                        │
│  ─────────────────────────────────────────────────────────                       │
│  Sessions 1-2:  Psychoeducation, case formulation, goal setting                 │
│  Sessions 3-4:  Cognitive model introduction, thought monitoring                │
│  Sessions 5-6:  Cognitive restructuring, thought challenging                    │
│  Sessions 7-8:  Behavioral activation, activity scheduling                      │
│  Sessions 9-10: Advanced techniques (behavioral experiments, exposure)          │
│  Sessions 11-12: Relapse prevention, termination planning                       │
│                                                                                  │
│  Techniques:                                                                      │
│  • Thought records (situation → thought → emotion → evidence → balanced thought)│
│  • Behavioral experiments (test predictions)                                    │
│  • Activity scheduling (mastery/pleasure ratings)                               │
│  • Cognitive distortion identification (15 types)                               │
│  • Socratic questioning                                                          │
│                                                                                  │
│  DBT (Dialectical Behavior Therapy) - 4 Modules                                  │
│  ────────────────────────────────────────────────                                │
│  Module 1: Mindfulness                                                           │
│    • Wise mind concept                                                           │
│    • "What" skills: Observe, Describe, Participate                              │
│    • "How" skills: Non-judgmentally, One-mindfully, Effectively                 │
│                                                                                  │
│  Module 2: Distress Tolerance                                                    │
│    • TIPP: Temperature, Intense exercise, Paced breathing, Paired relaxation   │
│    • STOP: Stop, Take a step back, Observe, Proceed mindfully                   │
│    • Pros and Cons, IMPROVE the moment, Self-soothe with senses                │
│    • Radical acceptance                                                          │
│                                                                                  │
│  Module 3: Emotion Regulation                                                    │
│    • ABC PLEASE: Accumulate positive experiences, Build mastery,               │
│      Cope ahead, Physical health (PL), Exercise (E), Avoid substances (A),     │
│      Sleep (S), Eat balanced (E)                                                │
│    • Check the facts, Opposite action                                           │
│    • Problem solving                                                             │
│                                                                                  │
│  Module 4: Interpersonal Effectiveness                                           │
│    • DEAR MAN: Describe, Express, Assert, Reinforce, Mindful, Appear           │
│      confident, Negotiate                                                        │
│    • GIVE: Gentle, Interested, Validate, Easy manner                            │
│    • FAST: Fair, no Apologies, Stick to values, Truthful                        │
│                                                                                  │
│  ACT (Acceptance & Commitment Therapy) - Hexaflex Model                          │
│  ───────────────────────────────────────────────────────                         │
│  1. Cognitive Defusion: "I notice I'm having the thought that..."               │
│  2. Acceptance: Willingness to experience difficult emotions                     │
│  3. Present Moment: Mindful awareness of here and now                           │
│  4. Self-as-Context: Observer self vs. content of thoughts                      │
│  5. Values Clarification: What matters most to you?                             │
│  6. Committed Action: Values-based behavioral goals                             │
│                                                                                  │
│  MI (Motivational Interviewing) - OARS Skills                                    │
│  ─────────────────────────────────────────────                                   │
│  O: Open-ended questions ("What would you like to change?")                     │
│  A: Affirmations ("You've shown real courage in sharing that")                  │
│  R: Reflections (simple, complex, double-sided)                                 │
│  S: Summaries (collecting, linking, transitional)                               │
│                                                                                  │
│  Change Talk Elicitation:                                                         │
│  • DARN-CAT: Desire, Ability, Reason, Need, Commitment, Activation, Taking steps│
│  • Rolling with resistance (never argue)                                        │
│  • Developing discrepancy (values vs. behavior)                                 │
│                                                                                  │
│  MINDFULNESS SCRIPTS                                                             │
│  ───────────────────                                                             │
│  • 4-7-8 Breathing: Inhale 4s, hold 7s, exhale 8s                              │
│  • Body Scan: Head to toe progressive awareness                                 │
│  • 5-4-3-2-1 Grounding: 5 things see, 4 hear, 3 touch, 2 smell, 1 taste        │
│  • Loving-Kindness: May I be happy, may I be healthy, may I be at peace        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Files (Therapy Service):**

| File | Modality | Responsibility |
|------|----------|----------------|
| `cbt_protocol.py` | CBT | 12-session structured protocol, thought records |
| `dbt_modules.py` | DBT | 4 modules: Mindfulness, Distress, Emotion, Interpersonal |
| `act_hexaflex.py` | ACT | 6 processes: Defusion, Acceptance, Values, etc. |
| `mi_skills.py` | MI | OARS skills, change talk elicitation |
| `mindfulness_scripts.py` | Mindfulness | Guided scripts, breathing exercises |
| `technique_selector.py` | All | 4-stage selection algorithm |
| `stepped_care_router.py` | All | PHQ-9 severity → treatment intensity mapping |

### 9.5 4-Stage Technique Selection Algorithm

Per `02-therapy-module/ARCHITECTURE.md`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    4-STAGE TECHNIQUE SELECTION ALGORITHM                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 1: CLINICAL FILTER                                                        │
│  ────────────────────────                                                        │
│  • Remove techniques contraindicated for user's conditions                       │
│  • Remove techniques requiring prerequisites not yet met                         │
│  • Remove techniques inappropriate for severity level                            │
│  • Output: Clinically safe technique pool                                        │
│                                                                                  │
│  STAGE 2: PERSONALIZATION SCORING                                                │
│  ────────────────────────────────                                                │
│  For each technique in pool, calculate:                                          │
│                                                                                  │
│  personalization_score = Σ(Big_Five_trait × technique_affinity)                 │
│                                                                                  │
│  Technique Affinities (from personality research):                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Technique          │ O    │ C    │ E    │ A    │ N    │                    │ │
│  ├───────────────────────────────────────────────────────────────────────────┤ │
│  │ Cognitive Restr.   │ 0.6  │ 0.8  │ 0.3  │ 0.4  │ -0.2 │                    │ │
│  │ Behavioral Activ.  │ 0.4  │ 0.7  │ 0.8  │ 0.5  │ -0.3 │                    │ │
│  │ Mindfulness        │ 0.7  │ 0.3  │ 0.2  │ 0.6  │ 0.5  │                    │ │
│  │ Values Exploration │ 0.9  │ 0.4  │ 0.5  │ 0.7  │ 0.3  │                    │ │
│  │ TIPP Skills        │ 0.2  │ 0.5  │ 0.4  │ 0.3  │ 0.8  │                    │ │
│  │ Socratic Question. │ 0.8  │ 0.6  │ 0.4  │ 0.5  │ 0.1  │                    │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  STAGE 3: CONTEXT RANKING                                                        │
│  ────────────────────────                                                        │
│  Consider:                                                                        │
│  • Current emotional state (from Personality Service)                            │
│  • Session phase (Opening/Working/Closing)                                       │
│  • Time remaining in session                                                     │
│  • Recent technique usage (avoid repetition)                                    │
│  • Treatment plan phase (Foundation/Active/Consolidation/Maintenance)            │
│                                                                                  │
│  STAGE 4: FINAL SELECTION                                                        │
│  ────────────────────────                                                        │
│  final_score = clinical_weight × 0.4 +                                          │
│                personalization_score × 0.3 +                                     │
│                context_score × 0.2 +                                             │
│                effectiveness_history × 0.1                                       │
│                                                                                  │
│  Select top-scoring technique                                                    │
│  Log selection rationale for transparency                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.6 Personality Detection Pipeline (OCEAN Ensemble + MoEL)

Per `03-personality-module/ARCHITECTURE.md`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OCEAN ENSEMBLE DETECTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT: User text + Voice (optional) + Behavioral signals                        │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ PATH 1:         │  │ PATH 2:         │  │ PATH 3:         │                 │
│  │ Fine-tuned      │  │ Zero-shot LLM   │  │ LIWC Features   │                 │
│  │ RoBERTa Large   │  │ Analysis        │  │ Mapping         │                 │
│  │                 │  │                 │  │                 │                 │
│  │ Primary Model   │  │ Validation      │  │ 93 LIWC → OCEAN │                 │
│  │ R² = 0.24       │  │ r = 0.29-0.38   │  │ correlations    │                 │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                 │
│           │                    │                    │                           │
│           └────────────────────┼────────────────────┘                           │
│                                │                                                │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     WEIGHTED ENSEMBLE AGGREGATION                        │   │
│  │                                                                          │   │
│  │  OCEAN_final = w1 × RoBERTa + w2 × LLM + w3 × LIWC                      │   │
│  │                                                                          │   │
│  │  Weights: w1=0.5, w2=0.3, w3=0.2 (tuned on validation set)              │   │
│  │                                                                          │   │
│  │  Confidence Calculation:                                                 │   │
│  │  confidence = 1 - std(predictions) / mean(predictions)                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  OUTPUT: OceanScores {                                                          │
│    openness: 0.72 ± 0.08,                                                       │
│    conscientiousness: 0.45 ± 0.12,                                              │
│    extraversion: 0.38 ± 0.10,                                                   │
│    agreeableness: 0.81 ± 0.06,                                                  │
│    neuroticism: 0.56 ± 0.09,                                                    │
│    overall_confidence: 0.82                                                     │
│  }                                                                               │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                  │
│                    MoEL (MIXTURE OF EMPATHETIC LISTENERS)                       │
│  ─────────────────────────────────────────────────────────                      │
│                                                                                  │
│  User Input → Transformer Encoder → Emotion Tracker → 32-Emotion Softmax        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    32 SPECIALIZED LISTENER DECODERS                      │   │
│  │                                                                          │   │
│  │  😢 Sadness │ 😰 Anxiety │ 😠 Anger │ 😊 Joy │ 😨 Fear │ 😔 Guilt │ ...  │   │
│  │                                                                          │   │
│  │  Each listener trained on emotion-specific empathic responses            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    META-LISTENER (SOFT COMBINATION)                      │   │
│  │                                                                          │   │
│  │  response = Σ(emotion_weight_i × listener_i_response)                   │   │
│  │                                                                          │   │
│  │  Example: 0.6 × sadness_response + 0.3 × anxiety_response + 0.1 × ...   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│  THREE-COMPONENT EMPATHY OUTPUT:                                                 │
│  • Cognitive: "It sounds like you're feeling..."                                │
│  • Affective: "That must be really difficult..."                                │
│  • Compassionate: "What would help right now?"                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Files (Personality Service):**

| File | Component | Responsibility |
|------|-----------|----------------|
| `roberta_detector.py` | Path 1 | Fine-tuned RoBERTa Big Five classifier |
| `llm_detector.py` | Path 2 | Zero-shot LLM personality analysis |
| `liwc_extractor.py` | Path 3 | LIWC feature extraction and mapping |
| `ensemble_aggregator.py` | Fusion | Weighted combination with confidence |
| `moel_empathy.py` | Empathy | 32-emotion listener soft combination |
| `style_mapper.py` | Adaptation | OCEAN → Style parameters (warmth, structure, etc.) |

---

## 10. Event Schemas & API Contracts

### 10.1 Kafka Event Schemas

```python
# Event Base Schema
class BaseEvent(BaseModel):
    event_id: UUID
    event_type: str
    timestamp: datetime
    user_id: UUID
    session_id: Optional[UUID]
    correlation_id: UUID

# Safety Events
class CrisisDetectedEvent(BaseEvent):
    event_type: Literal["safety.crisis.detected"] = "safety.crisis.detected"
    crisis_level: Literal["CRITICAL", "HIGH", "ELEVATED"]
    trigger_text: str
    detection_layer: int  # 1-4
    confidence: float
    escalation_action: str

class SafetyAssessmentEvent(BaseEvent):
    event_type: Literal["safety.assessment.completed"] = "safety.assessment.completed"
    risk_level: str
    risk_factors: List[RiskFactor]
    protective_factors: List[str]
    recommended_action: str

# Memory Events
class MemoryStoredEvent(BaseEvent):
    event_type: Literal["memory.stored"] = "memory.stored"
    memory_tier: Literal["INPUT", "WORKING", "SESSION", "EPISODIC", "SEMANTIC"]
    content_type: str
    retention_category: Literal["PERMANENT", "LONG_TERM", "MEDIUM_TERM", "SHORT_TERM"]

class MemoryConsolidatedEvent(BaseEvent):
    event_type: Literal["memory.consolidated"] = "memory.consolidated"
    session_id: UUID
    summary_id: UUID
    facts_extracted: int
    embeddings_created: int

# Diagnosis Events
class DiagnosisCompletedEvent(BaseEvent):
    event_type: Literal["diagnosis.completed"] = "diagnosis.completed"
    primary_hypothesis: ClinicalHypothesis
    differential: List[ClinicalHypothesis]
    confidence_level: str
    severity_assessment: SeverityLevel
    stepped_care_level: int

# Therapy Events
class SessionStartedEvent(BaseEvent):
    event_type: Literal["therapy.session.started"] = "therapy.session.started"
    session_number: int
    treatment_plan_id: UUID
    planned_focus: List[str]

class InterventionDeliveredEvent(BaseEvent):
    event_type: Literal["therapy.intervention.delivered"] = "therapy.intervention.delivered"
    technique: str
    modality: Literal["CBT", "DBT", "ACT", "MI", "MINDFULNESS"]
    selection_rationale: Dict[str, float]

# Personality Events
class PersonalityAssessedEvent(BaseEvent):
    event_type: Literal["personality.assessed"] = "personality.assessed"
    ocean_scores: OceanScores
    assessment_source: Literal["ROBERTA", "LLM", "LIWC", "ENSEMBLE"]
    confidence: float

class StyleGeneratedEvent(BaseEvent):
    event_type: Literal["personality.style.generated"] = "personality.style.generated"
    style_params: StyleParameters
    target_module: str
```

### 10.2 Service API Contracts

```python
# Memory Service API
class IMemoryService(Protocol):
    async def store(self, user_id: UUID, data: MemoryData, tier: MemoryTier) -> MemoryRecord
    async def retrieve(self, user_id: UUID, query: str, options: RetrievalOptions) -> List[MemoryRecord]
    async def get_context(self, user_id: UUID, token_budget: int) -> AssembledContext
    async def consolidate_session(self, session_id: UUID) -> ConsolidationResult

# Safety Service API
class ISafetyService(Protocol):
    async def check_input(self, user_id: UUID, message: str) -> SafetyCheckResult
    async def check_technique(self, user_id: UUID, technique: str) -> ContraindicationResult
    async def filter_output(self, user_id: UUID, response: str) -> FilteredResponse
    async def get_crisis_protocol(self, crisis_level: CrisisLevel) -> CrisisProtocol

# Diagnosis Service API
class IDiagnosisService(Protocol):
    async def assess(self, user_id: UUID, session_context: SessionContext) -> DiagnosisResult
    async def get_differential(self, user_id: UUID) -> List[ClinicalHypothesis]
    async def get_severity(self, user_id: UUID) -> SeverityAssessment

# Therapy Service API
class ITherapyService(Protocol):
    async def start_session(self, user_id: UUID, plan_id: UUID) -> SessionState
    async def process_message(self, session_id: UUID, message: str) -> TherapyResponse
    async def select_technique(self, session_id: UUID, context: TechniqueContext) -> SelectedTechnique
    async def end_session(self, session_id: UUID) -> SessionSummary

# Personality Service API
class IPersonalityService(Protocol):
    async def detect(self, user_id: UUID, text: str, audio: Optional[bytes]) -> PersonalityAssessment
    async def get_profile(self, user_id: UUID) -> PersonalityProfile
    async def get_style(self, user_id: UUID) -> StyleParameters
    async def generate_empathy(self, user_id: UUID, context: EmotionContext) -> EmpathyComponents
```

---

## 11. LangGraph Agent Priority Hierarchy

Per `00-system-integration/ARCHITECTURE.md`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT PRIORITY HIERARCHY                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PRIORITY 0: SAFETY OVERRIDE (Highest - Can interrupt ANY agent)                 │
│  ────────────────────────────────────────────────────────────────               │
│  • Monitors ALL messages (input and output)                                      │
│  • Can halt processing at any point                                              │
│  • Always runs in parallel with other agents                                     │
│  • Has escalation authority to external systems                                  │
│                                                                                  │
│  PRIORITY 1: SAFETY AGENT                                                        │
│  ────────────────────────                                                        │
│  • Runs BEFORE other agents process                                              │
│  • 4-layer safety checks                                                         │
│  • Crisis detection and escalation                                               │
│  • Can block agent activation                                                    │
│                                                                                  │
│  PRIORITY 2: ORCHESTRATOR/SUPERVISOR                                             │
│  ────────────────────────────────────                                            │
│  • Routes requests to appropriate agents                                         │
│  • Coordinates multi-agent workflows                                             │
│  • Manages shared state                                                          │
│  • Quality control on agent outputs                                              │
│                                                                                  │
│  PRIORITY 3: CLINICAL AGENTS (Parallel execution)                                │
│  ────────────────────────────────────────────────                                │
│  • Diagnosis Agent: AMIE 4-step reasoning                                        │
│  • Therapy Agent: Technique selection and delivery                               │
│  • Assessment Agent: Standardized measures                                       │
│                                                                                  │
│  PRIORITY 4: SUPPORT AGENTS (Parallel execution)                                 │
│  ───────────────────────────────────────────────                                 │
│  • Personality Agent: Big Five detection                                         │
│  • Emotion Agent: Real-time emotion tracking                                     │
│  • Chat Agent: General conversation handling                                     │
│                                                                                  │
│  ══════════════════════════════════════════════════════════════════════════════ │
│                                                                                  │
│  LANGGRAPH STATE SCHEMA:                                                         │
│                                                                                  │
│  class SolaceState(TypedDict):                                                   │
│      # Identity                                                                  │
│      user_id: UUID                                                               │
│      session_id: UUID                                                            │
│      conversation_id: UUID                                                       │
│                                                                                  │
│      # Current Input                                                             │
│      current_message: str                                                        │
│      message_timestamp: datetime                                                 │
│                                                                                  │
│      # Context (from Memory Service)                                             │
│      assembled_context: AssembledContext                                         │
│      user_profile: UserProfile                                                   │
│      personality_profile: PersonalityProfile                                     │
│      treatment_context: TreatmentContext                                         │
│                                                                                  │
│      # Safety State                                                              │
│      safety_flags: List[SafetyFlag]                                              │
│      crisis_level: Optional[CrisisLevel]                                         │
│      safety_override_active: bool                                                │
│                                                                                  │
│      # Agent Outputs                                                             │
│      diagnosis_output: Optional[DiagnosisResult]                                 │
│      therapy_output: Optional[TherapyResponse]                                   │
│      personality_output: Optional[StyleParameters]                               │
│      emotion_output: Optional[EmotionState]                                      │
│                                                                                  │
│      # Response Assembly                                                         │
│      aggregated_response: Optional[str]                                          │
│      styled_response: Optional[str]                                              │
│      final_response: Optional[str]                                               │
│                                                                                  │
│      # Routing                                                                   │
│      active_agents: List[str]                                                    │
│      next_agent: Optional[str]                                                   │
│      routing_reason: str                                                         │
│                                                                                  │
│  ══════════════════════════════════════════════════════════════════════════════ │
│                                                                                  │
│  GRAPH STRUCTURE (Conditional Edges):                                            │
│                                                                                  │
│  START → safety_pre_check → {                                                    │
│      "crisis": → crisis_handler → END                                           │
│      "safe": → supervisor → {                                                    │
│          "clinical": → [diagnosis, therapy] (parallel) → aggregator             │
│          "support": → [personality, chat] (parallel) → aggregator               │
│          "mixed": → [diagnosis, therapy, personality] (parallel) → aggregator   │
│      }                                                                           │
│  }                                                                               │
│  aggregator → style_applicator → safety_post_check → {                          │
│      "pass": → END                                                               │
│      "filter": → safety_filter → END                                            │
│  }                                                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Technology Stack: Latest Versions & Patterns (2025)

> **Last Updated**: January 2025
> **Source**: Context7 Documentation API

### 12.1 Core Framework Versions

| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| **Python** | 3.12+ | Runtime | Pattern matching, performance improvements |
| **FastAPI** | 0.128.0+ | Web Framework | Lifespan events, dependency injection |
| **Pydantic** | 2.10+ | Validation | `@field_validator`, `model_validator`, `ConfigDict` |
| **LangGraph** | 1.0.3+ | Agent Orchestration | StateGraph, checkpointing, multi-agent |
| **LangChain** | 0.3+ | LLM Framework | LCEL, RAG patterns, tool use |
| **SQLAlchemy** | 2.1+ | ORM | `Mapped`, `mapped_column`, async sessions |
| **Weaviate** | 4.10+ | Vector DB | Hybrid search, named vectors, collections |
| **Redis** | 5.2+ (redis-py 6.4+) | Cache/Streams | Async client, cluster, pub/sub |
| **aiokafka** | 0.12+ | Event Streaming | Async producer/consumer, manual commit |
| **HTTPX** | 0.28+ | HTTP Client | Async, connection pooling, timeouts |
| **Structlog** | 25.1+ | Logging | JSON, contextvars, FastAPI integration |
| **Prometheus** | 0.22+ (client) | Metrics | ASGI middleware, histograms |
| **OpenTelemetry** | 1.29+ | Tracing | OTLP exporters, auto-instrumentation |

### 12.2 FastAPI Lifespan Pattern (Required)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    app.state.redis = await create_redis_pool()
    app.state.kafka_producer = await create_kafka_producer()
    app.state.weaviate = await create_weaviate_client()
    yield
    # Shutdown: Cleanup resources
    await app.state.redis.close()
    await app.state.kafka_producer.stop()
    await app.state.weaviate.close()

app = FastAPI(lifespan=lifespan)
```

### 12.3 LangGraph StateGraph Pattern (Required)

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

class SolaceState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    safety_status: str
    diagnosis_output: dict | None
    therapy_output: dict | None
    personality_profile: dict | None
    memory_context: list
    active_agents: list[str]
    next_agent: str | None

# Build graph with checkpointing
builder = StateGraph(SolaceState)
builder.add_node("safety_pre_check", safety_pre_check_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("diagnosis", diagnosis_node)
builder.add_node("therapy", therapy_node)
builder.add_conditional_edges("safety_pre_check", route_safety)
builder.add_edge(START, "safety_pre_check")

# Compile with PostgreSQL checkpointer
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### 12.4 Pydantic V2 Patterns (Required)

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Self

class DiagnosisRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid"
    )

    user_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(..., pattern=r"^[a-f0-9-]{36}$")

    @field_validator("message", mode="before")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return v.strip()[:10000]

    @model_validator(mode="after")
    def validate_session(self) -> Self:
        if not self.session_id:
            raise ValueError("session_id is required")
        return self
```

### 12.5 SQLAlchemy 2.1 Declarative Pattern (Required)

```python
from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime
from typing import Optional

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(insert_default=func.now())

    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    diagnosis_state: Mapped[Optional[str]]

    user: Mapped["User"] = relationship(back_populates="sessions")
```

### 12.6 Weaviate Hybrid Search Pattern (Required)

```python
import weaviate
from weaviate.classes.query import HybridFusion

async def hybrid_search(
    client: weaviate.WeaviateAsyncClient,
    query: str,
    collection_name: str = "TherapyMemory",
    alpha: float = 0.5,  # 0=BM25, 1=vector
    limit: int = 10
) -> list[dict]:
    collection = client.collections.get(collection_name)

    response = await collection.query.hybrid(
        query=query,
        alpha=alpha,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=limit,
        return_metadata=["score", "distance"]
    )

    return [
        {"content": obj.properties, "score": obj.metadata.score}
        for obj in response.objects
    ]
```

### 12.7 Redis Async Pattern (Required)

```python
import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

async def create_redis_pool() -> aioredis.Redis:
    pool = ConnectionPool.from_url(
        "redis://localhost:6379",
        max_connections=50,
        decode_responses=True
    )
    return aioredis.Redis(connection_pool=pool)

async def cache_with_ttl(
    redis: aioredis.Redis,
    key: str,
    value: str,
    ttl_seconds: int = 3600
) -> None:
    async with redis.pipeline(transaction=True) as pipe:
        await pipe.set(key, value)
        await pipe.expire(key, ttl_seconds)
        await pipe.execute()
```

### 12.8 aiokafka Producer/Consumer Pattern (Required)

```python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json

async def create_kafka_producer() -> AIOKafkaProducer:
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        compression_type="gzip",
        acks="all"
    )
    await producer.start()
    return producer

async def create_kafka_consumer(
    topic: str,
    group_id: str
) -> AIOKafkaConsumer:
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False  # Manual commit for reliability
    )
    await consumer.start()
    return consumer
```

### 12.9 Prometheus Metrics Pattern (Required)

```python
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Define metrics
http_requests = Counter(
    "solace_http_requests_total",
    "Total HTTP requests",
    ["service", "method", "endpoint", "status"]
)

http_duration = Histogram(
    "solace_http_request_duration_seconds",
    "HTTP request duration",
    ["service", "method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str):
        super().__init__(app)
        self.service_name = service_name

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        http_requests.labels(
            service=self.service_name,
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        http_duration.labels(
            service=self.service_name,
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

        return response

# Mount metrics endpoint
app.mount("/metrics", make_asgi_app())
```

### 12.10 Structlog JSON Configuration (Required)

```python
import structlog
import logging
import orjson

def configure_logging(service_name: str) -> None:
    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            # Add service context
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(serializer=orjson.dumps),
        ],
        logger_factory=structlog.BytesLoggerFactory(),
    )

# Usage
log = structlog.get_logger()
log.info("session_started", user_id="123", session_id="abc-def")
```

### 12.11 HTTPX Async Client Pattern (Required)

```python
import httpx

async def create_http_client() -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )

    timeout = httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0
    )

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=True
    )
```

### 12.12 OpenTelemetry Tracing Pattern (Required)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

def configure_tracing(service_name: str) -> None:
    resource = Resource.create(attributes={
        SERVICE_NAME: service_name
    })

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://jaeger:4317")
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

# Usage
tracer = trace.get_tracer("solace.diagnosis")
with tracer.start_as_current_span("process_message") as span:
    span.set_attribute("user_id", user_id)
    span.set_attribute("session_id", session_id)
    # ... processing logic
```

### 12.13 Claude API Tool Use Pattern (Required)

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_memory_context",
        "description": "Retrieve relevant memories for the current conversation",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
]

async def chat_with_tools(messages: list, tools: list) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b.type == "tool_use")
        tool_result = await execute_tool(tool_use.name, tool_use.input)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(tool_result)
            }]
        })

        return await chat_with_tools(messages, tools)

    return next(b.text for b in response.content if hasattr(b, "text"))
```

### 12.14 Official Documentation Links

| Package | Documentation | PyPI | GitHub |
|---------|--------------|------|--------|
| **Python 3.12** | [docs.python.org](https://docs.python.org/3.12/) | - | [python/cpython](https://github.com/python/cpython) |
| **FastAPI** | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) | [pypi.org/project/fastapi](https://pypi.org/project/fastapi/) | [fastapi/fastapi](https://github.com/fastapi/fastapi) |
| **Pydantic** | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) | [pypi.org/project/pydantic](https://pypi.org/project/pydantic/) | [pydantic/pydantic](https://github.com/pydantic/pydantic) |
| **LangGraph** | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | [pypi.org/project/langgraph](https://pypi.org/project/langgraph/) | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |
| **LangChain** | [python.langchain.com](https://python.langchain.com/docs/) | [pypi.org/project/langchain](https://pypi.org/project/langchain/) | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) |
| **Anthropic SDK** | [docs.anthropic.com](https://docs.anthropic.com/en/api/) | [pypi.org/project/anthropic](https://pypi.org/project/anthropic/) | [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) |
| **SQLAlchemy** | [docs.sqlalchemy.org](https://docs.sqlalchemy.org/en/21/) | [pypi.org/project/sqlalchemy](https://pypi.org/project/sqlalchemy/) | [sqlalchemy/sqlalchemy](https://github.com/sqlalchemy/sqlalchemy) |
| **Alembic** | [alembic.sqlalchemy.org](https://alembic.sqlalchemy.org/en/latest/) | [pypi.org/project/alembic](https://pypi.org/project/alembic/) | [sqlalchemy/alembic](https://github.com/sqlalchemy/alembic) |
| **asyncpg** | [magicstack.github.io/asyncpg](https://magicstack.github.io/asyncpg/current/) | [pypi.org/project/asyncpg](https://pypi.org/project/asyncpg/) | [MagicStack/asyncpg](https://github.com/MagicStack/asyncpg) |
| **Weaviate** | [weaviate.io/developers](https://weaviate.io/developers/weaviate) | [pypi.org/project/weaviate-client](https://pypi.org/project/weaviate-client/) | [weaviate/weaviate-python-client](https://github.com/weaviate/weaviate-python-client) |
| **Redis (redis-py)** | [redis-py.readthedocs.io](https://redis-py.readthedocs.io/en/stable/) | [pypi.org/project/redis](https://pypi.org/project/redis/) | [redis/redis-py](https://github.com/redis/redis-py) |
| **aiokafka** | [aiokafka.readthedocs.io](https://aiokafka.readthedocs.io/en/stable/) | [pypi.org/project/aiokafka](https://pypi.org/project/aiokafka/) | [aio-libs/aiokafka](https://github.com/aio-libs/aiokafka) |
| **HTTPX** | [www.python-httpx.org](https://www.python-httpx.org/) | [pypi.org/project/httpx](https://pypi.org/project/httpx/) | [encode/httpx](https://github.com/encode/httpx) |
| **Structlog** | [www.structlog.org](https://www.structlog.org/en/stable/) | [pypi.org/project/structlog](https://pypi.org/project/structlog/) | [hynek/structlog](https://github.com/hynek/structlog) |
| **Prometheus Client** | [prometheus.github.io/client_python](https://prometheus.github.io/client_python/) | [pypi.org/project/prometheus-client](https://pypi.org/project/prometheus-client/) | [prometheus/client_python](https://github.com/prometheus/client_python) |
| **OpenTelemetry** | [opentelemetry.io/docs/languages/python](https://opentelemetry.io/docs/languages/python/) | [pypi.org/project/opentelemetry-api](https://pypi.org/project/opentelemetry-api/) | [open-telemetry/opentelemetry-python](https://github.com/open-telemetry/opentelemetry-python) |
| **Uvicorn** | [www.uvicorn.org](https://www.uvicorn.org/) | [pypi.org/project/uvicorn](https://pypi.org/project/uvicorn/) | [encode/uvicorn](https://github.com/encode/uvicorn) |
| **Tenacity** | [tenacity.readthedocs.io](https://tenacity.readthedocs.io/en/latest/) | [pypi.org/project/tenacity](https://pypi.org/project/tenacity/) | [jd/tenacity](https://github.com/jd/tenacity) |
| **orjson** | [github.com/ijl/orjson](https://github.com/ijl/orjson#readme) | [pypi.org/project/orjson](https://pypi.org/project/orjson/) | [ijl/orjson](https://github.com/ijl/orjson) |
| **pytest** | [docs.pytest.org](https://docs.pytest.org/en/stable/) | [pypi.org/project/pytest](https://pypi.org/project/pytest/) | [pytest-dev/pytest](https://github.com/pytest-dev/pytest) |
| **Ruff** | [docs.astral.sh/ruff](https://docs.astral.sh/ruff/) | [pypi.org/project/ruff](https://pypi.org/project/ruff/) | [astral-sh/ruff](https://github.com/astral-sh/ruff) |
| **mypy** | [mypy.readthedocs.io](https://mypy.readthedocs.io/en/stable/) | [pypi.org/project/mypy](https://pypi.org/project/mypy/) | [python/mypy](https://github.com/python/mypy) |

### 12.15 Infrastructure Documentation

| Technology | Documentation | Quick Start |
|------------|--------------|-------------|
| **Docker** | [docs.docker.com](https://docs.docker.com/) | [Get Docker](https://docs.docker.com/get-docker/) |
| **Kubernetes** | [kubernetes.io/docs](https://kubernetes.io/docs/home/) | [Minikube](https://minikube.sigs.k8s.io/docs/start/) |
| **Istio** | [istio.io/docs](https://istio.io/latest/docs/) | [Getting Started](https://istio.io/latest/docs/setup/getting-started/) |
| **Kafka** | [kafka.apache.org/documentation](https://kafka.apache.org/documentation/) | [Quickstart](https://kafka.apache.org/quickstart) |
| **Redis** | [redis.io/docs](https://redis.io/docs/) | [Get Started](https://redis.io/docs/getting-started/) |
| **PostgreSQL** | [postgresql.org/docs](https://www.postgresql.org/docs/current/) | [Tutorial](https://www.postgresql.org/docs/current/tutorial.html) |
| **Weaviate Server** | [weaviate.io/developers](https://weaviate.io/developers/weaviate) | [Docker Compose](https://weaviate.io/developers/weaviate/installation/docker-compose) |
| **Prometheus** | [prometheus.io/docs](https://prometheus.io/docs/introduction/overview/) | [Getting Started](https://prometheus.io/docs/prometheus/latest/getting_started/) |
| **Grafana** | [grafana.com/docs](https://grafana.com/docs/grafana/latest/) | [Getting Started](https://grafana.com/docs/grafana/latest/getting-started/) |
| **Jaeger** | [jaegertracing.io/docs](https://www.jaegertracing.io/docs/) | [Getting Started](https://www.jaegertracing.io/docs/getting-started/) |
| **Kong** | [docs.konghq.com](https://docs.konghq.com/) | [Get Started](https://docs.konghq.com/gateway/latest/get-started/) |
| **ELK Stack** | [elastic.co/guide](https://www.elastic.co/guide/index.html) | [Quick Start](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html) |

### 12.16 Package Requirements (pyproject.toml)

```toml
[project]
name = "solace-ai"
version = "1.0.0"
requires-python = ">=3.12"

dependencies = [
    # Web Framework
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",

    # AI/ML
    "langgraph>=1.0.3",
    "langchain>=0.3.14",
    "langchain-anthropic>=0.3.3",
    "anthropic>=0.42.0",

    # Database
    "sqlalchemy[asyncio]>=2.1.0",
    "asyncpg>=0.30.0",
    "alembic>=1.14.0",

    # Vector Database
    "weaviate-client>=4.10.0",

    # Cache & Messaging
    "redis>=5.2.0",
    "aiokafka>=0.12.0",

    # HTTP Client
    "httpx[http2]>=0.28.0",

    # Observability
    "structlog>=25.1.0",
    "orjson>=3.10.0",
    "prometheus-client>=0.22.0",
    "opentelemetry-api>=1.29.0",
    "opentelemetry-sdk>=1.29.0",
    "opentelemetry-exporter-otlp>=1.29.0",
    "opentelemetry-instrumentation-fastapi>=0.50b0",

    # Utilities
    "tenacity>=9.0.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "mypy>=1.13.0",
    "ruff>=0.8.0",
]
```

---

## Summary Statistics (Updated)

| Metric | Value |
|--------|-------|
| **Total Phases** | 10 |
| **Total Batches** | 36 |
| **Total Files** | 180+ |
| **Shared Libraries** | 6 |
| **Microservices** | 9 |
| **Infrastructure Components** | 3 |
| **Max LOC per File** | 400 |
| **Estimated Total LOC** | ~54,000 |
| **Architecture Gaps Addressed** | 42 |

### Alignment Verification

| Architecture Document | Alignment Status |
|----------------------|------------------|
| `00-system-integration/ARCHITECTURE.md` | ✅ Full alignment |
| `01-diagnosis-module/ARCHITECTURE.md` | ✅ Full alignment |
| `02-therapy-module/ARCHITECTURE.md` | ✅ Full alignment |
| `03-personality-module/ARCHITECTURE.md` | ✅ Full alignment |
| `04-memory-module/ARCHITECTURE.md` | ✅ Full alignment |

---

*Document Version: 3.0*
*Created: December 31, 2025*
*Updated: January 1, 2026*
*Status: Implementation Blueprint (Reviewed, Enhanced & Version-Verified)*
*Architecture: Microservices + Event-Driven*
*Alignment: Verified against all system-design/*.md documents*
*Technology Stack: All package versions verified via Context7 (January 2025)*
