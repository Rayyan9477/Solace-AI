# Solace-AI System Design Documentation

> **Version**: 2.0  
> **Date**: December 30, 2025  
> **Status**: Technical Blueprint

---

## Overview

This folder contains the complete system architecture documentation for Solace-AI, a mental health AI platform. Each module has dedicated architecture and diagram documentation.

---

## Module Index

| # | Module | Architecture | Diagrams | Status |
|---|--------|-------------|----------|--------|
| 00 | **System Integration** | [ARCHITECTURE.md](00-system-integration/ARCHITECTURE.md) | [DIAGRAMS.md](00-system-integration/DIAGRAMS.md) | ✅ Complete |
| 01 | **Diagnosis & Insight** | [ARCHITECTURE.md](01-diagnosis-module/ARCHITECTURE.md) | [DIAGRAMS.md](01-diagnosis-module/DIAGRAMS.md) | ✅ Complete |
| 02 | **Therapy** | [ARCHITECTURE.md](02-therapy-module/ARCHITECTURE.md) | [DIAGRAMS.md](02-therapy-module/DIAGRAMS.md) | ✅ Complete |
| 03 | **Personality Detection** | [ARCHITECTURE.md](03-personality-module/ARCHITECTURE.md) | [DIAGRAMS.md](03-personality-module/DIAGRAMS.md) | ✅ Complete |
| 04 | **Memory & Context** | [ARCHITECTURE.md](04-memory-module/ARCHITECTURE.md) | [DIAGRAMS.md](04-memory-module/DIAGRAMS.md) | ✅ Complete |

---

## Architecture Highlights

### 🌟 System Integration (00)
- **Pattern**: Multi-Agent Orchestration (LangGraph)
- **Safety**: Multi-layer always-active safety system
- **Communication**: Event-Driven + Sync API
- **Compliance**: HIPAA, SOC2, Zero Trust
- **Deployment**: Kubernetes with HA

### 🔍 Diagnosis Module (01)
- **Pattern**: Multi-Agent System (AMIE-inspired)
- **Agents**: Dialogue, Insight, Advocate, Reviewer, Safety
- **Reasoning**: 4-step Chain-of-Reasoning (Analyze → Hypothesize → Challenge → Synthesize)
- **Frameworks**: DSM-5-TR + HiTOP hybrid
- **Safety**: 3-layer crisis detection (<10ms, <100ms, <500ms)
- **Anti-Sycophancy**: Devil's Advocate agent

### 💆 Therapy Module (02)
- **Pattern**: Hybrid (Rules + LLM)
- **Framework**: 5-Component (Units, Decision Maker, Narrator, Supporter, Guardian)
- **Techniques**: CBT, DBT, ACT, MI, Mindfulness
- **Planning**: Stepped Care Model (4 levels)
- **Safety**: 4-layer guardrails with contraindication checking
- **Outcomes**: Measurement-Based Care (PHQ-9, GAD-7, ORS, SRS)

### 🎭 Personality Module (03)
- **Model**: Big Five (OCEAN) Continuous Scoring
- **Detection**: Ensemble (Fine-tuned RoBERTa + LLM)
- **Multimodal**: Late Fusion (Text + Voice + Behavior)
- **Empathy**: MoEL (Mixture of Empathetic Listeners)
- **Adaptation**: Personality-Aware Prompting (r>0.85)
- **Cultural**: Culture-aware response adaptation

### 🧠 Memory Module (04)
- **Hierarchy**: 5-Tier Cognitive Model (Working → Semantic)
- **Primary Store**: Temporal Knowledge Graph (Zep)
- **Vector DB**: Weaviate Hybrid Search (BM25 + Semantic)
- **Retrieval**: Agentic Corrective RAG
- **Decay**: Ebbinghaus with Safety Override
- **Consolidation**: Event-Driven Pipeline

---

## Document Structure

Each module follows this structure:

```
XX-module-name/
├── ARCHITECTURE.md    # Complete technical blueprint
│   ├── Philosophy & Principles
│   ├── High-Level Architecture
│   ├── Component Details
│   ├── Data Flow
│   ├── API Contracts
│   ├── Event Architecture
│   └── Deployment View
│
└── DIAGRAMS.md        # Visual reference (Mermaid)
    ├── Quick Reference Table
    ├── System Overview
    ├── Component Diagrams
    ├── Flow Diagrams
    └── Integration Maps
```

---

## Cross-Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SOLACE-AI MODULE DEPENDENCIES                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐   │
│  │ Diagnosis│────▶│  Therapy │────▶│ Response │────▶│   User   │   │
│  └────┬─────┘     └────┬─────┘     └────┬─────┘     └──────────┘   │
│       │                │                │                           │
│       ▼                ▼                ▼                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                      MEMORY MODULE                        │      │
│  │                 (Shared Context Layer)                    │      │
│  └──────────────────────────────────────────────────────────┘      │
│       │                │                │                           │
│       ▼                ▼                ▼                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                   PERSONALITY MODULE                      │      │
│  │                  (Personalization)                        │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ════════════════════════════════════════════════════════════       │
│  │                    SAFETY MODULE                          │      │
│  │              (Always Active, Cross-Cutting)               │      │
│  ════════════════════════════════════════════════════════════       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Decisions

| Decision | Pattern | Rationale |
|----------|---------|-----------|
| **Agent Architecture** | Multi-Agent with Orchestrator | Specialized processing, inspired by AMIE |
| **Content Generation** | Hybrid (Rules + LLM) | Clinical fidelity + conversational warmth |
| **Safety** | Multi-Layer Progressive | Balances speed with accuracy |
| **Communication** | Event-Driven + Sync API | Loose coupling + real-time capability |
| **Memory** | Multi-Tier Hierarchy | Speed + persistence balance |
| **Clinical Frameworks** | DSM-5-TR + HiTOP | Categorical + dimensional assessment |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React, React Native | Web & Mobile apps |
| **API Gateway** | Kong/Istio | Routing, auth, rate limiting |
| **Orchestration** | LangGraph | Multi-agent coordination |
| **Services** | Python/FastAPI | Backend services |
| **LLM** | Anthropic/OpenAI | AI inference |
| **Vector Store** | Weaviate | Semantic memory & hybrid search |
| **Database** | PostgreSQL | Structured data persistence |
| **Cache** | Redis Cluster | Session state & fast access |
| **Events** | Kafka | Event bus & audit trail |
| **Container** | Kubernetes | Orchestration & scaling |
| **Service Mesh** | Istio | mTLS, traffic management |
| **Monitoring** | Prometheus/Grafana | Metrics & alerting |

---

## Module Dependency Matrix

| Module | Depends On | Provides To |
|--------|------------|-------------|
| **Orchestrator** | All modules | All modules |
| **Diagnosis** | Memory, Safety | Therapy, Orchestrator |
| **Therapy** | Memory, Diagnosis, Personality, Safety | Orchestrator |
| **Personality** | Memory | Therapy, Response |
| **Memory** | Data stores | All modules |
| **Safety** | Memory | All modules (override) |

---

## References

- Google DeepMind AMIE Architecture
- Woebot (FDA Breakthrough Device)
- Wysa (NHS/CE Mark Certified)
- DSM-5-TR & ICD-11 Clinical Guidelines
- HiTOP Dimensional Model
- APA, NICE Clinical Guidelines
- MemGPT OS-like Memory Management
- Zep Temporal Knowledge Graph

---

*Last Updated: December 30, 2025*
