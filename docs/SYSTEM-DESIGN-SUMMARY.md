# Solace-AI System Design Summary

> Synthesized from 22 design documents (~20,000+ lines) + 3 architecture diagrams.
> Purpose: Unified reference for comparing design intent vs. codebase implementation.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Service Architecture](#2-service-architecture)
3. [Orchestration (LangGraph)](#3-orchestration-langgraph)
4. [Safety Architecture](#4-safety-architecture)
5. [Diagnosis Module](#5-diagnosis-module)
6. [Therapy Module](#6-therapy-module)
7. [Personality Module](#7-personality-module)
8. [Memory Module](#8-memory-module)
9. [Event-Driven Architecture](#9-event-driven-architecture)
10. [Infrastructure & Deployment](#10-infrastructure--deployment)
11. [API Contracts](#11-api-contracts)
12. [Cross-Cutting Concerns](#12-cross-cutting-concerns)
13. [Performance Targets](#13-performance-targets)
14. [Design Inconsistencies](#14-design-inconsistencies-found)

---

## 1. System Overview

**What**: AI-powered mental health companion backend providing evidence-based therapeutic interventions through multi-agent orchestration.

**Core Principles**: Safety First, Clinical Accuracy, Modularity, Honesty (calibrated confidence), Continuity (5-tier memory), Privacy (HIPAA), Loose Coupling (events), Observable First.

**Inspirations**: Google AMIE (multi-agent reasoning), Woebot/Wysa (hybrid AI + FDA standards), MemGPT (memory hierarchy), Zep (temporal knowledge graph).

**Tech Stack**:
- Language: Python 3.12+ / FastAPI 0.115+ / Uvicorn 0.32+
- Orchestration: LangGraph 0.2+ / LangChain 0.3+
- Validation: Pydantic 2.10+
- Databases: PostgreSQL 16+, Redis 7.4+, Weaviate (or ChromaDB — see §14)
- Messaging: Kafka with Schema Registry
- LLM: Multi-provider via abstract interface (Anthropic, OpenAI, Google, Deepseek, Groq, Ollama)
- Infrastructure: Docker, Kubernetes 1.27+, Istio, Kong, Prometheus/Grafana/Jaeger
- Security: HashiCorp Vault, mTLS, AES-256-GCM, RS256 JWT

---

## 2. Service Architecture

### 10 Microservices (Design Spec)

| # | Service | Layer | Responsibility | Protocol |
|---|---------|-------|---------------|----------|
| 1 | **Orchestrator** | L1 | LangGraph multi-agent coordination, WebSocket streaming | gRPC/HTTP |
| 2 | **Safety** | L2 | 4-layer crisis detection, escalation, content filtering | gRPC/HTTP |
| 3 | **Diagnosis** | L3 | AMIE 4-step reasoning, DSM-5-TR/HiTOP, assessments | gRPC/HTTP |
| 4 | **Therapy** | L3 | 6 modalities, stepped care, homework, progress | gRPC/HTTP |
| 5 | **Personality** | L4 | OCEAN ensemble, MoEL empathy, style adaptation | gRPC/HTTP |
| 6 | **Memory** | L5 | 5-tier hierarchy, context assembly, decay, consolidation | gRPC/HTTP |
| 7 | **User** | L6 | User profiles, auth, consent, GDPR | gRPC/HTTP |
| 8 | **Notification** | L6 | Email, SMS, push notifications | HTTP |
| 9 | **Analytics** | L6 | Event consumption, aggregation, reporting | HTTP |
| 10 | **Config** | L0 | Centralized config, secrets, feature flags | HTTP |

### 7 Agents in Orchestrator

| Priority | Agent | Authority | Status in Design |
|----------|-------|-----------|-----------------|
| 0 | Safety Agent | Override all | Required |
| 1 | Diagnosis Agent | Clinical | Required |
| 1 | Therapy Agent | Clinical | Required |
| 1 | Assessment Agent | Clinical | Required |
| 2 | Personality Agent | Support | Required |
| 2 | Emotion Agent | Support | Required |
| 3 | Chat Agent | Fallback | Required |

### 6 Shared Libraries

| Library | Exports | Purpose |
|---------|---------|---------|
| solace-common | Entities, exceptions, enums, utils | DDD foundation |
| solace-events | Schemas, publisher, consumer, DLQ, config | Event infrastructure |
| solace-infrastructure | Postgres, Redis, Weaviate, health, observability | Data layer |
| solace-security | Auth, authorization, encryption, audit, PHI | Security |
| solace-ml | LLM client, Anthropic/OpenAI adapters, embeddings, inference | ML abstraction |
| solace-testing | Fixtures, mocks, factories, integration, contracts | Test utilities |

---

## 3. Orchestration (LangGraph)

### Graph Topology

```
START → SAFETY_PRE_CHECK → {
  crisis → CRISIS_HANDLER → END
  safe → SUPERVISOR → ROUTER → {
    clinical → [DIAGNOSIS, THERAPY] (parallel) → AGGREGATOR
    support → [PERSONALITY, CHAT] (parallel) → AGGREGATOR
    mixed → [DIAGNOSIS, THERAPY, PERSONALITY] → AGGREGATOR
  }
}
AGGREGATOR → STYLE_APPLICATOR → SAFETY_POST_CHECK → {
  pass → END
  filter → SAFETY_FILTER → END
}
```

### State Schema (SolaceState TypedDict)

- Identities: user_id, session_id, conversation_id
- Input: current_message, message_timestamp
- Context: assembled_context, user_profile, personality_profile, treatment_context
- Safety: safety_flags, crisis_level, safety_override_active
- Agent Outputs: diagnosis_output, therapy_output, personality_output, emotion_output
- Response: aggregated_response, styled_response, final_response
- Routing: active_agents, next_agent, routing_reason

### Supervisor/Router

- Intent classification: NLP analysis of user message
- Agent selection based on: message type, session phase, detected needs
- Fallback: Chat Agent if primary agent inappropriate
- Intent→Agent mapping: Symptoms→Diagnosis, Coping→Therapy, Validation→Personality, General→Chat, Crisis→Safety (override)

### Error Handling

| Error | Fallback |
|-------|----------|
| Agent unavailable | Chat Agent |
| LLM timeout | Pre-configured response |
| Safety service down | FAIL-SAFE to crisis mode |
| Memory service down | In-memory context only |
| Database loss | Redis cache, queue updates |

---

## 4. Safety Architecture

### 4-Layer Crisis Detection

| Layer | Method | Latency Target | Details |
|-------|--------|----------------|---------|
| L1 | Keyword Detection (trie/FSM) | <10ms | Crisis keywords: "kill myself", "end it all", "harm myself", etc. |
| L2 | Sentiment/NLU Analysis (fine-tuned BERT) | <50-100ms | Hopelessness, despair, emotional valence scoring |
| L3 | Pattern Matching (time-series) | <100-500ms | Escalation patterns, temporal risk clustering, behavioral changes |
| L4 | LLM Assessment (Claude/GPT) | <500ms | Sarcasm detection, nuanced intent, passive vs active ideation |

### 5-Level Risk Classification

| Level | Score Range | Color | Response |
|-------|------------|-------|----------|
| NONE | 0.0-0.2 | GREEN | Continue normally |
| LOW | 0.2-0.4 | YELLOW | Monitor, wellness resources |
| ELEVATED | 0.4-0.6 | ORANGE | Assess, crisis hotline info |
| HIGH | 0.6-0.85 | RED | Intervene, clinician notification, safety plan |
| CRITICAL | 0.85-1.0 | EMERGENCY | Escalate immediately, 988/911, session lock |

### Risk Scoring Formula

```
risk_score = 0.6 × (layer_flags) + 0.2 × user_history + 0.15 × context_severity + 0.05 × temporal_patterns
```

### Fusion Weights (alternative formula from integration doc)

```
risk = L1(0.3) + L2(0.2) + L3(0.2) + L4(0.3)
```

### Protective Factors

| Factor | Weight | Detection |
|--------|--------|-----------|
| Social Support | 0.6 | Keywords: "friends", "family" |
| Family Connection | 0.7 | Keywords: "parents", "siblings" |
| Treatment Engagement | 0.7 | Keywords: "therapy", "counseling" |
| Active Treatment Plan | 0.8 | Context from profile |
| Positive Outlook | 0.5 | Keywords: "hope", "future", "plans" |
| Coping Skills | 0.5 | Keywords: "coping", "managing" |

### Escalation Protocol

- **CRITICAL**: Auto-escalate → 988 Lifeline + 911 + clinician alert + session lock + stay-with-user
- **HIGH**: Auto-escalate (configurable) → safety planning + resources + urgent follow-up (24-48h)
- **ELEVATED**: Manual escalation available → check-in + monitoring + follow-up (1 week)

### Safety Gates in Pipeline

- **Pre-Check**: Before routing to agents. Crisis → bypass to Crisis Handler.
- **Post-Check**: Before response delivery. Unsafe content → filter/reject. Append crisis resources for HIGH/CRITICAL.
- **Override Authority**: Safety > Supervisor > Clinical > Support

### Contraindication Matrix

- **Absolute**: Exposure therapy + Active psychosis, Behavioral activation + Catatonic depression
- **Relative**: DBT diary cards + First session, Values exploration + Acute crisis

---

## 5. Diagnosis Module

### AMIE-Inspired 4-Step Chain-of-Reasoning

| Step | Name | Purpose | LLM Calls | Latency |
|------|------|---------|-----------|---------|
| 1 | **Analyze** | Symptom extraction, temporal analysis, risk indicators | 1 | <1s |
| 2 | **Hypothesize** | DSM-5-TR mapping, differential generation, HiTOP scoring | 1 | 2-5s |
| 3 | **Challenge** | Devil's Advocate bias detection, alternative explanations | 1 | 2-5s |
| 4 | **Synthesize** | Perspective integration, confidence calibration, response | 1 | 2-5s |

**Total per turn**: 4 LLM calls, 6-16 seconds

### Devil's Advocate (Anti-Sycophancy)

**3 Bias Types Specified in Design** (implementation has 6):

| Bias | Detection | Penalty |
|------|-----------|---------|
| Confirmation Bias | Only seeking supporting evidence | -0.10 × strength |
| Premature Closure | Stopping before sufficient criteria met | -0.15 × severity |
| Anchoring Bias | Over-weighting initial impressions | -0.05 × severity |

### Bayesian Confidence Calibration

```
Posterior = (Likelihood × Prior) / Evidence

Prior = base_prevalence(age, sex) × risk_factor_multiplier
Likelihood = criteria_coverage(0.4) + specificity(0.3) + exclusion_criteria(0.3)
```

**Sample Consistency Method**: N=3-5 LLM samples, measure agreement:
- <60% agreement: "Uncertain" (wide CI)
- 60-80%: Moderate confidence
- >80%: High confidence (narrow CI)

**Confidence Components (weighted)**:
- criteria_coverage: 0.30
- sample_consistency: 0.25
- information_completeness: 0.20
- temporal_stability: 0.15
- instrument_alignment: 0.10

### Confidence Thresholds

| Range | Meaning | Action |
|-------|---------|--------|
| 0.70-1.00 | High | Proceed with recommendations |
| 0.50-0.70 | Moderate | Request more information |
| 0.30-0.50 | Low | Flag for clinician review |
| 0.00-0.30 | Very Low | Escalate, don't generate insight |

### Assessment Tools

| Instrument | Items | Scale | Score Range | Severity Thresholds |
|------------|-------|-------|-------------|-------------------|
| PHQ-9 | 9 | 0-3 | 0-27 | 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 ModSevere, 20-27 Severe |
| GAD-7 | 7 | 0-3 | 0-21 | 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-21 Severe |
| PCL-5 | 20 | 0-4 | 0-80 | ≥31-33 Probable PTSD |
| ORS | 4 | VAS | 0-40 | <25 Clinical distress |
| SRS | 4 | VAS | 0-40 | <36 Alliance concern |

### HiTOP Dimensional Model (6 Spectra)

1. Internalizing (depression, anxiety)
2. Thought Disorder (psychosis, reality distortion)
3. Disinhibited Externalizing (impulsivity, substance use)
4. Antagonistic Externalizing (aggression, callousness)
5. Detachment (withdrawal, emotional constriction)
6. Somatoform (somatic symptoms)

### Session State Machine (5 Phases)

RAPPORT → HISTORY_TAKING → ASSESSMENT → DIAGNOSIS → CLOSURE

With CRISIS override from any state. Transition criteria specified per phase.

---

## 6. Therapy Module

### 6 Therapeutic Modalities

| Modality | Techniques | Target Conditions |
|----------|-----------|-------------------|
| **CBT** | Cognitive restructuring, behavioral activation, thought records, exposure, problem solving | Depression, anxiety, PTSD, panic |
| **DBT** | 4 modules: Mindfulness, Distress Tolerance (TIPP/STOP), Emotion Regulation, Interpersonal (DEAR MAN/GIVE/FAST) | BPD, emotion dysregulation, chronic suicidality |
| **ACT** | 6 core processes: Defusion, Acceptance, Values, Committed Action, Present Moment, Self-as-Context | Anxiety, depression, chronic pain, avoidance |
| **MI** | OARS skills, Change Talk elicitation (DARN), Rolling with Resistance | Low motivation, ambivalence, substance use |
| **Mindfulness** | 4-7-8 breathing, body scan, 5-4-3-2-1 grounding, loving-kindness | All conditions (adjunct) |
| **SFBT** | Miracle question, exception finding, scaling questions, coping questions | Adjustment, stress, engagement |

### 4-Stage Technique Selection Algorithm

```
Stage 1: Clinical Filter (diagnosis match, severity, contraindications)
Stage 2: Personalization (Big Five trait affinity weights)
Stage 3: Context Ranking (session phase, treatment plan, recency)
Stage 4: Final Score = 0.4×clinical + 0.3×personal + 0.2×context + 0.1×history
```

### PHQ-9 Stepped Care (5 Levels)

| Step | PHQ-9 | Frequency | Intensity | Human |
|------|-------|-----------|-----------|-------|
| 0 | 0-4 | Monthly | Self-guided only | None |
| 1 | 5-9 | Bi-weekly | Self-guided + coping | Minimal |
| 2 | 10-14 | 1-2x/week | Guided CBT | Periodic |
| 3 | 15-19 | 2-3x/week | Intensive protocol | Weekly |
| 4 | 20+ | Daily | AI adjunct only; human primary | Required |

### 4-Phase Treatment Protocol

1. **Foundation** (Weeks 1-2): Alliance building, assessment, baseline, safety plan
2. **Active Treatment** (Weeks 3-10): Core skills, technique delivery, homework, weekly measures
3. **Consolidation** (Weeks 11-12): Skill generalization, relapse prevention, coping cards
4. **Maintenance** (Ongoing): Monthly check-ins, boosters, monitoring

### Treatment Response Classification (4-6 Week Review)

| Type | Symptom Reduction | Action |
|------|------------------|--------|
| Responder | ≥50% | Continue, begin consolidation |
| Partial | 25-49% | Augment, add adjunct modality |
| Non-Responder | <25% | Switch modality, clinician review |
| Deterioration | Increasing ≥RCI | IMMEDIATE safety assessment |

**RCI**: PHQ-9 ≥5 points, GAD-7 ≥4 points

### Homework System

- Assignment at session closing with personalization
- Between-session tracking with reminders (24h before, day-of)
- Session opening review with structured dialogue
- Adaptation based on completion × effectiveness matrix

### Session Structure (State Machine)

PreSession → Opening (3-5 min: mood check, bridge, agenda) → Working (15-25 min: homework review, technique, practice) → Closing (3-5 min: summary, homework, SRS) → PostSession

---

## 7. Personality Module

### Big Five (OCEAN) Trait Model

| Trait | Range | High (>0.7) | Low (<0.3) | Therapeutic Impact |
|-------|-------|-------------|------------|-------------------|
| Openness | 0-1 | Exploratory, abstract | Practical, structured | ACT vs CBT preference |
| Conscientiousness | 0-1 | Organized, goal-oriented | Flexible, spontaneous | Homework complexity |
| Extraversion | 0-1 | Energetic, talkative | Reserved, reflective | Response length/energy |
| Agreeableness | 0-1 | Cooperative, trusting | Skeptical, challenging | Validation level |
| Neuroticism | 0-1 | Anxious, reactive | Calm, stable | Safety emphasis |

### 3-Source Ensemble Detection

| Source | Model | Dimensions | Weight | Accuracy |
|--------|-------|-----------|--------|----------|
| RoBERTa Large | Fine-tuned transformer | 768-dim BERT embeddings | 0.5 | R²=0.24 |
| LLM Zero-Shot | Claude/GPT-4 | Structured output | 0.3 | r=0.29-0.38 |
| LIWC Features | 93 psycholinguistic categories | Feature mapping | 0.2 | r=0.15-0.40 |

### 5 Style Parameters (StyleAdapter Output)

| Parameter | Formula | Range | Meaning |
|-----------|---------|-------|---------|
| Warmth | f(Agreeableness, Neuroticism) | 0-1 | Professional → Nurturing |
| Structure | f(Conscientiousness) | 0-1 | Flexible → Organized |
| Complexity | f(Openness) | 0-1 | Simple → Abstract |
| Directness | f(Agreeableness, Conscientiousness) | 0-1 | Indirect → Direct |
| Energy | f(Extraversion) | 0-1 | Calm → Enthusiastic |

### MoEL (Mixture of Empathetic Listeners)

- 32 emotion classes with softmax distribution
- 4 primary listeners: Sadness, Anxiety, Anger, Joy
- 28 additional specialized listeners
- Meta-Listener soft combination layer
- 3-component empathy: Cognitive, Affective, Compassionate

### Multimodal Fusion (Design)

| Modality | Features | Encoder | Dimensions |
|----------|----------|---------|-----------|
| Text | Messages, word choice | BERT/RoBERTa | 768 |
| Voice | Pitch, rate, energy | Wav2Vec 2.0 / CNN-BiLSTM | 512 |
| Behavioral | Response patterns, engagement | Dense engineering | 128 |

**Late Fusion**: Concatenate (1408-dim) → Cross-Modal Attention → Weighted combination

### Profile Evolution

| Phase | Sessions | Confidence | Variance |
|-------|----------|------------|----------|
| Initialization | 1 | 0.3 | ±0.20 |
| Refinement | 2-5 | 0.4-0.6 | ±0.12 |
| Stabilization | 6+ | 0.7-1.0 | ±0.06 |

EMA aggregation: α=0.3, confidence grows with √n, significant change >0.15 triggers versioning.

### Cultural Adaptation (Hofstede Dimensions)

- Individualism vs Collectivism
- Power Distance
- Uncertainty Avoidance
- Masculinity vs Femininity

---

## 8. Memory Module

### 5-Tier Cognitive Hierarchy

| Tier | Name | Backend | Retention | Latency |
|------|------|---------|-----------|---------|
| 1 | Input Buffer | In-memory | Request duration | <1ms |
| 2 | Working Memory | Redis + In-memory | Session duration | <10ms |
| 3 | Session Memory | Redis + PostgreSQL | 24h post-session | <50ms |
| 4 | Episodic Memory | PostgreSQL + Weaviate | Decay-based | <200ms |
| 5 | Semantic Memory | Weaviate + PostgreSQL | Permanent (versioned) | <200ms |

### Ebbinghaus Decay Formula

```
R(t) = e^(-λt) × S
```

| Category | λ (decay rate) | Half-Life | Examples |
|----------|---------------|-----------|---------|
| Permanent | 0.00 | ∞ | Safety plans, diagnoses, crisis history, emergency contacts |
| Long-term | 0.02/day | 35 days | Treatment plans, major milestones |
| Medium-term | 0.05/day | 14 days | Session summaries, homework |
| Short-term | 0.15/day | 5 days | Casual conversation, temporary context |

**Safety override**: λ=0 (never decays). S increases 1.5× per recall (max 3×).
**Archive**: retention < 0.3 → S3 Glacier. Delete: retention < 0.1.

### Context Assembly (Relevance Scoring)

```
Relevance = Semantic_Similarity(0.4) + Recency_Weight(0.3) + Importance_Score(0.2) + Source_Authority(0.1)
```

**Recency**: weight(age_days) = e^(-age_days / 30)

### Token Budget Allocation (4000-8000 tokens)

| Component | Tokens | Priority |
|-----------|--------|----------|
| System Prompt | 500-1000 | Highest |
| Safety Context | 200-400 | Critical |
| User Profile | 200-400 | High |
| Recent Messages | 2000-4000 | High |
| Retrieved Context | 1000-2000 | Medium |
| Response Buffer | 1000-2000 | Reserved |

### Vector Storage (Weaviate)

**5 Collections**: ConversationMemory, SessionSummary, TherapeuticInsight, UserFact, CrisisEvent

**Embedding**: text-embedding-3-small (1536-dim), backup: all-MiniLM-L6-v2 (384-dim)

**Hybrid Search**: BM25 + semantic, alpha=0.5, certainty threshold=0.7, limit=20

### Consolidation Pipeline

Session end → Summary generation (Tier 3→4) → Fact extraction (→Tier 5) → Knowledge graph update → Embedding computation → Storage → Archive scheduling → Event publication

---

## 9. Event-Driven Architecture

### Kafka Topics

| Topic | Partitions | Retention | Purpose |
|-------|-----------|-----------|---------|
| solace.sessions | 6-12 | 7-30 days | Session lifecycle |
| solace.assessments | 3-6 | 30-90 days | Diagnosis events |
| solace.therapy | 3-6 | 30 days | Therapy delivery |
| solace.safety | 6-12 | 1 year | Crisis/safety (immutable) |
| solace.memory | 2-6 | 7-30 days | Memory consolidation |
| solace.personality | 6 | 30 days | Personality updates |
| solace.analytics | 2 | 90 days | Aggregate analytics |
| solace.audit | 12 | 10 years | Immutable audit log |
| solace.dlq | 1 | 30 days | Dead letter queue |

**Partition key**: user_id (ordering per user)
**Replication factor**: 3
**Min in-sync replicas**: 2

### Transactional Outbox Pattern

1. Service writes event to local `outbox` table in same DB transaction
2. Worker polls outbox every 100ms for unpublished events
3. Worker publishes to Kafka, updates outbox with published_at
4. Idempotency key prevents duplicates

### Dead Letter Queue

- Trigger: Consumer failure after 3 retries with backoff
- Format: original message + error details + failure count
- Alert: >10 messages/hour
- Resolution: Manual review → fix → replay

### Event Schema Versioning

- Format: Avro with Schema Registry (or Pydantic JSON)
- Versioning: Semantic (backward compatible — new fields have defaults)
- Subject naming: `{topic}-value` and `{topic}-key`

---

## 10. Infrastructure & Deployment

### Service Communication

**Design specifies gRPC + Protobuf** for inter-service communication with:
- Service mesh: Istio (mTLS, circuit breaking, retries)
- Service discovery: Kubernetes DNS
- Load balancing: Envoy (round-robin)

**API Gateway**: Kong + Istio Ingress
- Routes: /api/v1/chat/* → orchestrator, /api/v1/sessions/* → session-service, etc.
- Auth: JWT (RS256, 1h expiry) + refresh tokens (7d)
- Rate limiting: 100 req/min per user, 10 msg/min per session, 1000 req/min per IP

### Kubernetes Design

- Multi-AZ (3 availability zones)
- Namespaces: solace-prod, solace-data, solace-monitoring
- Service replicas: 2-10 per service (HPA: CPU>70%, Memory>80%)
- StatefulSets for: PostgreSQL (3 replicas), Redis (6 nodes cluster), Kafka (3 brokers)

### Database Architecture

- **PostgreSQL 16+**: Primary + 2 standbys, Patroni failover, column-level PHI encryption
- **Redis 7.4+**: Cluster mode (6 nodes), Sentinel HA, RDB+AOF persistence
- **Weaviate**: Persistent directory, hybrid search, collection-based partitioning
- **S3/Glacier**: Archives, backups, 6-year audit retention

### CI/CD Pipeline

GitHub Actions → Lint (Ruff/mypy/Black) → Unit Tests (>80%) → Docker Build → Integration Tests → Push to ECR → Deploy to Staging → E2E Tests → Promote to Prod (manual)

### HA/DR Targets

| Metric | Target |
|--------|--------|
| RPO | <1 minute |
| RTO | <5 minutes |
| Availability | 99.95% SLA |
| Safety SLO | 99.99% |

---

## 11. API Contracts

### External REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /api/v1/chat/message | POST | Send message, get response |
| /api/v1/ws/chat | WS | WebSocket streaming |
| /api/v1/session/start | POST | Start session |
| /api/v1/session/end | POST | End session |
| /api/v1/assessment/phq9 | POST | Submit PHQ-9 |
| /api/v1/assessment/gad7 | POST | Submit GAD-7 |
| /api/v1/profile | GET/PUT | User profile |
| /api/v1/profile/personality | GET | OCEAN scores |
| /api/v1/safety/status | GET | Current risk level |
| /api/v1/safety/report | POST | Report concern |
| /api/v1/clinician/sessions | GET | Clinician oversight |
| /health, /ready, /live | GET | Health checks |

### Internal gRPC Contracts (Protobuf)

- **Diagnosis**: `Assess(AssessmentRequest) → DiagnosisResponse`
- **Therapy**: `GetIntervention(InterventionRequest) → InterventionResponse`
- **Memory**: `GetContext(ContextRequest) → ContextResponse`
- **Personality**: `DetectTraits(TraitRequest) → PersonalityResponse`

### Response Envelope Format

```json
{
  "status": "success|error",
  "data": { ... },
  "meta": { "trace_id": "...", "response_time_ms": 234 }
}
```

---

## 12. Cross-Cutting Concerns

### Authentication & Authorization

- **Client Auth**: OAuth2 (Google/Apple) or email/password + MFA
- **JWT**: RS256, 15-min access token, 7-day refresh
- **Service-to-Service**: mTLS (Istio Citadel) + service account tokens
- **RBAC Roles**: USER, CLINICIAN, ADMIN, SYSTEM
- **Scopes**: chat:read/write, assessment:read/write, profile:read/write, clinician:read, admin:*
- **DB**: Row-level security on patient data

### HIPAA Compliance

- AES-256-GCM encryption at rest (column-level for PHI)
- TLS 1.3 in transit
- Immutable audit logs (10-year retention)
- Automatic session timeout (15 min idle)
- BAAs with all vendors
- Breach notification <72h

### Observability

- **Metrics**: Prometheus (15s scrape) + Grafana dashboards
- **Logs**: Structured JSON with trace IDs, PII-masked (Loki or ELK)
- **Traces**: Jaeger/OpenTelemetry distributed tracing
- **Alerting**: AlertManager with rules for safety latency, error rates, etc.

### Error Handling

- Strict error hierarchy: ValidationError(4xx), AuthN(401), AuthZ(403), NotFound(404), RateLimit(429), ServiceUnavailable(503), Internal(500), SafetyError(special)
- No cross-component fallbacks (fail-fast)
- Circuit breaker: 5 failures → open → 30s → half-open → 3 test requests
- Retry: 3 attempts, exponential backoff (100ms, 200ms, 400ms)

---

## 13. Performance Targets

### Latency (p99)

| Operation | Target |
|-----------|--------|
| Orchestrator total (p50) | <1-2s |
| Orchestrator total (p99) | <3-5s |
| Safety Layer 1 | <10ms |
| Safety total | <100ms |
| Memory context assembly | <100-200ms |
| Diagnosis (4-step) | <3-10s |
| Therapy technique selection | <300-600ms |
| Personality detection (cached) | <200ms |
| Personality detection (cold) | <1s |
| Database query | <10ms |
| WebSocket TTFT | <500ms |

### Throughput

| Service | Target |
|---------|--------|
| Orchestrator | 1000 RPS |
| Safety | 5000 RPS |
| Memory | 2000 RPS |

### Reliability

| Service | SLO |
|---------|-----|
| Safety | 99.99% |
| Others | 99.9% |
| Cache hit rate | >80% |
| Safety false negative rate | <0.5% |

---

## 14. Design Inconsistencies Found

During analysis, the following inconsistencies were found **within the design documents themselves**:

### 14.1 Vector Database: ChromaDB vs Weaviate

- **Core Architecture doc** (`Solace-AI-Final-System-Architecture.md`): Specifies **ChromaDB 0.5+** as vector store with `all-MiniLM-L6-v2` embeddings
- **Module Architecture docs** (Memory, Personality): Specify **Weaviate** with `text-embedding-3-small` (1536-dim)
- **Implementation Plan**: References both ChromaDB and Weaviate in different sections
- **Resolution**: Codebase uses Weaviate — module docs take precedence

### 14.2 Risk Score Ranges

- **Core doc**: GREEN(0-0.2), YELLOW(0.2-0.4), ORANGE(0.4-0.7), RED(0.7-0.9), EMERGENCY(>0.9)
- **Integration doc**: NONE(0-0.2), LOW(0.2-0.4), ELEVATED(0.4-0.6), HIGH(0.6-0.85), CRITICAL(0.85-1.0)
- **Resolution**: Integration doc's 5-level named system is more granular and matches implementation

### 14.3 Risk Scoring Formula

- **Core doc**: `risk = 0.6×layers + 0.2×history + 0.15×context + 0.05×temporal`
- **Integration doc**: `risk = L1(0.3) + L2(0.2) + L3(0.2) + L4(0.3)`
- **Resolution**: Both may coexist (layer fusion vs. final risk calculation)

### 14.4 Communication Protocol

- **Core doc**: gRPC + Protobuf for all inter-service communication
- **Implementation Plan**: References both gRPC and HTTP/FastAPI endpoints
- **Resolution**: Codebase uses HTTP/REST (FastAPI) — gRPC was aspirational

### 14.5 Memory Tier Numbering

- **Core doc**: T1=Working Memory, T2=Session Buffer, T3=Episodic, T4=Semantic
- **Memory module doc**: T1=Input Buffer, T2=Working Memory, T3=Session, T4=Episodic, T5=Semantic
- **Resolution**: Memory module doc's 5-tier numbering is canonical

### 14.6 Directory Structure

- **Implementation Plan**: `libs/` for shared libraries
- **Codebase**: `src/` prefix (e.g., `src/solace_common/`)
- **Resolution**: Codebase structure takes precedence

### 14.7 Safety Layer Count and Naming

- **Core doc**: 3 layers (Keyword, Semantic NLU, Pattern)
- **Safety integration doc**: 4 layers (Keyword, Sentiment, Pattern, LLM Assessment)
- **Safety competitive analysis**: 4 layers confirmed
- **Resolution**: 4 layers is the canonical design

### 14.8 Bias Types in Devil's Advocate

- **Diagnosis design doc**: 3 bias types (Confirmation, Premature Closure, Anchoring)
- **Codebase implementation**: 6 bias types (adds Availability, Base Rate Neglect, Attribution)
- **Resolution**: Implementation exceeds design (positive deviation)

---

## Appendix A: Competitive Positioning Summary

| Module | Competitive Rating | Key Differentiator |
|--------|-------------------|-------------------|
| Orchestration | World-Class | LangGraph StateGraph with clinical agents |
| Safety | World-Class | 4-layer detection + protective factors (unique) |
| Diagnosis | Industry-Leading | AMIE 4-step + Devil's Advocate + Bayesian calibration |
| Therapy | Industry-Leading | 6 modalities (most comprehensive) + 4-stage selection |
| Personality | Innovative | 3-source OCEAN ensemble + MoEL empathy |
| Memory | Industry-Leading | 5-tier cognitive hierarchy + Ebbinghaus decay (unique) |

**Critical gaps for market**: No published RCTs, no FDA clearance pathway initiated, no voice/audio modality.

## Appendix B: Implementation Scope

- **Total planned**: ~54,000 LOC across 180+ files in 36 batches over 12 weeks
- **10 phases**: Libraries → Infrastructure → Safety → Memory → Diagnosis/Therapy/Personality (parallel) → Orchestrator → API Gateway → Supporting → Integration
- **Gate criteria**: Defined per phase (safety <10ms, memory <100ms, etc.)
