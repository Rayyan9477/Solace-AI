# Solace-AI Architecture Diagrams

> **Version**: 1.0
> **Last Updated**: December 22, 2025
> **Purpose**: Visual representation of system architecture, data flows, and component interactions

---

## Table of Contents

1. [High-Level System Architecture](#1-high-level-system-architecture)
2. [Agent Orchestration Flow](#2-agent-orchestration-flow)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Memory System Architecture](#4-memory-system-architecture)
5. [Diagnosis Pipeline Flow](#5-diagnosis-pipeline-flow)
6. [API Request Lifecycle](#6-api-request-lifecycle)
7. [Security & Authentication Flow](#7-security--authentication-flow)
8. [Component Dependency Graph](#8-component-dependency-graph)

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph CLIENT["Client Layer"]
        WEB[Web Client]
        MOBILE[Mobile App]
        CLI[CLI Interface]
        VOICE[Voice Interface]
    end

    subgraph API["API Gateway Layer"]
        FASTAPI[FastAPI Server]
        AUTH[JWT Authentication]
        RATE[Rate Limiter]
        VALID[Input Validator]
    end

    subgraph ORCH["Orchestration Layer"]
        AO[Agent Orchestrator]
        SUP[Supervisor Agent]
        EB[Event Bus]
        CB[Circuit Breaker]
    end

    subgraph AGENTS["Agent Layer"]
        subgraph CORE["Core Agents"]
            CHAT[Chat Agent]
            EMO[Emotion Agent]
            SAFE[Safety Agent]
        end
        subgraph CLINICAL["Clinical Agents"]
            DIAG[Diagnosis Agent]
            THER[Therapy Agent]
            CRISIS[Crisis Detection]
        end
        subgraph SUPPORT["Support Agents"]
            PERSON[Personality Agent]
            RESEARCH[Research Agent]
        end
    end

    subgraph SERVICES["Service Layer"]
        DS[Diagnosis Service]
        US[User Service]
        SS[Session Service]
        NS[Notification Service]
    end

    subgraph INFRA["Infrastructure Layer"]
        subgraph MEMORY["Memory Systems"]
            EMS[Enhanced Memory]
            SEM[Semantic Memory]
            CONV[Conversation Memory]
        end
        subgraph DATA["Data Storage"]
            VDB[(Vector DB)]
            CACHE[(Redis Cache)]
            FILES[(File Storage)]
        end
        subgraph EXTERNAL["External Services"]
            LLM[LLM Providers]
            TTS[TTS Service]
            ASR[ASR Service]
        end
    end

    CLIENT --> API
    API --> ORCH
    ORCH --> AGENTS
    AGENTS --> SERVICES
    SERVICES --> INFRA

    AO <--> EB
    SUP --> AO

    CHAT <--> EMO
    CHAT <--> SAFE
    DIAG <--> DS

    DS --> VDB
    DS --> EMS
    US --> CACHE
    SS --> CONV
```

---

## 2. Agent Orchestration Flow

```mermaid
flowchart TD
    START([User Message]) --> VALID{Input Validation}

    VALID -->|Invalid| REJECT[Return Error Response]
    VALID -->|Valid| SAFETY{Safety Check}

    SAFETY -->|Unsafe| CRISIS_FLOW[Crisis Intervention Flow]
    SAFETY -->|Safe| EMOTION[Emotion Analysis]

    EMOTION --> CONTEXT[Load Context from Memory]
    CONTEXT --> ROUTE{Route to Agent}

    ROUTE -->|General Chat| CHAT_AGENT[Chat Agent]
    ROUTE -->|Clinical Need| CLINICAL_AGENT[Clinical Agent]
    ROUTE -->|Assessment| ASSESS_AGENT[Assessment Agent]
    ROUTE -->|Therapy| THERAPY_AGENT[Therapy Agent]

    CHAT_AGENT --> GENERATE[Generate Response]
    CLINICAL_AGENT --> DIAGNOSE[Run Diagnosis]
    ASSESS_AGENT --> EVALUATE[Run Assessment]
    THERAPY_AGENT --> THERAPEUTIC[Apply Techniques]

    DIAGNOSE --> GENERATE
    EVALUATE --> GENERATE
    THERAPEUTIC --> GENERATE

    GENERATE --> SUPERVISOR{Supervisor Review}

    SUPERVISOR -->|Approved| STORE[Store in Memory]
    SUPERVISOR -->|Needs Revision| REVISE[Revise Response]
    REVISE --> SUPERVISOR

    STORE --> RESPOND([Return Response])

    CRISIS_FLOW --> ESCALATE[Escalate to Human]
    CRISIS_FLOW --> RESOURCES[Provide Crisis Resources]
    RESOURCES --> RESPOND

    style CRISIS_FLOW fill:#ff6b6b
    style SAFETY fill:#ffd93d
    style SUPERVISOR fill:#6bcb77
```

---

## 3. Data Flow Diagram

```mermaid
flowchart LR
    subgraph INPUT["Input Sources"]
        TEXT[Text Input]
        AUDIO[Voice Input]
        FILE[File Upload]
    end

    subgraph PROCESSING["Processing Pipeline"]
        subgraph PREPROCESS["Preprocessing"]
            SANITIZE[Sanitize Input]
            TOKENIZE[Tokenization]
            EMBED[Generate Embeddings]
        end

        subgraph ANALYSIS["Analysis"]
            NLP[NLP Processing]
            SENT[Sentiment Analysis]
            INTENT[Intent Detection]
            ENTITY[Entity Extraction]
        end

        subgraph ENRICHMENT["Context Enrichment"]
            HIST[Conversation History]
            PROF[User Profile]
            KNOW[Knowledge Base]
        end
    end

    subgraph STORAGE["Data Storage"]
        subgraph VECTOR["Vector Storage"]
            CHROMA[(ChromaDB)]
        end
        subgraph STRUCTURED["Structured Data"]
            JSON[(JSON Files)]
            META[(Metadata)]
        end
        subgraph CACHE["Cache Layer"]
            REDIS[(Redis)]
            MEM[(In-Memory)]
        end
    end

    subgraph OUTPUT["Output Generation"]
        LLM_CALL[LLM API Call]
        RESPONSE[Response Formatting]
        TTS_OUT[Text-to-Speech]
    end

    TEXT --> SANITIZE
    AUDIO --> ASR_PROC[ASR Processing] --> SANITIZE
    FILE --> PARSE[Parse File] --> SANITIZE

    SANITIZE --> TOKENIZE --> EMBED
    EMBED --> NLP
    NLP --> SENT & INTENT & ENTITY

    SENT --> HIST
    INTENT --> PROF
    ENTITY --> KNOW

    HIST --> CHROMA
    PROF --> JSON
    KNOW --> CHROMA

    ENRICHMENT --> LLM_CALL
    LLM_CALL --> RESPONSE
    RESPONSE --> TTS_OUT

    CHROMA --> ENRICHMENT
    JSON --> ENRICHMENT
    REDIS --> ENRICHMENT
```

---

## 4. Memory System Architecture

```mermaid
flowchart TB
    subgraph MEMORY_TYPES["Memory Types"]
        subgraph SHORT["Short-Term Memory"]
            CONV_BUF[Conversation Buffer]
            SESSION[Session State]
            WORKING[Working Memory]
        end

        subgraph LONG["Long-Term Memory"]
            EPISODIC[Episodic Memory]
            SEMANTIC[Semantic Memory]
            PROCEDURAL[Procedural Memory]
        end

        subgraph CONTEXT["Context Memory"]
            USER_CTX[User Context]
            THERAPY_CTX[Therapy Context]
            DIAG_CTX[Diagnosis Context]
        end
    end

    subgraph OPERATIONS["Memory Operations"]
        STORE_OP[Store]
        RETRIEVE_OP[Retrieve]
        UPDATE_OP[Update]
        FORGET_OP[Forget/Archive]
    end

    subgraph STORAGE_BACKEND["Storage Backends"]
        VECTOR_STORE[(Vector Store)]
        FILE_STORE[(File System)]
        CACHE_STORE[(Cache)]
    end

    CONV_BUF --> STORE_OP
    SESSION --> STORE_OP
    EPISODIC --> STORE_OP
    SEMANTIC --> STORE_OP
    USER_CTX --> STORE_OP

    STORE_OP --> VECTOR_STORE
    STORE_OP --> FILE_STORE
    STORE_OP --> CACHE_STORE

    RETRIEVE_OP --> CONV_BUF
    RETRIEVE_OP --> EPISODIC
    RETRIEVE_OP --> SEMANTIC

    VECTOR_STORE --> RETRIEVE_OP
    FILE_STORE --> RETRIEVE_OP
    CACHE_STORE --> RETRIEVE_OP

    UPDATE_OP --> LONG
    FORGET_OP --> LONG

    subgraph MEMORY_FLOW["Memory Flow"]
        direction LR
        NEW[New Interaction] --> SHORT
        SHORT -->|Consolidation| LONG
        LONG -->|Retrieval| CONTEXT
        CONTEXT -->|Enrichment| RESPONSE[Response Generation]
    end
```

---

## 5. Diagnosis Pipeline Flow

```mermaid
flowchart TD
    subgraph INPUT_STAGE["Input Stage"]
        USER_INPUT[User Input]
        HISTORY[Conversation History]
        ASSESSMENTS[Prior Assessments]
    end

    subgraph FEATURE_EXTRACTION["Feature Extraction"]
        TEXT_FEAT[Text Features]
        VOICE_FEAT[Voice Features]
        BEHAV_FEAT[Behavioral Features]
        TEMP_FEAT[Temporal Features]
    end

    subgraph ANALYSIS_ENGINE["Analysis Engine"]
        subgraph PRIMARY["Primary Analysis"]
            SYMPTOM[Symptom Detection]
            SEVERITY[Severity Assessment]
            PATTERN[Pattern Recognition]
        end

        subgraph DIFFERENTIAL["Differential Diagnosis"]
            RULE_BASED[Rule-Based Analysis]
            ML_BASED[ML-Based Analysis]
            BAYESIAN[Bayesian Inference]
        end

        subgraph CLINICAL["Clinical Decision Support"]
            DSM5[DSM-5 Criteria Matching]
            RISK[Risk Assessment]
            GUIDELINES[Clinical Guidelines]
        end
    end

    subgraph OUTPUT_STAGE["Output Stage"]
        DIAGNOSIS[Diagnosis Results]
        CONFIDENCE[Confidence Scores]
        RECOMMEND[Recommendations]
        REPORT[Clinical Report]
    end

    USER_INPUT --> TEXT_FEAT
    USER_INPUT --> VOICE_FEAT
    HISTORY --> BEHAV_FEAT
    ASSESSMENTS --> TEMP_FEAT

    TEXT_FEAT --> SYMPTOM
    VOICE_FEAT --> SEVERITY
    BEHAV_FEAT --> PATTERN
    TEMP_FEAT --> PATTERN

    SYMPTOM --> RULE_BASED
    SEVERITY --> ML_BASED
    PATTERN --> BAYESIAN

    RULE_BASED --> DSM5
    ML_BASED --> RISK
    BAYESIAN --> GUIDELINES

    DSM5 --> DIAGNOSIS
    RISK --> CONFIDENCE
    GUIDELINES --> RECOMMEND

    DIAGNOSIS --> REPORT
    CONFIDENCE --> REPORT
    RECOMMEND --> REPORT

    style RISK fill:#ff6b6b
    style DSM5 fill:#4ecdc4
```

---

## 6. API Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant G as API Gateway
    participant A as Auth Middleware
    participant V as Validator
    participant O as Orchestrator
    participant AG as Agent
    participant S as Service
    participant M as Memory
    participant L as LLM

    C->>G: HTTP Request
    G->>A: Validate JWT Token

    alt Invalid Token
        A-->>C: 401 Unauthorized
    else Valid Token
        A->>V: Validate Input

        alt Invalid Input
            V-->>C: 400 Bad Request
        else Valid Input
            V->>O: Route Request
            O->>O: Select Agent
            O->>AG: Process Message

            AG->>M: Load Context
            M-->>AG: Context Data

            AG->>S: Call Service
            S->>L: LLM Request
            L-->>S: LLM Response
            S-->>AG: Service Response

            AG->>M: Store Response
            AG-->>O: Agent Response

            O->>O: Supervisor Review
            O-->>G: Final Response
            G-->>C: 200 OK + Response
        end
    end
```

---

## 7. Security & Authentication Flow

```mermaid
flowchart TD
    subgraph AUTH_FLOW["Authentication Flow"]
        LOGIN[Login Request] --> VALIDATE_CREDS{Validate Credentials}
        VALIDATE_CREDS -->|Invalid| AUTH_FAIL[Authentication Failed]
        VALIDATE_CREDS -->|Valid| GEN_TOKEN[Generate JWT]
        GEN_TOKEN --> STORE_TOKEN[Store in Session]
        STORE_TOKEN --> RETURN_TOKEN[Return Token to Client]
    end

    subgraph REQUEST_FLOW["Request Authorization"]
        REQUEST[API Request] --> EXTRACT[Extract JWT]
        EXTRACT --> VERIFY{Verify Token}
        VERIFY -->|Invalid| REJECT[401 Unauthorized]
        VERIFY -->|Expired| REFRESH{Refresh Token?}
        REFRESH -->|Yes| GEN_NEW[Generate New Token]
        REFRESH -->|No| REJECT
        VERIFY -->|Valid| AUTHORIZE{Check Permissions}
        AUTHORIZE -->|Denied| FORBIDDEN[403 Forbidden]
        AUTHORIZE -->|Granted| PROCESS[Process Request]
    end

    subgraph SECURITY_LAYERS["Security Layers"]
        INPUT_VAL[Input Validation]
        RATE_LIMIT[Rate Limiting]
        CSRF[CSRF Protection]
        HIPAA[HIPAA Compliance]
        ENCRYPT[Data Encryption]
    end

    PROCESS --> INPUT_VAL
    INPUT_VAL --> RATE_LIMIT
    RATE_LIMIT --> CSRF
    CSRF --> HIPAA
    HIPAA --> ENCRYPT
    ENCRYPT --> EXECUTE[Execute Business Logic]

    style AUTH_FAIL fill:#ff6b6b
    style REJECT fill:#ff6b6b
    style FORBIDDEN fill:#ff6b6b
    style HIPAA fill:#4ecdc4
```

---

## 8. Component Dependency Graph

```mermaid
flowchart TD
    subgraph CORE["Core Components"]
        BASE_AGENT[BaseAgent]
        EXCEPTIONS[Exceptions]
        INTERFACES[Interfaces]
        FACTORIES[Factories]
    end

    subgraph AGENTS["Agents"]
        CHAT_A[ChatAgent]
        EMO_A[EmotionAgent]
        SAFE_A[SafetyAgent]
        DIAG_A[DiagnosisAgent]
        THER_A[TherapyAgent]
    end

    subgraph SERVICES["Services"]
        DIAG_S[DiagnosisService]
        USER_S[UserService]
        SESSION_S[SessionService]
    end

    subgraph INFRA["Infrastructure"]
        MEMORY[MemorySystem]
        VECTOR[VectorDB]
        CONFIG[Configuration]
        DI[DI Container]
        EVENTS[EventBus]
    end

    subgraph EXTERNAL["External"]
        LLM_PROV[LLM Provider]
        VOICE_PROV[Voice Provider]
    end

    %% Core Dependencies
    CHAT_A --> BASE_AGENT
    EMO_A --> BASE_AGENT
    SAFE_A --> BASE_AGENT
    DIAG_A --> BASE_AGENT
    THER_A --> BASE_AGENT

    BASE_AGENT --> INTERFACES
    BASE_AGENT --> EXCEPTIONS

    %% Service Dependencies
    DIAG_A --> DIAG_S
    DIAG_S --> MEMORY
    DIAG_S --> VECTOR

    USER_S --> MEMORY
    SESSION_S --> MEMORY

    %% Infrastructure Dependencies
    MEMORY --> CONFIG
    VECTOR --> CONFIG

    %% Factory Dependencies
    FACTORIES --> LLM_PROV
    FACTORIES --> VOICE_PROV
    FACTORIES --> MEMORY

    %% DI Container
    DI --> FACTORIES
    DI --> SERVICES
    DI --> AGENTS

    %% Event Bus
    EVENTS --> AGENTS
    EVENTS --> SERVICES
```

---

## Current vs Target Architecture

### Current State (Problematic)

```mermaid
flowchart TB
    subgraph CURRENT["Current Architecture Issues"]
        subgraph ORCH_MESS["Orchestration Chaos"]
            O1[agent_orchestrator.py<br/>2,382 lines]
            O2[enterprise_orchestrator.py<br/>BROKEN]
            O3[optimized_orchestrator.py<br/>DUPLICATE]
            O4[diagnosis/orchestrator.py]
        end

        subgraph MEM_MESS["Memory Fragmentation"]
            M1[enhanced_memory.py]
            M2[semantic_memory.py]
            M3[context_aware.py]
            M4[conversation.py]
            M5[utils/memory_*.py]
        end

        subgraph ENT_MESS["Enterprise Chaos"]
            E1[enterprise/<br/>DEAD CODE]
            E2[src/enterprise/<br/>BROKEN]
            E3[diagnosis/enterprise/<br/>DUPLICATE]
        end
    end

    style O1 fill:#ff6b6b
    style O2 fill:#ff6b6b
    style O3 fill:#ffd93d
    style E1 fill:#ff6b6b
    style E2 fill:#ff6b6b
    style E3 fill:#ffd93d
```

### Target State (Clean)

```mermaid
flowchart TB
    subgraph TARGET["Target Architecture"]
        subgraph ORCH_CLEAN["Orchestration (1 file)"]
            TO[orchestrator.py<br/>~500 lines]
        end

        subgraph MEM_CLEAN["Memory (3 files)"]
            TM1[memory_system.py]
            TM2[semantic.py]
            TM3[factory.py]
        end

        subgraph NO_ENT["No Enterprise Folder"]
            TE[Features integrated<br/>into core modules]
        end
    end

    style TO fill:#6bcb77
    style TM1 fill:#6bcb77
    style TM2 fill:#6bcb77
    style TM3 fill:#6bcb77
    style TE fill:#6bcb77
```

---

## Module Communication Patterns

```mermaid
flowchart LR
    subgraph SYNC["Synchronous Communication"]
        A1[Agent] -->|Direct Call| S1[Service]
        S1 -->|Query| D1[(Database)]
    end

    subgraph ASYNC["Asynchronous Communication"]
        A2[Agent] -->|Publish| EB[Event Bus]
        EB -->|Subscribe| S2[Service]
        EB -->|Subscribe| M2[Monitor]
    end

    subgraph CALLBACK["Callback Pattern"]
        A3[Agent] -->|Register| CB[Callback Registry]
        CB -->|Notify| H[Handler]
    end
```

---

*Diagrams created for Solace-AI codebase visualization*
*Use Mermaid-compatible markdown viewer to render*
