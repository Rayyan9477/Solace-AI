# Solace-AI Diagnosis Module - Master Architecture Diagrams

> **Version**: 2.0  
> **Date**: December 30, 2025  
> **Purpose**: Visual Reference for Diagnosis Module Architecture

---

## Quick Reference

| Diagram | Description |
|---------|-------------|
| [1. System Architecture](#1-complete-system-architecture-overview) | High-level system overview |
| [2. Multi-Agent Flow](#2-multi-agent-orchestration-flow) | Agent orchestration patterns |
| [3. Chain-of-Reasoning](#3-chain-of-reasoning-pipeline) | 4-step reasoning process |
| [4. Phase State Machine](#4-dialogue-phase-state-machine) | Interview phase transitions |
| [5. Clinical Knowledge](#5-clinical-knowledge-integration) | DSM-5/HiTOP integration |
| [6. Memory Architecture](#6-memory-architecture) | 4-tier memory system |
| [7. Safety Detection](#7-safety--crisis-detection) | 3-layer crisis detection |
| [8. Anti-Sycophancy](#8-anti-sycophancy-framework) | Bias prevention |
| [9. Confidence Calibration](#9-confidence-calibration) | Scoring methodology |
| [10. Data Flow](#10-data-flow-overview) | End-to-end processing |
| [11. Module Integration](#11-module-integration) | Cross-module communication |
| [12. Event Architecture](#12-event-driven-communication) | Event-driven patterns |
| [13. Deployment](#13-deployment-architecture) | Container architecture |
| [14. System Context](#14-system-context) | External integrations |

---

## 1. Complete System Architecture Overview

```mermaid
flowchart TB
    subgraph CLIENT["👤 Client Layer"]
        WEB[Web App]
        MOBILE[Mobile App]
        VOICE[Voice Interface]
    end

    subgraph GATEWAY["🚪 API Gateway"]
        AUTH[Authentication]
        RATE[Rate Limiting]
        VALID[Validation]
    end

    subgraph SAFETY_LAYER["🛡️ Safety Layer (Always First)"]
        direction LR
        L1["Layer 1<br/>Keyword Detection<br/><10ms"]
        L2["Layer 2<br/>Semantic Analysis<br/><100ms"]
        L3["Layer 3<br/>Pattern Recognition<br/><500ms"]
        L1 --> L2 --> L3
    end

    subgraph DIAGNOSIS_ENGINE["💡 Diagnosis Engine (Core)"]
        direction TB
        
        subgraph AGENTS["Multi-Agent System"]
            direction LR
            DA["💬 Dialogue<br/>Agent"]
            IA["🔍 Insight<br/>Agent"]
            AA["⚖️ Advocate<br/>Agent"]
            RA["✅ Reviewer<br/>Agent"]
        end
        
        subgraph REASONING["Chain-of-Reasoning Pipeline"]
            direction LR
            R1["1. Analyze"]
            R2["2. Hypothesize"]
            R3["3. Challenge"]
            R4["4. Synthesize"]
            R1 --> R2 --> R3 --> R4
        end
        
        subgraph STATE["State Machine"]
            direction LR
            P1["Rapport"]
            P2["History"]
            P3["Assessment"]
            P4["Diagnosis"]
            P5["Closure"]
            P1 --> P2 --> P3 --> P4 --> P5
        end
    end

    subgraph KNOWLEDGE["📚 Knowledge Layer"]
        DSM5[(DSM-5-TR<br/>Criteria)]
        HITOP[(HiTOP<br/>Dimensions)]
        SCREENS[(Validated<br/>Instruments)]
    end

    subgraph MEMORY["💾 Memory Layer"]
        direction LR
        WORKING["Working<br/>Memory"]
        SESSION["Session<br/>Memory"]
        EPISODIC["Episodic<br/>Memory"]
        SEMANTIC["Semantic<br/>Memory"]
    end

    subgraph OUTPUT["📤 Output Layer"]
        RESPONSE[Response Generation]
        INSIGHTS[Insight Reports]
        ALERTS[Clinical Alerts]
    end

    CLIENT --> GATEWAY
    GATEWAY --> SAFETY_LAYER
    SAFETY_LAYER -->|Safe| DIAGNOSIS_ENGINE
    SAFETY_LAYER -->|Crisis| CRISIS[🚨 Crisis Protocol]
    KNOWLEDGE --> DIAGNOSIS_ENGINE
    MEMORY <--> DIAGNOSIS_ENGINE
    DIAGNOSIS_ENGINE --> OUTPUT
    OUTPUT --> CLIENT

    style SAFETY_LAYER fill:#ffcdd2,stroke:#c62828
    style DIAGNOSIS_ENGINE fill:#e3f2fd,stroke:#1565c0
    style MEMORY fill:#e8f5e9,stroke:#2e7d32
    style CRISIS fill:#ff6b6b,color:#fff
```

## 2. Multi-Agent Orchestration Flow

```mermaid
flowchart TB
    INPUT[/"User Message"/] --> ORCH

    subgraph ORCH["🎭 Agent Orchestrator"]
        COORD[Coordinator]
        ROUTER[Task Router]
    end

    COORD --> ROUTER

    ROUTER --> DA
    ROUTER --> IA
    
    subgraph DA["💬 Dialogue Agent"]
        DA_T["• History taking<br/>• Question generation<br/>• Empathetic responses"]
    end

    subgraph IA["🔍 Insight Agent"]
        IA_T["• Symptom mapping<br/>• Differential generation<br/>• Severity assessment"]
    end

    DA --> AGG
    IA --> AGG

    AGG[Aggregator] --> AA

    subgraph AA["⚖️ Advocate Agent (Anti-Sycophancy)"]
        AA_T["• Challenge diagnoses<br/>• Propose alternatives<br/>• Identify biases"]
    end

    AA --> RA

    subgraph RA["✅ Reviewer Agent"]
        RA_T["• Synthesize perspectives<br/>• Calibrate confidence<br/>• Validate safety"]
    end

    RA --> OUTPUT[/"Final Response + Insights"/]

    SA["🚨 Safety Agent<br/>(Always Monitoring)"] -.->|Priority Override| OUTPUT

    style AA fill:#fff3e0,stroke:#ef6c00
    style SA fill:#ffcdd2,stroke:#c62828
```

## 3. Chain-of-Reasoning Pipeline

```mermaid
flowchart LR
    subgraph STEP1["Step 1: ANALYZE"]
        S1["Extract symptoms<br/>Identify +/- findings<br/>Note timeline<br/>Update profile"]
    end

    subgraph STEP2["Step 2: HYPOTHESIZE"]
        S2["Generate hypotheses<br/>Map to DSM-5-TR<br/>Rank by likelihood<br/>Identify gaps"]
    end

    subgraph STEP3["Step 3: CHALLENGE"]
        S3["Devil's advocate<br/>Alternative explanations<br/>Detect biases<br/>Counter-questions"]
    end

    subgraph STEP4["Step 4: SYNTHESIZE"]
        S4["Integrate perspectives<br/>Calibrate confidence<br/>Generate response<br/>Update state"]
    end

    INPUT[/"Context + Message"/] --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> OUTPUT[/"Response + DDx"/]

    style STEP1 fill:#e3f2fd
    style STEP2 fill:#e8f5e9
    style STEP3 fill:#fff3e0
    style STEP4 fill:#f3e5f5
```

## 4. Dialogue Phase State Machine

```mermaid
stateDiagram-v2
    [*] --> Rapport: Session Start
    
    Rapport --> HistoryTaking: Trust Established
    HistoryTaking --> Assessment: History Complete
    Assessment --> Diagnosis: Assessment Complete
    Diagnosis --> Closure: Insights Ready
    Closure --> [*]: Session End
    
    Rapport --> Crisis: 🚨
    HistoryTaking --> Crisis: 🚨
    Assessment --> Crisis: 🚨
    Diagnosis --> Crisis: 🚨
    
    Crisis --> [*]: Escalated

    note right of Rapport
        Build therapeutic alliance
        Initial safety screen
    end note
    
    note right of HistoryTaking
        Chief complaint
        Present illness
        Past/Family/Social history
    end note
    
    note right of Assessment
        Symptom exploration
        Severity assessment
        Functional impact
    end note
    
    note right of Diagnosis
        Differential generation
        Confidence calibration
        Insight formulation
    end note
```

## 5. Clinical Knowledge Integration

```mermaid
flowchart TB
    subgraph INPUT_SYM["Collected Symptoms"]
        S1[Mood symptoms]
        S2[Anxiety symptoms]
        S3[Cognitive symptoms]
        S4[Behavioral symptoms]
        S5[Somatic symptoms]
    end

    subgraph MAPPING["Mapping Engine"]
        direction TB
        PATTERN[Pattern Matching]
        CRITERIA[Criteria Evaluation]
        DIMENSIONAL[Dimensional Scoring]
    end

    subgraph KNOWLEDGE["Clinical Knowledge"]
        DSM["DSM-5-TR<br/>Categorical"]
        HITOP["HiTOP<br/>Dimensional"]
        SCREENS["PHQ-9, GAD-7<br/>PCL-5, etc."]
    end

    subgraph OUTPUT_DX["Diagnostic Output"]
        DDX["Differential List<br/>{diagnosis, confidence}"]
        DIM["Dimensional Profile<br/>{spectrum: score}"]
        SEV["Severity Rating<br/>{mild/moderate/severe}"]
        MISSING["Missing Information<br/>[questions needed]"]
    end

    INPUT_SYM --> MAPPING
    KNOWLEDGE --> MAPPING
    MAPPING --> OUTPUT_DX
```

## 6. Memory Architecture

```mermaid
flowchart TB
    subgraph TIER1["Tier 1: Working Memory"]
        T1["Current session<br/>Active profile<br/>~8K tokens"]
    end

    subgraph TIER2["Tier 2: Session Memory"]
        T2["Full history<br/>Emotional trajectory<br/>Redis cache"]
    end

    subgraph TIER3["Tier 3: Episodic Memory"]
        T3["Past sessions<br/>Key insights<br/>Vector store"]
    end

    subgraph TIER4["Tier 4: Semantic Memory"]
        T4["Persistent profile<br/>Longitudinal trends<br/>Structured DB"]
    end

    TIER1 <-->|"Overflow"| TIER2
    TIER2 <-->|"Consolidate"| TIER3
    TIER3 <-->|"Persist"| TIER4

    RETRIEVAL["Hybrid Retrieval<br/>Semantic + Keyword"]
    
    TIER3 --> RETRIEVAL
    TIER4 --> RETRIEVAL
    RETRIEVAL --> CONTEXT["Enriched Context"]

    style TIER1 fill:#ffebee
    style TIER2 fill:#e3f2fd
    style TIER3 fill:#e8f5e9
    style TIER4 fill:#fff3e0
```

## 7. Safety & Crisis Detection

```mermaid
flowchart TB
    MSG[/"User Message"/] --> L1

    subgraph DETECTION["Three-Layer Detection"]
        L1["🔴 Layer 1: Keywords<br/><10ms latency"]
        L2["🟠 Layer 2: Semantics<br/><100ms latency"]
        L3["🟡 Layer 3: Patterns<br/><500ms latency"]
        
        L1 --> L2 --> L3
    end

    L3 --> SCORE["Risk Score<br/>Aggregation"]

    SCORE --> ROUTE{Risk Level?}

    ROUTE -->|"< 0.2"| NORMAL["Continue Normal"]
    ROUTE -->|"0.2-0.4"| CHECK["Gentle Check-in"]
    ROUTE -->|"0.4-0.6"| INQUIRY["Direct Safety Inquiry"]
    ROUTE -->|"0.6-0.8"| PLAN["Safety Planning"]
    ROUTE -->|"> 0.8"| CRISIS["🚨 Crisis Protocol"]

    CRISIS --> RESOURCES["Crisis Resources<br/>988 Lifeline<br/>Crisis Text Line"]
    CRISIS --> HANDOFF["Human Handoff<br/>(if available)"]

    style L1 fill:#ffcdd2
    style L2 fill:#ffe0b2
    style L3 fill:#fff9c4
    style CRISIS fill:#ff6b6b,color:#fff
```

## 8. Anti-Sycophancy Framework

```mermaid
flowchart TB
    subgraph PROBLEM["❌ Sycophancy Problem"]
        P1["LLM agrees with user"]
        P2["Confirms self-diagnosis"]
        P3["Avoids disagreement"]
        P4["Up to 100% compliance with<br/>incorrect medical claims"]
    end

    subgraph SOLUTION["✅ Anti-Sycophancy Solution"]
        S1["Explicit rejection<br/>permission in prompts"]
        S2["Devil's Advocate Agent<br/>challenges every assessment"]
        S3["Alternative hypothesis<br/>requirement (≥2)"]
        S4["Sample consistency<br/>confidence scoring"]
    end

    subgraph OUTCOME["Honest Assessment"]
        O1["Accurate diagnoses"]
        O2["Calibrated confidence"]
        O3["Acknowledged uncertainty"]
        O4["Multiple perspectives"]
    end

    PROBLEM --> SOLUTION --> OUTCOME

    style PROBLEM fill:#ffcdd2
    style SOLUTION fill:#e8f5e9
    style OUTCOME fill:#e3f2fd
```

## 9. Confidence Calibration

```mermaid
flowchart TB
    QUERY["Assessment Query"] --> SAMPLING

    subgraph SAMPLING["Multi-Sample Generation"]
        S1["Sample 1"]
        S2["Sample 2"]
        S3["Sample 3"]
        S4["Sample 4"]
        S5["Sample 5"]
    end

    SAMPLING --> COMPARE["Compare Responses"]

    COMPARE --> AGREEMENT{Agreement<br/>Level?}

    AGREEMENT -->|">80%"| HIGH["High Confidence<br/>Proceed with insight"]
    AGREEMENT -->|"50-80%"| MED["Moderate Confidence<br/>Note uncertainty"]
    AGREEMENT -->|"<50%"| LOW["Low Confidence<br/>Request more info<br/>or escalate"]

    subgraph COMPONENTS["Confidence Components"]
        C1["Criteria coverage: 30%"]
        C2["Sample consistency: 25%"]
        C3["Info completeness: 20%"]
        C4["Temporal stability: 15%"]
        C5["Instrument alignment: 10%"]
    end

    COMPONENTS --> FINAL["Final Calibrated Score"]

    style HIGH fill:#e8f5e9
    style MED fill:#fff3e0
    style LOW fill:#ffcdd2
```

## 10. Data Flow Overview

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        I1[Text]
        I2[Voice]
        I3[History]
    end

    subgraph PROCESS["Processing"]
        P1[Safety Check]
        P2[Context Load]
        P3[Agent Processing]
        P4[Reasoning Chain]
        P5[Calibration]
    end

    subgraph STATE["State Updates"]
        S1[Patient Profile]
        S2[Differential]
        S3[Phase State]
        S4[Memory]
    end

    subgraph OUTPUT["Output"]
        O1[Response]
        O2[Insights]
        O3[Alerts]
        O4[Events]
    end

    INPUT --> PROCESS
    PROCESS --> STATE
    STATE --> OUTPUT
    STATE -->|Feedback| PROCESS
```

## 11. Module Integration

```mermaid
flowchart TB
    subgraph DIAGNOSIS["🔍 Diagnosis Module"]
        D_CORE["Core Engine"]
    end

    subgraph THERAPY["💆 Therapy Module"]
        T_CORE["Technique Selection"]
    end

    subgraph PERSONALITY["🎭 Personality Module"]
        P_CORE["Style Adaptation"]
    end

    subgraph MEMORY["🧠 Memory Module"]
        M_CORE["Context Storage"]
    end

    subgraph RESPONSE["💬 Response Module"]
        R_CORE["Output Generation"]
    end

    subgraph SAFETY["🛡️ Safety Module"]
        S_CORE["Crisis Detection"]
    end

    P_CORE -->|"Communication Style"| D_CORE
    M_CORE -->|"Historical Context"| D_CORE
    D_CORE -->|"Assessment"| T_CORE
    D_CORE -->|"Insights"| R_CORE
    D_CORE -->|"New Data"| M_CORE
    T_CORE -->|"Techniques"| R_CORE
    P_CORE -->|"Tone"| R_CORE
    S_CORE <-->|"Safety Checks"| D_CORE

    style DIAGNOSIS fill:#4ecdc4,color:#fff
    style SAFETY fill:#ff6b6b,color:#fff
```

## 12. Event-Driven Communication

```mermaid
flowchart TB
    subgraph PUBLISHERS["Publishers"]
        PUB1["Diagnosis Service"]
        PUB2["Safety Service"]
        PUB3["Session Manager"]
    end

    subgraph EVENTS["Event Bus"]
        E1["SessionStarted"]
        E2["SymptomDetected"]
        E3["DiagnosisUpdated"]
        E4["InsightGenerated"]
        E5["CrisisDetected"]
        E6["SessionEnded"]
    end

    subgraph SUBSCRIBERS["Subscribers"]
        SUB1["Memory Persistence"]
        SUB2["Analytics Handler"]
        SUB3["Alert Handler"]
        SUB4["Audit Logger"]
        SUB5["Therapy Module"]
    end

    PUB1 --> E1 & E2 & E3 & E4
    PUB2 --> E5
    PUB3 --> E6

    E1 --> SUB1 & SUB4
    E2 --> SUB1 & SUB5
    E3 --> SUB1 & SUB2
    E4 --> SUB1 & SUB2
    E5 --> SUB3 & SUB4
    E6 --> SUB1 & SUB2 & SUB4
```

---

## Key Architecture Decisions Summary

| Decision | Pattern Chosen | Rationale |
|----------|---------------|-----------|
| **Agent Architecture** | Multi-Agent with Orchestrator | Inspired by AMIE; enables specialized processing |
| **Reasoning Strategy** | 4-Step Chain-of-Reasoning | Ensures thorough analysis with challenge step |
| **Anti-Sycophancy** | Devil's Advocate Agent | Prevents confirmation bias in assessments |
| **Safety Detection** | 3-Layer Progressive | Balances speed with accuracy |
| **Confidence Scoring** | Sample Consistency | More reliable than verbalized confidence |
| **Clinical Framework** | DSM-5-TR + HiTOP Hybrid | Combines categorical with dimensional |
| **Memory System** | 4-Tier Hierarchy | Optimizes for both speed and persistence |
| **Communication** | Event-Driven + Sync API | Decouples modules while ensuring responsiveness |
| **State Management** | Phase-Based State Machine | Structures clinical interview naturally |

---

## 13. Deployment Architecture

```mermaid
flowchart TB
    subgraph KUBERNETES["Kubernetes Cluster"]
        direction TB
        
        subgraph INGRESS["Ingress Layer"]
            IG["Nginx Ingress"]
            SSL["SSL Termination"]
        end
        
        subgraph SERVICES["Service Pods"]
            subgraph DIAGNOSIS_SVC["Diagnosis Service"]
                D_POD1["Pod 1"]
                D_POD2["Pod 2"]
                D_POD3["Pod 3"]
            end
            
            subgraph SAFETY_SVC["Safety Service"]
                S_POD1["Pod 1"]
                S_POD2["Pod 2"]
            end
            
            subgraph MEMORY_SVC["Memory Service"]
                M_POD1["Pod 1"]
                M_POD2["Pod 2"]
            end
        end
        
        subgraph DATA["Data Layer"]
            REDIS[("Redis Cluster")]
            POSTGRES[("PostgreSQL")]
            CHROMA[("ChromaDB")]
        end
        
        subgraph MONITORING["Monitoring"]
            PROM["Prometheus"]
            GRAF["Grafana"]
            ALERT["AlertManager"]
        end
    end

    subgraph EXTERNAL_DEPS["External Dependencies"]
        LLM_API["LLM API"]
        EMBED_API["Embedding API"]
    end

    INGRESS --> SERVICES
    SERVICES --> DATA
    SERVICES --> EXTERNAL_DEPS
    SERVICES --> MONITORING
```

---

## 14. System Context

```mermaid
flowchart TB
    subgraph USERS["Users"]
        USER["User<br/>Person seeking support"]
        CLINICIAN["Clinician<br/>Mental health professional"]
    end

    subgraph SOLACE["Solace-AI System"]
        CORE["Solace-AI<br/>Mental Health AI Companion"]
    end

    subgraph EXTERNAL["External Systems"]
        LLM["LLM Provider"]
        CRISIS["Crisis Services"]
        EHR["EHR Systems"]
    end

    USER -->|"Text/Voice"| CORE
    CORE -->|"Insights"| USER
    CLINICIAN -->|"Oversight"| CORE
    CORE -->|"API"| LLM
    CORE -->|"Escalation"| CRISIS
    CORE -.->|"FHIR"| EHR

    style SOLACE fill:#4ecdc4,color:#fff
    style CRISIS fill:#ff6b6b,color:#fff
```

---

## Cross-Reference

For detailed explanations of each component, refer to:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete technical blueprint

---

*Generated for Solace-AI Diagnosis Module v2.0*  
*Last Updated: December 30, 2025*
