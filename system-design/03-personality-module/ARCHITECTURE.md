# Solace-AI: Personality Detection Module
## Complete System Architecture & Design

> **Version**: 2.0  
> **Date**: December 30, 2025  
> **Author**: System Architecture Team  
> **Status**: Technical Blueprint  
> **Integration**: Diagnosis Module, Therapy Module, Memory Module

---

## Executive Summary

This document presents the complete architecture for the Personality Detection Module of Solace-AI. The module detects user personality traits using the scientifically-validated Big Five (OCEAN) model, adapts communication style accordingly, and generates empathic responses tailored to individual psychological profiles.

### Key Architecture Decisions

| Decision | Pattern | Rationale |
|----------|---------|-----------|
| **Personality Model** | Big Five (OCEAN) Continuous | Test-retest reliability >0.80 vs MBTI's 0.24-0.61 |
| **Detection Method** | Ensemble (Fine-tuned RoBERTa + LLM) | Accuracy + interpretability balance |
| **Trait Scoring** | Continuous (0.0-1.0) | Granular personalization, clinical alignment |
| **Multimodal Fusion** | Late Fusion with Attention | Handles missing modalities gracefully |
| **Empathy Model** | MoEL (Mixture of Empathetic Listeners) | State-of-the-art empathic generation |
| **Adaptation Strategy** | Personality-Aware Prompting | r>0.85 correlation with trait levels |

---

## Table of Contents

1. [Architecture Philosophy](#1-architecture-philosophy)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Component Architecture](#3-component-architecture)
4. [Big Five Detection Engine](#4-big-five-detection-engine)
5. [Multimodal Personality Analysis](#5-multimodal-personality-analysis)
6. [Personality Profile Management](#6-personality-profile-management)
7. [Response Adaptation Engine](#7-response-adaptation-engine)
8. [Empathic Response Generation](#8-empathic-response-generation)
9. [Cultural Adaptation System](#9-cultural-adaptation-system)
10. [Data Flow Architecture](#10-data-flow-architecture)
11. [Integration Interfaces](#11-integration-interfaces)
12. [Event-Driven Architecture](#12-event-driven-architecture)

---

## 1. Architecture Philosophy

### 1.1 Core Design Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PERSONALITY MODULE DESIGN PRINCIPLES                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │ SCIENTIFIC  │   │ CONTINUOUS  │   │  ADAPTIVE   │   │  EMPATHIC   │    │
│   │  VALIDITY   │   │   SCORING   │   │   COMMS     │   │  RESPONSE   │    │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│          │                 │                 │                 │            │
│          ▼                 ▼                 ▼                 ▼            │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │ Big Five    │   │ 0.0-1.0     │   │ Style match │   │ Three-layer │    │
│   │ OCEAN model │   │ granular    │   │ to user     │   │ empathy:    │    │
│   │ validated   │   │ traits with │   │ personality │   │ cognitive,  │    │
│   │ over MBTI   │   │ confidence  │   │ profile     │   │ affective,  │    │
│   │             │   │ intervals   │   │             │   │ compassion  │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                                              │
│   WHY BIG FIVE OVER MBTI:                                                   │
│   • Test-retest reliability: Big Five >0.80 vs MBTI 0.24-0.61              │
│   • Includes Neuroticism (critical for mental health)                       │
│   • Continuous traits enable progress tracking                              │
│   • DSM-5 dimensional model alignment                                       │
│   • GPT-4 achieves r=0.29-0.38 correlation with self-reported traits       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Big Five (OCEAN) Model

```mermaid
flowchart TB
    subgraph OCEAN["Big Five Personality Traits"]
        direction TB
        
        subgraph O["🔮 OPENNESS"]
            O_DESC["Intellectual curiosity, creativity"]
            O_HIGH["High: Curious, imaginative, open to new experiences"]
            O_LOW["Low: Practical, conventional, prefer routine"]
            O_THERAPY["Therapy: Exploratory techniques, metaphors, ACT"]
        end
        
        subgraph C["📋 CONSCIENTIOUSNESS"]
            C_DESC["Organization, dependability, self-discipline"]
            C_HIGH["High: Organized, reliable, goal-oriented"]
            C_LOW["Low: Flexible, spontaneous, less structured"]
            C_THERAPY["Therapy: Structured homework, clear timelines"]
        end
        
        subgraph E["🎭 EXTRAVERSION"]
            E_DESC["Sociability, assertiveness, positive emotions"]
            E_HIGH["High: Outgoing, energetic, talkative"]
            E_LOW["Low: Reserved, introspective, prefer solitude"]
            E_THERAPY["Therapy: Adjust session energy, social exposure"]
        end
        
        subgraph A["🤝 AGREEABLENESS"]
            A_DESC["Cooperation, trust, empathy toward others"]
            A_HIGH["High: Cooperative, trusting, helpful"]
            A_LOW["Low: Competitive, skeptical, challenging"]
            A_THERAPY["Therapy: Validation level, directness of feedback"]
        end
        
        subgraph N["💭 NEUROTICISM"]
            N_DESC["Emotional instability, anxiety, moodiness"]
            N_HIGH["High: Anxious, moody, emotionally reactive"]
            N_LOW["Low: Calm, emotionally stable, resilient"]
            N_THERAPY["Therapy: Safety focus, grounding, reassurance"]
        end
    end

    style O fill:#e3f2fd,stroke:#1565c0
    style C fill:#e8f5e9,stroke:#2e7d32
    style E fill:#fff3e0,stroke:#ef6c00
    style A fill:#fce4ec,stroke:#c2185b
    style N fill:#f3e5f5,stroke:#7b1fa2
```

---

## 2. High-Level System Architecture

### 2.1 Complete Module Overview

```mermaid
flowchart TB
    subgraph INPUT_LAYER["🎯 INPUT LAYER"]
        direction LR
        I1[/"User Text"/]
        I2[/"Voice Audio"/]
        I3[/"Behavioral Signals"/]
        I4[/"Session History"/]
    end

    subgraph DETECTION_ENGINE["🔍 PERSONALITY DETECTION ENGINE"]
        direction TB
        
        subgraph TEXT_ANALYSIS["Text Analysis"]
            TA1["BERT Embeddings"]
            TA2["LIWC Features"]
            TA3["Fine-tuned RoBERTa"]
        end
        
        subgraph VOICE_ANALYSIS["Voice Analysis"]
            VA1["Prosodic Features"]
            VA2["Speech Patterns"]
            VA3["Emotional Tone"]
        end
        
        subgraph BEHAVIORAL["Behavioral Analysis"]
            BA1["Response Patterns"]
            BA2["Engagement Metrics"]
            BA3["Temporal Patterns"]
        end
        
        subgraph FUSION["Multimodal Fusion"]
            FU1["Late Fusion"]
            FU2["Attention Weighting"]
            FU3["Confidence Scoring"]
        end
    end

    subgraph PROFILE_MGR["📊 PROFILE MANAGER"]
        direction TB
        PM1["Trait Aggregation"]
        PM2["Temporal Tracking"]
        PM3["Confidence Calibration"]
        PM4["Profile Versioning"]
    end

    subgraph ADAPTATION_ENGINE["🎨 ADAPTATION ENGINE"]
        direction TB
        
        subgraph STYLE["Style Adaptation"]
            ST1["Communication Style"]
            ST2["Tone Selection"]
            ST3["Complexity Level"]
        end
        
        subgraph EMPATHY["Empathy Generation"]
            EM1["Emotion Detection"]
            EM2["MoEL Listeners"]
            EM3["Response Assembly"]
        end
    end

    subgraph OUTPUT_LAYER["📤 OUTPUT"]
        direction TB
        O1[/"Personality Profile"/]
        O2[/"Adapted Response Style"/]
        O3[/"Empathic Elements"/]
    end

    INPUT_LAYER --> DETECTION_ENGINE
    TEXT_ANALYSIS --> FUSION
    VOICE_ANALYSIS --> FUSION
    BEHAVIORAL --> FUSION
    FUSION --> PROFILE_MGR
    PROFILE_MGR --> ADAPTATION_ENGINE
    ADAPTATION_ENGINE --> OUTPUT_LAYER

    style DETECTION_ENGINE fill:#e3f2fd,stroke:#1565c0
    style ADAPTATION_ENGINE fill:#e8f5e9,stroke:#2e7d32
```

### 2.2 System Context

```mermaid
flowchart TB
    subgraph EXTERNAL["External Actors"]
        USER["👤 User"]
        CLINICIAN["👨‍⚕️ Clinician"]
    end

    subgraph SOLACE["Solace-AI Platform"]
        subgraph MODULES["Core Modules"]
            DIAG["🔍 Diagnosis Module"]
            THERAPY["💆 Therapy Module"]
            PERSONALITY["🎭 Personality Module"]
            MEMORY["🧠 Memory Module"]
            RESPONSE["💬 Response Module"]
        end
    end

    subgraph INFRA["Infrastructure"]
        LLM["LLM Provider"]
        ML["ML Models<br/>(RoBERTa, BERT)"]
        VECTOR[(Vector Store)]
    end

    USER -->|"Conversation"| PERSONALITY
    PERSONALITY -->|"Profile"| THERAPY
    PERSONALITY -->|"Profile"| DIAG
    PERSONALITY -->|"Style"| RESPONSE
    PERSONALITY <-->|"History"| MEMORY
    PERSONALITY --> LLM
    PERSONALITY --> ML
    PERSONALITY --> VECTOR
    CLINICIAN -->|"Review"| PERSONALITY

    style PERSONALITY fill:#f3e5f5,stroke:#7b1fa2
```

---

## 3. Component Architecture

### 3.1 Clean Architecture Layers

```mermaid
flowchart TB
    subgraph PRESENTATION["📺 Presentation Layer"]
        direction LR
        P1["Personality API"]
        P2["Profile Endpoints"]
        P3["Event Publishers"]
    end

    subgraph APPLICATION["⚙️ Application Layer"]
        direction TB
        
        subgraph USE_CASES["Use Cases"]
            UC1["DetectPersonality"]
            UC2["UpdateProfile"]
            UC3["AdaptResponse"]
            UC4["GenerateEmpathy"]
            UC5["GetStyleRecommendation"]
        end
        
        subgraph APP_SERVICES["Services"]
            AS1["PersonalityOrchestrator"]
            AS2["ProfileManager"]
            AS3["AdaptationService"]
            AS4["EmpathyService"]
        end
    end

    subgraph DOMAIN["🎯 Domain Layer"]
        direction TB
        
        subgraph ENTITIES["Entities"]
            E1["PersonalityProfile"]
            E2["TraitAssessment"]
            E3["StylePreference"]
            E4["EmpathyContext"]
        end
        
        subgraph VALUE_OBJ["Value Objects"]
            VO1["OceanScore"]
            VO2["ConfidenceInterval"]
            VO3["CommunicationStyle"]
            VO4["EmotionState"]
        end
        
        subgraph DOMAIN_SVC["Domain Services"]
            DS1["TraitDetectionService"]
            DS2["ProfileAggregationService"]
            DS3["StyleMappingService"]
        end
    end

    subgraph INFRASTRUCTURE["🏗️ Infrastructure Layer"]
        direction TB
        
        subgraph ADAPTERS["Adapters"]
            AD1["RoBERTaAdapter"]
            AD2["LLMAdapter"]
            AD3["VoiceAnalyzerAdapter"]
        end
        
        subgraph REPOS["Repositories"]
            RP1["ProfileRepository"]
            RP2["AssessmentRepository"]
            RP3["StyleRepository"]
        end
    end

    PRESENTATION --> APPLICATION
    APPLICATION --> DOMAIN
    APPLICATION --> INFRASTRUCTURE
    INFRASTRUCTURE -.-> DOMAIN

    style DOMAIN fill:#f3e5f5,stroke:#7b1fa2
    style APPLICATION fill:#e3f2fd,stroke:#1565c0
```

---

## 4. Big Five Detection Engine

### 4.1 Detection Pipeline Architecture

```mermaid
flowchart TB
    subgraph INPUT["User Input"]
        I1["Text Messages"]
        I2["Voice Samples"]
        I3["Behavioral Data"]
    end

    subgraph FEATURE_EXTRACTION["Feature Extraction Layer"]
        direction TB
        
        subgraph TEXT_FEAT["Text Features"]
            TF1["BERT Embeddings<br/>(768 dimensions)"]
            TF2["LIWC Categories<br/>(93 categories)"]
            TF3["Sentiment Scores"]
            TF4["Linguistic Markers"]
        end
        
        subgraph VOICE_FEAT["Voice Features"]
            VF1["MFCCs (13 coefficients)"]
            VF2["Pitch Variability"]
            VF3["Speech Rate"]
            VF4["Pause Patterns"]
        end
        
        subgraph BEHAV_FEAT["Behavioral Features"]
            BF1["Response Latency"]
            BF2["Message Length Patterns"]
            BF3["Session Engagement"]
            BF4["Temporal Patterns"]
        end
    end

    subgraph MODEL_ENSEMBLE["Model Ensemble"]
        direction TB
        
        M1["Fine-tuned RoBERTa Large<br/>(Primary: R²=0.24)"]
        M2["Zero-shot LLM<br/>(Validation: r=0.29-0.38)"]
        M3["Voice Personality Model<br/>(CNN-BiLSTM)"]
        
        M1 --> ENSEMBLE["Ensemble Aggregator"]
        M2 --> ENSEMBLE
        M3 --> ENSEMBLE
    end

    subgraph OUTPUT["Personality Output"]
        direction TB
        O1["OCEAN Scores (0.0-1.0)"]
        O2["Confidence Intervals"]
        O3["Evidence Markers"]
    end

    INPUT --> FEATURE_EXTRACTION
    TEXT_FEAT --> M1
    TEXT_FEAT --> M2
    VOICE_FEAT --> M3
    BEHAV_FEAT --> ENSEMBLE
    ENSEMBLE --> OUTPUT

    style MODEL_ENSEMBLE fill:#e3f2fd,stroke:#1565c0
```

### 4.2 Trait Detection Flow

```mermaid
flowchart TB
    subgraph DETECTION_FLOW["Trait Detection Flow"]
        direction TB
        
        INPUT["User Message"] --> PREPROCESS["Preprocessing"]
        
        PREPROCESS --> PARALLEL["Parallel Processing"]
        
        subgraph PARALLEL["Parallel Analysis"]
            direction LR
            
            subgraph ROBERTA["RoBERTa Path"]
                R1["Tokenize"]
                R2["Encode"]
                R3["Classify Traits"]
            end
            
            subgraph LLM["LLM Path"]
                L1["Prompt Construction"]
                L2["Zero-shot Analysis"]
                L3["Parse Response"]
            end
            
            subgraph LIWC["LIWC Path"]
                W1["Word Categorization"]
                W2["Feature Computation"]
                W3["Trait Correlation"]
            end
        end
        
        ROBERTA --> AGGREGATE["Weighted Aggregation"]
        LLM --> AGGREGATE
        LIWC --> AGGREGATE
        
        AGGREGATE --> CALIBRATE["Confidence Calibration"]
        
        CALIBRATE --> SCORES["Final OCEAN Scores"]
    end

    subgraph SCORES["Output Scores"]
        direction LR
        SC_O["O: 0.72 ± 0.08"]
        SC_C["C: 0.45 ± 0.12"]
        SC_E["E: 0.38 ± 0.10"]
        SC_A["A: 0.81 ± 0.06"]
        SC_N["N: 0.56 ± 0.09"]
    end

    style PARALLEL fill:#e8f5e9
```

### 4.3 LIWC Feature Mapping to Big Five

```mermaid
flowchart LR
    subgraph LIWC_FEATURES["LIWC Categories"]
        direction TB
        L1["Word Count"]
        L2["I-words (I, me, my)"]
        L3["Social words"]
        L4["Positive emotion"]
        L5["Negative emotion"]
        L6["Cognitive processes"]
        L7["Achievement words"]
        L8["Tentative words"]
    end

    subgraph MAPPING["Feature → Trait Mapping"]
        direction TB
        
        M1["Insight words, Question marks<br/>→ Openness (+)"]
        M2["Achievement, Work words<br/>→ Conscientiousness (+)"]
        M3["Social, Inclusive words<br/>→ Extraversion (+)"]
        M4["Affiliation, Positive emotion<br/>→ Agreeableness (+)"]
        M5["Negative emotion, Anxiety words<br/>→ Neuroticism (+)"]
    end

    subgraph TRAITS["Big Five Traits"]
        direction TB
        T_O["Openness"]
        T_C["Conscientiousness"]
        T_E["Extraversion"]
        T_A["Agreeableness"]
        T_N["Neuroticism"]
    end

    L1 & L6 --> M1 --> T_O
    L7 --> M2 --> T_C
    L3 & L4 --> M3 --> T_E
    L3 & L4 --> M4 --> T_A
    L5 & L8 --> M5 --> T_N
```

---

## 5. Multimodal Personality Analysis

### 5.1 Multimodal Fusion Architecture

```mermaid
flowchart TB
    subgraph INPUTS["Multimodal Inputs"]
        direction LR
        TEXT["📝 Text<br/>Conversation"]
        VOICE["🎤 Voice<br/>Audio"]
        BEHAV["📊 Behavioral<br/>Patterns"]
    end

    subgraph ENCODERS["Modality Encoders"]
        direction TB
        
        subgraph TEXT_ENC["Text Encoder"]
            TE1["BERT/RoBERTa"]
            TE2["[CLS] embedding"]
            TE3["768-dim vector"]
        end
        
        subgraph VOICE_ENC["Voice Encoder"]
            VE1["Wav2Vec 2.0"]
            VE2["Prosodic features"]
            VE3["512-dim vector"]
        end
        
        subgraph BEHAV_ENC["Behavioral Encoder"]
            BE1["Feature Engineering"]
            BE2["Dense layers"]
            BE3["128-dim vector"]
        end
    end

    subgraph FUSION["Late Fusion with Attention"]
        direction TB
        
        F1["Concatenate Modality Vectors"]
        F2["Cross-Modal Attention"]
        F3["Modality Weighting"]
        F4["Missing Modality Handling"]
        
        F1 --> F2 --> F3 --> F4
    end

    subgraph OUTPUT["Fused Personality Prediction"]
        FO1["Combined Feature Vector"]
        FO2["OCEAN Classifier"]
        FO3["Per-Modality Confidence"]
    end

    TEXT --> TEXT_ENC --> FUSION
    VOICE --> VOICE_ENC --> FUSION
    BEHAV --> BEHAV_ENC --> FUSION
    FUSION --> OUTPUT

    style FUSION fill:#fff3e0,stroke:#ef6c00
```

### 5.2 Voice-Based Personality Indicators

```mermaid
flowchart TB
    subgraph VOICE_FEATURES["Voice Feature Extraction"]
        direction TB
        
        subgraph PROSODIC["Prosodic Features"]
            P1["Pitch (F0) mean & variance"]
            P2["Speech rate (syllables/sec)"]
            P3["Pause frequency & duration"]
            P4["Energy contour"]
        end
        
        subgraph SPECTRAL["Spectral Features"]
            S1["MFCCs (13 coefficients)"]
            S2["Mel-spectrograms"]
            S3["Formant frequencies"]
        end
        
        subgraph TEMPORAL["Temporal Features"]
            T1["Speaking turn duration"]
            T2["Response latency"]
            T3["Interruption patterns"]
        end
    end

    subgraph CORRELATIONS["Voice → Personality Correlations"]
        direction TB
        
        C1["High pitch variability<br/>→ Extraversion (+) r=0.35"]
        C2["Faster speech rate<br/>→ Extraversion (+) r=0.30"]
        C3["More pauses<br/>→ Neuroticism (+) r=0.28"]
        C4["Vocal tremor<br/>→ Neuroticism (+) r=0.25"]
        C5["Louder speech<br/>→ Extraversion (+) r=0.32"]
        C6["Monotone pitch<br/>→ Low Openness r=-0.22"]
    end

    VOICE_FEATURES --> CORRELATIONS
```

### 5.3 Missing Modality Handling

```mermaid
flowchart TB
    INPUT["Input Check"] --> MODAL{Available<br/>Modalities?}
    
    MODAL -->|"Text + Voice + Behavior"| FULL["Full Multimodal<br/>(+12% F1 vs unimodal)"]
    MODAL -->|"Text + Voice"| PARTIAL1["Text-Voice Fusion"]
    MODAL -->|"Text + Behavior"| PARTIAL2["Text-Behavior Fusion"]
    MODAL -->|"Text Only"| TEXT_ONLY["Text-Only Mode"]
    MODAL -->|"Voice Only"| VOICE_ONLY["Voice-Only Mode"]

    subgraph HANDLING["Graceful Degradation"]
        H1["Zero-mask missing modalities"]
        H2["Reweight attention scores"]
        H3["Adjust confidence intervals"]
        H4["Flag lower reliability"]
    end

    FULL --> OUTPUT["Personality Output"]
    PARTIAL1 --> HANDLING --> OUTPUT
    PARTIAL2 --> HANDLING --> OUTPUT
    TEXT_ONLY --> HANDLING --> OUTPUT
    VOICE_ONLY --> HANDLING --> OUTPUT

    style FULL fill:#e8f5e9
    style TEXT_ONLY fill:#fff3e0
```

---

## 6. Personality Profile Management

### 6.1 Profile Entity Structure

```mermaid
classDiagram
    class PersonalityProfile {
        +UUID profileId
        +UUID userId
        +DateTime createdAt
        +DateTime lastUpdated
        +Int version
        +OceanScores currentScores
        +List~TraitAssessment~ assessmentHistory
        +ProfileMetadata metadata
        +CommunicationPreferences preferences
    }

    class OceanScores {
        +TraitScore openness
        +TraitScore conscientiousness
        +TraitScore extraversion
        +TraitScore agreeableness
        +TraitScore neuroticism
        +DateTime assessedAt
        +Float overallConfidence
    }

    class TraitScore {
        +Float value
        +Float confidenceLower
        +Float confidenceUpper
        +Int sampleCount
        +List~String~ evidenceMarkers
    }

    class TraitAssessment {
        +UUID assessmentId
        +DateTime timestamp
        +AssessmentSource source
        +OceanScores scores
        +Map~String, Float~ featureWeights
    }

    class CommunicationPreferences {
        +CommunicationStyle preferredStyle
        +Float preferredComplexity
        +Float preferredWarmth
        +Float preferredDirectness
        +Boolean prefersStructure
    }

    class ProfileMetadata {
        +Int totalInteractions
        +Int assessmentCount
        +DateTime firstAssessment
        +Float stabilityScore
        +List~String~ dominantTraits
    }

    PersonalityProfile --> OceanScores
    PersonalityProfile --> TraitAssessment
    PersonalityProfile --> CommunicationPreferences
    PersonalityProfile --> ProfileMetadata
    OceanScores --> TraitScore
```

### 6.2 Temporal Profile Evolution

```mermaid
flowchart TB
    subgraph TEMPORAL["Profile Evolution Over Time"]
        direction TB
        
        subgraph SESSION1["Session 1 (Initial)"]
            S1_ASSESS["Initial Assessment"]
            S1_CONF["Low Confidence<br/>(few samples)"]
            S1_SCORES["O:0.6 C:0.5 E:0.4 A:0.7 N:0.5<br/>±0.20"]
        end
        
        subgraph SESSION5["Session 5"]
            S5_ASSESS["5 Sessions Aggregated"]
            S5_CONF["Medium Confidence"]
            S5_SCORES["O:0.65 C:0.48 E:0.42 A:0.72 N:0.52<br/>±0.12"]
        end
        
        subgraph SESSION20["Session 20"]
            S20_ASSESS["Stable Profile"]
            S20_CONF["High Confidence"]
            S20_SCORES["O:0.68 C:0.46 E:0.40 A:0.75 N:0.50<br/>±0.06"]
        end
    end

    SESSION1 --> SESSION5 --> SESSION20

    subgraph AGGREGATION["Aggregation Strategy"]
        AGG1["Exponential Moving Average<br/>(α = 0.3 for recent weighting)"]
        AGG2["Confidence grows with √n samples"]
        AGG3["Outlier detection for sudden changes"]
        AGG4["Stability score tracks variance"]
    end

    TEMPORAL --> AGGREGATION

    style SESSION20 fill:#e8f5e9
    style SESSION1 fill:#fff3e0
```

### 6.3 Profile Versioning & History

```mermaid
flowchart LR
    subgraph VERSION_HISTORY["Profile Version History"]
        direction TB
        
        V1["Version 1<br/>2024-01-01<br/>Initial Assessment"]
        V2["Version 2<br/>2024-01-15<br/>After 5 sessions"]
        V3["Version 3<br/>2024-02-01<br/>Significant shift detected"]
        V4["Version 4<br/>2024-03-01<br/>Stable profile"]
    end

    V1 --> V2 --> V3 --> V4

    subgraph TRIGGERS["Version Update Triggers"]
        T1["Session count threshold (every 5)"]
        T2["Significant trait shift (>0.15)"]
        T3["Confidence threshold reached"]
        T4["Manual clinician update"]
    end

    subgraph STORAGE["Storage Strategy"]
        ST1["Current version: Fast access (Redis)"]
        ST2["History: Append-only (PostgreSQL)"]
        ST3["Snapshots: Periodic backup"]
    end
```

---

## 7. Response Adaptation Engine

### 7.1 Personality-Based Style Mapping

```mermaid
flowchart TB
    subgraph OCEAN_INPUT["OCEAN Profile Input"]
        direction LR
        IN_O["O: 0.72"]
        IN_C["C: 0.45"]
        IN_E["E: 0.38"]
        IN_A["A: 0.81"]
        IN_N["N: 0.56"]
    end

    subgraph STYLE_MAPPING["Style Mapping Engine"]
        direction TB
        
        subgraph OPENNESS_MAP["Openness Mapping"]
            OM1["High O → Exploratory language"]
            OM2["High O → Abstract concepts"]
            OM3["High O → Metaphors welcome"]
            OM4["Low O → Concrete, practical"]
        end
        
        subgraph CONSCIENT_MAP["Conscientiousness Mapping"]
            CM1["High C → Structured responses"]
            CM2["High C → Clear timelines"]
            CM3["High C → Detailed homework"]
            CM4["Low C → Flexible, less structured"]
        end
        
        subgraph EXTRA_MAP["Extraversion Mapping"]
            EM1["High E → Energetic tone"]
            EM2["High E → More elaboration"]
            EM3["Low E → Concise, space for reflection"]
            EM4["Low E → Less overwhelming"]
        end
        
        subgraph AGREE_MAP["Agreeableness Mapping"]
            AM1["High A → High validation"]
            AM2["High A → Collaborative framing"]
            AM3["Low A → Direct, challenging OK"]
            AM4["Low A → Evidence-based"]
        end
        
        subgraph NEURO_MAP["Neuroticism Mapping"]
            NM1["High N → Extra reassurance"]
            NM2["High N → Safety emphasis"]
            NM3["High N → Grounding language"]
            NM4["Low N → Standard approach"]
        end
    end

    subgraph OUTPUT_STYLE["Output Style Parameters"]
        direction LR
        OUT1["Warmth: 0.85"]
        OUT2["Structure: 0.60"]
        OUT3["Complexity: 0.70"]
        OUT4["Directness: 0.55"]
        OUT5["Energy: 0.45"]
    end

    OCEAN_INPUT --> STYLE_MAPPING --> OUTPUT_STYLE
```

### 7.2 Communication Style Matrix

```mermaid
flowchart TB
    subgraph STYLE_MATRIX["Communication Style Matrix"]
        direction TB
        
        subgraph HIGH_N["High Neuroticism (N > 0.6)"]
            HN1["✓ Extra validation"]
            HN2["✓ Gentle pacing"]
            HN3["✓ Reassurance phrases"]
            HN4["✓ Safety reminders"]
            HN5["✗ Avoid overwhelming"]
        end
        
        subgraph LOW_E["Low Extraversion (E < 0.4)"]
            LE1["✓ Concise responses"]
            LE2["✓ Space for processing"]
            LE3["✓ Written over verbal"]
            LE4["✓ One question at a time"]
            LE5["✗ Avoid excessive energy"]
        end
        
        subgraph HIGH_O["High Openness (O > 0.7)"]
            HO1["✓ Metaphors, analogies"]
            HO2["✓ Novel perspectives"]
            HO3["✓ Abstract concepts"]
            HO4["✓ Creative exercises"]
        end
        
        subgraph HIGH_C["High Conscientiousness (C > 0.7)"]
            HC1["✓ Clear structure"]
            HC2["✓ Specific timelines"]
            HC3["✓ Progress tracking"]
            HC4["✓ Detailed plans"]
        end
    end

    subgraph EXAMPLE["Example: High N, Low E User"]
        EX1["'I understand this feels overwhelming<br/>right now. Let's take this slowly,<br/>one small step at a time.<br/>What feels most manageable to<br/>focus on first?'"]
    end

    HIGH_N --> EXAMPLE
    LOW_E --> EXAMPLE
```

### 7.3 Dynamic Adaptation Flow

```mermaid
flowchart TB
    INPUT["Response to Generate"] --> LOAD["Load User Profile"]
    
    LOAD --> PARAMS["Extract Style Parameters"]
    
    PARAMS --> PROMPT["Construct Adapted Prompt"]
    
    subgraph PROMPT_CONSTRUCTION["Prompt Construction"]
        PC1["Base therapeutic content"]
        PC2["+ Personality-specific instructions"]
        PC3["+ Tone modifiers"]
        PC4["+ Structural preferences"]
    end
    
    PROMPT --> GENERATE["LLM Generation"]
    
    GENERATE --> VALIDATE["Style Validation"]
    
    VALIDATE --> CHECK{Matches<br/>Profile?}
    
    CHECK -->|Yes| OUTPUT["Final Response"]
    CHECK -->|No| REVISE["Revise with Feedback"]
    REVISE --> GENERATE
```

---

## 8. Empathic Response Generation

### 8.1 Three-Component Empathy Model

```mermaid
flowchart TB
    subgraph EMPATHY_MODEL["Three-Component Empathy Architecture"]
        direction TB
        
        subgraph COGNITIVE["🧠 Cognitive Empathy"]
            COG1["Understanding user's perspective"]
            COG2["Recognizing their mental state"]
            COG3["Interpretations of feelings"]
            COG4["'It sounds like you're feeling...'"]
        end
        
        subgraph AFFECTIVE["💜 Affective Empathy"]
            AFF1["Emotional resonance"]
            AFF2["Sharing the feeling"]
            AFF3["Emotional reactions"]
            AFF4["'That must be really difficult'"]
        end
        
        subgraph COMPASSIONATE["🤝 Compassionate Empathy"]
            COMP1["Desire to help"]
            COMP2["Action-oriented support"]
            COMP3["Explorations of feelings"]
            COMP4["'What would help right now?'"]
        end
    end

    subgraph OUTPUT["Empathic Response"]
        OUT["Combines all three components<br/>weighted by context and personality"]
    end

    COGNITIVE --> OUTPUT
    AFFECTIVE --> OUTPUT
    COMPASSIONATE --> OUTPUT

    style COGNITIVE fill:#e3f2fd
    style AFFECTIVE fill:#fce4ec
    style COMPASSIONATE fill:#e8f5e9
```

### 8.2 MoEL (Mixture of Empathetic Listeners) Architecture

```mermaid
flowchart TB
    subgraph MOEL["MoEL Architecture"]
        direction TB
        
        INPUT["User Message + Context"] --> ENCODER["Transformer Encoder"]
        
        ENCODER --> EMOTION_TRACKER["Emotion Tracker"]
        
        EMOTION_TRACKER --> DISTRIBUTION["Emotion Distribution<br/>(32 emotion softmax)"]
        
        subgraph LISTENERS["Specialized Listener Decoders"]
            direction LR
            L1["😢 Sadness<br/>Listener"]
            L2["😰 Anxiety<br/>Listener"]
            L3["😠 Anger<br/>Listener"]
            L4["😊 Joy<br/>Listener"]
            L5["... 28 more<br/>listeners"]
        end
        
        DISTRIBUTION --> LISTENERS
        
        LISTENERS --> META["Meta-Listener<br/>(Soft Combination)"]
        
        META --> OUTPUT["Empathic Response"]
    end

    subgraph ATTENTION["Interpretability"]
        ATT1["Attention visualization"]
        ATT2["Which emotions weighted"]
        ATT3["Which listener contributed"]
    end

    META --> ATTENTION

    style LISTENERS fill:#f3e5f5
```

### 8.3 Emotion-Aware Response Pipeline

```mermaid
flowchart TB
    subgraph PIPELINE["Empathic Response Pipeline"]
        direction TB
        
        INPUT["User Message"] --> DETECT["Emotion Detection"]
        
        DETECT --> CLASSIFY["Classify Emotion State<br/>(Primary + Secondary)"]
        
        CLASSIFY --> INTENSITY["Assess Intensity<br/>(0.0 - 1.0)"]
        
        INTENSITY --> STRATEGY["Select Empathy Strategy"]
        
        subgraph STRATEGY["Strategy Selection"]
            direction LR
            
            ST1["High distress → Validation first"]
            ST2["Moderate → Balance all three"]
            ST3["Low distress → Compassionate action"]
        end
        
        STRATEGY --> COMPONENTS["Assemble Components"]
        
        subgraph COMPONENTS["Response Components"]
            C1["Validation statement"]
            C2["Reflection of feeling"]
            C3["Supportive action offer"]
        end
        
        COMPONENTS --> PERSONALITY["Apply Personality Adaptation"]
        
        PERSONALITY --> OUTPUT["Final Empathic Response"]
    end

    style STRATEGY fill:#fff3e0
```

---

## 9. Cultural Adaptation System

### 9.1 Cultural Context Framework

```mermaid
flowchart TB
    subgraph CULTURAL["Cultural Adaptation Framework"]
        direction TB
        
        subgraph DETECTION["Cultural Context Detection"]
            CD1["Language patterns"]
            CD2["Communication style cues"]
            CD3["Explicit user preference"]
            CD4["Geographic context"]
        end
        
        subgraph DIMENSIONS["Cultural Dimensions"]
            direction LR
            
            subgraph HOFSTEDE["Hofstede Dimensions"]
                H1["Individualism vs Collectivism"]
                H2["Power Distance"]
                H3["Uncertainty Avoidance"]
                H4["Masculinity vs Femininity"]
            end
            
            subgraph MENTAL_HEALTH["Mental Health Context"]
                M1["Stigma levels"]
                M2["Help-seeking norms"]
                M3["Family involvement"]
                M4["Emotional expression norms"]
            end
        end
        
        subgraph ADAPTATION["Cultural Adaptations"]
            A1["Eastern: Indirect emotional expression"]
            A2["Eastern: Family-inclusive framing"]
            A3["Western: Direct problem discussion"]
            A4["Western: Individual autonomy emphasis"]
        end
    end

    DETECTION --> DIMENSIONS --> ADAPTATION
```

### 9.2 Cultural Prompting Strategy

```mermaid
flowchart TB
    subgraph CULTURAL_PROMPTING["Cultural Prompting for Empathy"]
        direction TB
        
        BASE["Base Therapeutic Response"] --> ENHANCE
        
        subgraph ENHANCE["Cultural Enhancement"]
            E1["Add cultural context to system prompt"]
            E2["Include relevant traditions/values"]
            E3["Adapt family dynamics understanding"]
            E4["Adjust directness level"]
        end
        
        ENHANCE --> EXAMPLES
        
        subgraph EXAMPLES["Example Adaptations"]
            direction LR
            
            EX1["Western Default:<br/>'How are YOU feeling<br/>about this situation?'"]
            
            EX2["Collectivist Adapted:<br/>'How has this affected<br/>you and your family?'"]
            
            EX3["High Context Adapted:<br/>'I sense there may be<br/>more beneath the surface...'"]
        end
    end

    style ENHANCE fill:#e8f5e9
```

---

## 10. Data Flow Architecture

### 10.1 Complete Personality Data Flow

```mermaid
flowchart TB
    subgraph INPUT["📥 Input Processing"]
        I1[/"User Message"/]
        I2[/"Voice Audio"/]
        I3[/"Session Context"/]
    end

    subgraph FEATURE_EXTRACT["🔍 Feature Extraction"]
        FE1["Text: BERT embeddings + LIWC"]
        FE2["Voice: Prosodic features"]
        FE3["Behavior: Engagement metrics"]
    end

    subgraph DETECTION["🎯 Personality Detection"]
        D1["RoBERTa Classification"]
        D2["LLM Zero-shot"]
        D3["Voice Model"]
        D4["Ensemble Fusion"]
    end

    subgraph PROFILE_UPDATE["📊 Profile Management"]
        PU1["Load Current Profile"]
        PU2["Aggregate New Assessment"]
        PU3["Update Confidence"]
        PU4["Version if Changed"]
    end

    subgraph ADAPTATION["🎨 Response Adaptation"]
        AD1["Map Traits to Style"]
        AD2["Select Empathy Strategy"]
        AD3["Apply Cultural Context"]
        AD4["Generate Adapted Prompt"]
    end

    subgraph OUTPUT["📤 Output"]
        O1[/"Style Parameters"/]
        O2[/"Empathy Components"/]
        O3[/"Updated Profile"/]
    end

    INPUT --> FEATURE_EXTRACT
    FEATURE_EXTRACT --> DETECTION
    DETECTION --> PROFILE_UPDATE
    PROFILE_UPDATE --> ADAPTATION
    ADAPTATION --> OUTPUT

    style DETECTION fill:#e3f2fd
    style ADAPTATION fill:#e8f5e9
```

### 10.2 Profile Synchronization Flow

```mermaid
sequenceDiagram
    participant UI as User Interface
    participant PEM as Personality Module
    participant TM as Therapy Module
    participant DM as Diagnosis Module
    participant MM as Memory Module

    Note over UI,MM: Session Start
    UI->>PEM: User message
    PEM->>MM: Get current profile
    MM-->>PEM: PersonalityProfile v4
    
    PEM->>PEM: Detect traits from message
    PEM->>PEM: Update profile (if changed)
    
    alt Profile Updated
        PEM->>MM: Store new version (v5)
        PEM->>TM: ProfileUpdatedEvent
        PEM->>DM: ProfileUpdatedEvent
    end
    
    TM->>PEM: GetStyleRecommendation
    PEM-->>TM: StyleParameters
    
    DM->>PEM: GetPersonalityContext
    PEM-->>DM: OceanScores + Confidence
```

---

## 11. Integration Interfaces

### 11.1 Public Service Interfaces

```mermaid
classDiagram
    class IPersonalityService {
        <<interface>>
        +detectPersonality(userId, text, audio?) PersonalityAssessment
        +getProfile(userId) PersonalityProfile
        +updateProfile(userId, assessment) PersonalityProfile
        +getStyleRecommendation(userId) StyleParameters
    }

    class IAdaptationService {
        <<interface>>
        +adaptResponse(userId, baseResponse) AdaptedResponse
        +getEmpathyComponents(userId, emotion) EmpathyComponents
        +getCulturalContext(userId) CulturalContext
    }

    class IEmpathyService {
        <<interface>>
        +detectEmotion(text, audio?) EmotionState
        +generateEmpathicResponse(context) EmpathicResponse
        +selectEmpathyStrategy(emotionState) EmpathyStrategy
    }

    class PersonalityAssessment {
        +OceanScores scores
        +Float confidence
        +AssessmentSource source
        +List~String~ evidence
    }

    class StyleParameters {
        +Float warmth
        +Float structure
        +Float complexity
        +Float directness
        +Float energy
        +Map~String, Any~ customParams
    }

    class AdaptedResponse {
        +String content
        +StyleParameters appliedStyle
        +EmpathyComponents empathy
        +Float adaptationConfidence
    }

    IPersonalityService --> PersonalityAssessment
    IAdaptationService --> StyleParameters
    IAdaptationService --> AdaptedResponse
```

### 11.2 Module Integration Contracts

```mermaid
flowchart TB
    subgraph PERSONALITY_PROVIDES["Personality Module Provides"]
        PP1["PersonalityProfile"]
        PP2["StyleParameters"]
        PP3["EmpathyComponents"]
        PP4["CulturalContext"]
        PP5["EmotionState"]
    end

    subgraph CONSUMERS["Consumer Modules"]
        direction TB
        
        subgraph THERAPY_USES["Therapy Module Uses"]
            TU1["Style for session pacing"]
            TU2["Technique selection weighting"]
            TU3["Homework complexity"]
        end
        
        subgraph DIAG_USES["Diagnosis Module Uses"]
            DU1["Neuroticism for risk weighting"]
            DU2["Response style adjustment"]
        end
        
        subgraph RESP_USES["Response Module Uses"]
            RU1["All style parameters"]
            RU2["Empathy components"]
            RU3["Cultural adaptation"]
        end
    end

    PP1 --> THERAPY_USES
    PP1 --> DIAG_USES
    PP2 --> RESP_USES
    PP3 --> RESP_USES
    PP4 --> RESP_USES
```

---

## 12. Event-Driven Architecture

### 12.1 Personality Events

```mermaid
flowchart TB
    subgraph PUBLISHERS["📤 Event Publishers"]
        P1["Detection Service"]
        P2["Profile Manager"]
        P3["Adaptation Service"]
    end

    subgraph EVENTS["📨 Event Topics"]
        E1["personality.assessment.completed"]
        E2["personality.profile.updated"]
        E3["personality.style.generated"]
        E4["personality.emotion.detected"]
        E5["personality.significant.change"]
    end

    subgraph SUBSCRIBERS["📥 Event Subscribers"]
        S1["Therapy Module"]
        S2["Diagnosis Module"]
        S3["Memory Module"]
        S4["Analytics Service"]
        S5["Response Module"]
    end

    P1 --> E1 & E4
    P2 --> E2 & E5
    P3 --> E3

    E1 --> S3 & S4
    E2 --> S1 & S2 & S3
    E3 --> S5
    E4 --> S1 & S2
    E5 --> S1 & S4
```

### 12.2 Event Schema

```mermaid
classDiagram
    class PersonalityEvent {
        <<abstract>>
        +UUID eventId
        +DateTime timestamp
        +UUID userId
        +String eventType
    }

    class AssessmentCompletedEvent {
        +UUID assessmentId
        +OceanScores scores
        +Float confidence
        +AssessmentSource source
        +List~String~ evidence
    }

    class ProfileUpdatedEvent {
        +Int previousVersion
        +Int newVersion
        +OceanScores previousScores
        +OceanScores newScores
        +List~String~ changedTraits
        +String updateReason
    }

    class StyleGeneratedEvent {
        +StyleParameters style
        +UUID sessionId
        +String targetModule
    }

    class EmotionDetectedEvent {
        +EmotionState emotion
        +Float intensity
        +String context
    }

    class SignificantChangeEvent {
        +String trait
        +Float previousValue
        +Float newValue
        +Float changeAmount
        +String possibleCause
    }

    PersonalityEvent <|-- AssessmentCompletedEvent
    PersonalityEvent <|-- ProfileUpdatedEvent
    PersonalityEvent <|-- StyleGeneratedEvent
    PersonalityEvent <|-- EmotionDetectedEvent
    PersonalityEvent <|-- SignificantChangeEvent
```

---

## Appendix A: Big Five Trait Quick Reference

| Trait | High (>0.7) | Low (<0.3) | Therapeutic Implication |
|-------|-------------|------------|------------------------|
| **Openness** | Creative, curious, abstract | Practical, conventional | ACT/exploratory vs structured CBT |
| **Conscientiousness** | Organized, disciplined | Flexible, spontaneous | Homework complexity, structure level |
| **Extraversion** | Outgoing, energetic | Reserved, introspective | Session energy, response length |
| **Agreeableness** | Cooperative, trusting | Skeptical, challenging | Validation level, directness |
| **Neuroticism** | Anxious, emotionally reactive | Calm, stable | Safety emphasis, reassurance frequency |

## Appendix B: Style Parameter Ranges

| Parameter | Range | Low Interpretation | High Interpretation |
|-----------|-------|-------------------|---------------------|
| **Warmth** | 0.0-1.0 | Professional, neutral | Warm, nurturing |
| **Structure** | 0.0-1.0 | Flexible, open-ended | Highly organized |
| **Complexity** | 0.0-1.0 | Simple, concrete | Abstract, nuanced |
| **Directness** | 0.0-1.0 | Indirect, gentle | Direct, clear |
| **Energy** | 0.0-1.0 | Calm, subdued | Energetic, enthusiastic |

---

*Document Version: 2.0*  
*Last Updated: December 30, 2025*  
*Status: Technical Blueprint*  
*Dependencies: Memory Module, Therapy Module, Response Module*
