# 🗺️ SOLACE-AI COMPLETE PROJECT MAP

**Generated**: 2025-11-15
**Purpose**: Comprehensive visualization of the Solace-AI mental health chatbot architecture
**Status**: Current implementation analysis with 205 Python files

---

## 📑 TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Complete Directory Structure](#2-complete-directory-structure)
3. [Module Dependency Map](#3-module-dependency-map)
4. [Agent System Architecture](#4-agent-system-architecture)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [API Endpoints Map](#6-api-endpoints-map)
7. [Configuration Structure](#7-configuration-structure)
8. [Integration Points](#8-integration-points)
9. [Memory Architecture](#9-memory-architecture)
10. [Security & Compliance](#10-security--compliance)
11. [Entry Points & Workflows](#11-entry-points--workflows)
12. [Service Layer Map](#12-service-layer-map)

---

## 1. PROJECT OVERVIEW

### **Purpose**
Solace-AI is an advanced mental health AI companion that provides personalized support through:
- Multi-agent architecture with specialized agents (emotion, safety, therapy, personality, diagnosis)
- Voice and text interaction capabilities
- Comprehensive mental health assessment (PHQ-9, GAD-7, personality tests)
- Evidence-based therapeutic techniques (CBT, mindfulness, solution-focused therapy)
- Vector database for contextual memory and semantic search

### **Core Technology Stack**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Framework** | LangChain, Agno | Agent orchestration and LLM integration |
| **LLM Providers** | Google Gemini, OpenAI | Language understanding and generation |
| **Voice** | Whisper V3 Turbo ASR, TTS | Speech recognition and synthesis |
| **Memory** | ChromaDB, Vector Embeddings | Contextual memory and retrieval |
| **API** | FastAPI, Uvicorn | REST API for mobile integration |
| **Security** | JWT, HIPAA validator | Authentication and compliance |
| **Infrastructure** | Dependency Injection, Event Bus | Modular architecture |

### **Key Statistics**
- **Total Python Files**: 205
- **Lines of Code**: ~86,470
- **Main Modules**: 24 top-level directories
- **Agents**: 13+ specialized agents
- **API Endpoints**: 30+ REST endpoints
- **Data Namespaces**: 7 vector DB collections

---

## 2. COMPLETE DIRECTORY STRUCTURE

```
R:\Solace-AI\
│
├── 📄 Root Files
│   ├── api_server.py              # FastAPI server for mobile integration
│   ├── main.py                    # Application entry point (imports from src)
│   ├── test_optimization.py       # Optimization tests
│   ├── requirements.txt           # Core dependencies
│   ├── requirements_voice.txt     # Voice-specific dependencies
│   ├── Dockerfile                 # Container configuration
│   ├── pytest.ini                 # Testing configuration
│   ├── README.md                  # Project documentation
│   ├── improvements.md            # Improvement suggestions
│   ├── OPTIMIZATION_REPORT.md     # Performance optimization report
│   └── .env                       # Environment configuration (not in git)
│
└── 📁 src/                        # Main source code directory
    │
    ├── 📁 agents/                 # Multi-agent system (24 files)
    │   ├── 📁 base/               # Base agent classes
    │   │   ├── base_agent.py     # Abstract base agent using Agno framework
    │   │   └── __init__.py
    │   │
    │   ├── 📁 core/               # Core conversation agents
    │   │   ├── chat_agent.py     # Main conversational agent
    │   │   ├── emotion_agent.py  # Emotion detection and analysis
    │   │   ├── personality_agent.py  # Personality adaptation
    │   │   ├── safety_agent.py   # Crisis detection and safety
    │   │   └── __init__.py
    │   │
    │   ├── 📁 clinical/           # Clinical agents
    │   │   ├── diagnosis_agent.py  # Mental health diagnosis (LEGACY - 1,324 lines)
    │   │   ├── therapy_agent.py  # Therapeutic techniques
    │   │   └── __init__.py
    │   │
    │   ├── 📁 orchestration/      # Agent coordination (2 massive files)
    │   │   ├── agent_orchestrator.py  # Main orchestrator (2,382 lines)
    │   │   ├── supervisor_agent.py    # Quality assurance (917 lines)
    │   │   └── __init__.py
    │   │
    │   ├── 📁 therapeutic_friction/  # Therapeutic breakthrough detection
    │   │   ├── base_friction_agent.py
    │   │   ├── breakthrough_detection_agent.py  # (822 lines)
    │   │   ├── friction_coordinator.py          # (1,136 lines)
    │   │   ├── readiness_assessment_agent.py
    │   │   └── __init__.py
    │   │
    │   ├── 📁 support/            # Utility agents
    │   │   ├── search_agent.py   # Web search capabilities
    │   │   ├── crawler_agent.py  # Web crawling for knowledge
    │   │   └── __init__.py
    │   │
    │   └── 📁 validation/         # Agent validation
    │       └── __init__.py
    │
    ├── 📁 diagnosis/              # Diagnosis implementations (24 files) ⚠️ DUPLICATION ISSUE
    │   ├── comprehensive_diagnosis.py      # Main implementation (1,452 lines)
    │   ├── enhanced_diagnosis.py           # Enhanced variant (1,436 lines)
    │   ├── differential_diagnosis.py       # Differential diagnosis (1,366 lines)
    │   ├── integrated_diagnosis.py         # Integration attempt
    │   ├── enterprise_multimodal_pipeline.py  # Enterprise version (1,620 lines)
    │   ├── comprehensive_diagnostic_report.py # Report generation (1,290 lines)
    │   ├── temporal_analysis.py            # Temporal symptom tracking
    │   ├── cultural_sensitivity.py         # Cultural adaptations
    │   ├── adaptive_learning.py            # Model adaptation
    │   ├── model_management.py             # Model lifecycle
    │   ├── therapeutic_friction.py         # Friction analysis
    │   ├── enhanced_diagnosis_example.py   # Usage examples
    │   ├── enterprise_pipeline_example.py  # Enterprise examples
    │   ├── enhanced_integrated_system.py   # System integration
    │   │
    │   └── 📁 enterprise/         # Enterprise-grade features
    │       ├── 📁 config/         # Configuration
    │       │   ├── base_config.py
    │       │   ├── constants.py
    │       │   ├── validation.py
    │       │   └── __init__.py
    │       ├── 📁 models/         # ML models
    │       │   ├── base.py
    │       │   ├── bayesian.py   # Bayesian diagnosis models
    │       │   ├── fusion.py     # Multimodal fusion
    │       │   └── __init__.py
    │       ├── 📁 clinical/       # Clinical features
    │       ├── 📁 feature_extraction/  # Feature extractors
    │       ├── 📁 management/     # Model management
    │       ├── 📁 utils/          # Utilities
    │       ├── 📁 validation/     # Validation logic
    │       └── __init__.py
    │
    ├── 📁 services/               # Service layer (7 files)
    │   └── 📁 diagnosis/          # Diagnosis service abstraction
    │       ├── interfaces.py      # IDiagnosisService, IDiagnosisOrchestrator
    │       ├── unified_service.py # Unified diagnosis service (810 lines)
    │       ├── orchestrator.py    # Service orchestration
    │       ├── agent_adapter.py   # Adapter for legacy agents
    │       ├── memory_integration.py  # Memory integration
    │       ├── integration_setup.py   # Setup utilities
    │       └── __init__.py
    │
    ├── 📁 memory/                 # Memory management (4 files)
    │   ├── enhanced_memory_system.py  # Main memory system (1,118 lines)
    │   ├── 📁 semantic_memory/    # Semantic memory manager
    │   │   ├── semantic_memory_manager.py
    │   │   └── __init__.py
    │   └── __init__.py
    │
    ├── 📁 database/               # Data storage (4 files)
    │   ├── central_vector_db.py   # Centralized vector database
    │   ├── vector_store.py        # Vector store abstraction
    │   ├── conversation_tracker.py  # Conversation persistence
    │   ├── therapeutic_friction_vector_manager.py
    │   └── __init__.py
    │
    ├── 📁 models/                 # LLM integration (5 files)
    │   ├── llm.py                 # Base LLM interface
    │   ├── gemini_llm.py          # Gemini-specific wrapper
    │   ├── gemini_api.py          # Gemini API client
    │   ├── agno_llm_wrapper.py    # Agno framework wrapper
    │   └── __init__.py
    │
    ├── 📁 providers/              # Provider implementations
    │   ├── 📁 llm/                # LLM providers
    │   │   ├── gemini_provider.py
    │   │   ├── openai_provider.py
    │   │   └── __init__.py
    │   ├── 📁 storage/            # Storage providers
    │   └── 📁 voice/              # Voice providers
    │
    ├── 📁 personality/            # Personality assessment (3 files)
    │   ├── chatbot_personality.py  # Chatbot's personality
    │   ├── big_five.py            # Big Five model (837 lines)
    │   ├── mbti.py                # MBTI assessment
    │   ├── 📁 profiles/           # Personality profile templates
    │   │   ├── analytical_advisor.json
    │   │   ├── empathetic_listener.json
    │   │   └── supportive_counselor.json
    │   └── __init__.py
    │
    ├── 📁 components/             # Reusable components (11 files)
    │   ├── base_module.py         # Module system base
    │   ├── llm_module.py          # LLM component
    │   ├── central_vector_db_module.py  # Vector DB component
    │   ├── vector_store_module.py
    │   ├── voice_component.py     # Voice integration
    │   ├── voice_module.py
    │   ├── dynamic_personality_assessment.py  # User personality (975 lines)
    │   ├── integrated_assessment.py  # Assessment integration
    │   ├── diagnosis_results.py   # Results formatting
    │   ├── ui_manager.py          # UI management
    │   └── __init__.py
    │
    ├── 📁 clinical_decision_support/  # Clinical support (6 files)
    │   ├── clinical_guidelines.py  # Clinical guidelines database
    │   ├── diagnostic_algorithms.py  # Diagnosis algorithms
    │   ├── risk_assessment.py     # Risk scoring
    │   ├── treatment_recommendations.py  # Treatment suggestions
    │   ├── rule_engine.py         # Clinical rules
    │   ├── alerts.py              # Clinical alerts
    │   └── __init__.py
    │
    ├── 📁 knowledge/              # Knowledge bases (4 files)
    │   ├── 📁 therapeutic/        # Therapeutic knowledge
    │   │   ├── knowledge_base.py
    │   │   ├── technique_service.py
    │   │   ├── techniques.json    # CBT, mindfulness, etc.
    │   │   └── __init__.py
    │   └── 📁 clinical/           # Clinical knowledge
    │       ├── clinical_guidelines_db.py
    │       └── __init__.py
    │
    ├── 📁 enterprise/             # Enterprise features (8 files) ⚠️ NOT INTEGRATED
    │   ├── analytics_dashboard.py  # Analytics (1,187 lines)
    │   ├── real_time_monitoring.py  # Monitoring (997 lines)
    │   ├── data_reliability.py    # Data quality (1,936 lines)
    │   ├── quality_assurance.py   # QA system (1,551 lines)
    │   ├── clinical_compliance.py  # Compliance checking
    │   ├── knowledge_integration.py  # Knowledge graph
    │   ├── dependency_injection.py  # DI setup
    │   ├── enterprise_orchestrator.py  # Enterprise orchestration
    │   └── __init__.py
    │
    ├── 📁 feature_extractors/     # Feature extraction (7 files)
    │   ├── base.py                # Base extractor
    │   ├── text_extractors.py     # Text features
    │   ├── voice_extractors.py    # Voice features
    │   ├── behavioral_extractors.py  # Behavioral patterns
    │   ├── contextual_extractors.py  # Context features
    │   ├── temporal_extractors.py  # Temporal features
    │   ├── multimodal_fusion.py   # Feature fusion
    │   └── __init__.py
    │
    ├── 📁 ml_models/              # ML model implementations (4 files)
    │   ├── base.py                # Base model class
    │   ├── bayesian.py            # Bayesian models
    │   ├── fusion.py              # Fusion models
    │   └── __init__.py
    │
    ├── 📁 config/                 # Configuration (7 files)
    │   ├── settings.py            # Main app configuration
    │   ├── security.py            # Security settings
    │   ├── credential_manager.py  # Credential management
    │   ├── feature_flags.py       # Feature toggles
    │   ├── supervision_config.py  # Supervisor configuration
    │   ├── optimization_config.py  # Performance config
    │   └── __init__.py
    │
    ├── 📁 core/                   # Core infrastructure
    │   ├── 📁 exceptions/         # Exception hierarchy
    │   │   ├── base_exceptions.py
    │   │   ├── agent_exceptions.py
    │   │   ├── llm_exceptions.py
    │   │   ├── security_exceptions.py
    │   │   ├── storage_exceptions.py
    │   │   ├── factory_exceptions.py
    │   │   └── __init__.py
    │   ├── 📁 interfaces/         # Interface definitions
    │   │   ├── agent_interface.py
    │   │   ├── llm_interface.py
    │   │   ├── storage_interface.py
    │   │   ├── config_interface.py
    │   │   ├── logger_interface.py
    │   │   ├── event_interface.py
    │   │   └── __init__.py
    │   ├── 📁 factories/          # Factory patterns
    │   │   ├── llm_factory.py     # LLM provider factory
    │   │   └── __init__.py
    │   ├── 📁 events/             # Event system
    │   └── 📁 services/           # Core services
    │
    ├── 📁 infrastructure/         # Infrastructure layer
    │   ├── 📁 di/                 # Dependency injection
    │   │   ├── container.py       # DI container (sophisticated)
    │   │   ├── decorators.py      # Injection decorators
    │   │   ├── diagnosis_registration.py  # Diagnosis DI setup
    │   │   └── __init__.py
    │   ├── 📁 config/             # Config management
    │   │   ├── config_manager.py
    │   │   └── __init__.py
    │   └── 📁 logging/            # Logging infrastructure
    │
    ├── 📁 integration/            # Integration layer (4 files)
    │   ├── event_bus.py           # Event-driven messaging
    │   ├── friction_engine.py     # Therapeutic friction
    │   ├── supervision_mesh.py    # Supervision integration
    │   └── __init__.py
    │
    ├── 📁 security/               # Security (2 files)
    │   ├── input_validator.py     # Input validation (SQL injection, XSS, etc.)
    │   ├── secrets_manager.py     # Secret management
    │   └── __init__.py
    │
    ├── 📁 compliance/             # Compliance (1 file)
    │   ├── hipaa_validator.py     # HIPAA PHI detection
    │   └── __init__.py
    │
    ├── 📁 auth/                   # Authentication
    │   ├── jwt_utils.py           # JWT token management
    │   ├── dependencies.py        # Auth dependencies for API
    │   ├── models.py              # Auth models (UserCreate, Token, etc.)
    │   └── __init__.py
    │
    ├── 📁 middleware/             # API middleware
    │   ├── security.py            # Security middleware (headers, rate limiting)
    │   └── __init__.py
    │
    ├── 📁 utils/                  # Utilities (28 files)
    │   ├── logger.py              # Logging utilities
    │   ├── metrics.py             # Metrics tracking
    │   ├── memory_factory.py      # Memory instance factory
    │   ├── vector_db_integration.py  # Vector DB helpers
    │   ├── context_aware_memory.py  # Context memory
    │   ├── conversation_memory.py  # Conversation memory
    │   ├── agentic_rag.py         # RAG implementation
    │   ├── error_handling.py      # Error utilities
    │   ├── helpers.py             # General helpers
    │   ├── console_utils.py       # Console formatting
    │   ├── device_utils.py        # Device detection (CPU/GPU)
    │   ├── response_envelope.py   # Response formatting
    │   ├── sentiment_utils.py     # Sentiment analysis
    │   ├── migration_utils.py     # Data migration
    │   ├── import_analyzer.py     # Import analysis
    │   ├── 📁 Voice Utilities
    │   │   ├── whisper_asr.py     # Whisper speech recognition
    │   │   ├── voice_ai.py        # Voice AI integration
    │   │   ├── voice_input_manager.py  # Voice input handling
    │   │   ├── voice_emotion_analyzer.py  # Voice emotion
    │   │   ├── voice_clone_integration.py  # Voice cloning
    │   │   ├── celebrity_voice_cloner.py  # Celebrity voices
    │   │   ├── dia_tts.py         # Text-to-speech
    │   │   └── audio_player.py    # Audio playback
    │   └── __init__.py
    │
    ├── 📁 analysis/               # Analysis modules (2 files)
    │   ├── conversation_analysis.py  # Conversation insights
    │   ├── emotion_analysis.py    # Emotion tracking
    │   └── __init__.py
    │
    ├── 📁 monitoring/             # Monitoring (2 files)
    │   ├── health_monitor.py      # System health
    │   ├── supervisor_metrics.py  # Supervisor metrics
    │   └── __init__.py
    │
    ├── 📁 optimization/           # Performance optimization (6 files)
    │   ├── performance_profiler.py  # Performance profiling
    │   ├── agent_performance_analyzer.py  # Agent analysis
    │   ├── context_optimizer.py   # Context optimization
    │   ├── prompt_optimizer.py    # Prompt optimization
    │   ├── optimized_orchestrator.py  # Optimized orchestrator
    │   └── __init__.py
    │
    ├── 📁 auditing/               # Audit system (1 file)
    │   ├── audit_system.py        # Audit trail
    │   └── __init__.py
    │
    ├── 📁 research/               # Research tools (1 file)
    │   ├── real_time_research.py  # Real-time research
    │   └── __init__.py
    │
    ├── 📁 dashboard/              # Dashboards (1 file)
    │   ├── supervision_dashboard.py  # Supervision UI
    │   └── __init__.py
    │
    ├── 📁 cli/                    # CLI interfaces (1 file)
    │   ├── voice_chat.py          # Voice chat CLI
    │   └── __init__.py
    │
    ├── 📁 data/                   # Data storage
    │   ├── 📁 conversations/      # Conversation history
    │   │   └── 📁 test_user/
    │   │       └── metadata.json
    │   ├── 📁 diagnostic_data/    # Diagnosis data
    │   │   └── test_user_metadata.json
    │   ├── 📁 knowledge/          # Knowledge base
    │   │   └── test_user_metadata.json
    │   ├── 📁 personality/        # Personality data
    │   │   ├── big_five_questions.json
    │   │   └── diagnosis_questions.json
    │   ├── 📁 personality_assessment/  # Assessment data
    │   │   └── test_user_metadata.json
    │   ├── 📁 therapy_resource/   # Therapy resources
    │   │   └── test_user_metadata.json
    │   ├── 📁 user_profile/       # User profiles
    │   │   └── test_user_metadata.json
    │   └── 📁 vector_store/       # Vector database
    │       ├── cache.json
    │       └── documents.json
    │
    ├── main.py                    # Main application entry
    └── __init__.py
```

---

## 3. MODULE DEPENDENCY MAP

### **Layer Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  api_server.py (FastAPI)                                 │   │
│  │  - 30+ REST endpoints                                    │   │
│  │  - JWT authentication                                    │   │
│  │  - Rate limiting                                         │   │
│  │  - Security middleware                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  src/main.py                                             │   │
│  │  - Application initialization                            │   │
│  │  - Module manager                                        │   │
│  │  - Device detection (CPU/GPU)                           │   │
│  │  - Performance profiling                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  agents/orchestration/agent_orchestrator.py (2,382 lines)│   │
│  │  - Workflow management (12 predefined workflows)         │   │
│  │  - Message bus (event-driven)                            │   │
│  │  - Circuit breaker                                       │   │
│  │  - Context management                                    │   │
│  │  - Validator registry                                    │   │
│  │  - Performance monitoring                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  agents/orchestration/supervisor_agent.py (917 lines)    │   │
│  │  - Quality assurance                                     │   │
│  │  - Ethics validation                                     │   │
│  │  - Clinical risk assessment                              │   │
│  │  - Content validation                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐   │
│  │ Emotion  │ Safety   │ Chat     │ Therapy  │ Personality  │   │
│  │  Agent   │  Agent   │  Agent   │  Agent   │   Agent      │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Clinical Agents                                         │   │
│  │  - diagnosis_agent.py (LEGACY - not used in main flow)  │   │
│  │  - therapy_agent.py                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Therapeutic Friction Agents                             │   │
│  │  - breakthrough_detection_agent.py                       │   │
│  │  - friction_coordinator.py                               │   │
│  │  - readiness_assessment_agent.py                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Support Agents                                          │   │
│  │  - search_agent.py                                       │   │
│  │  - crawler_agent.py                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Diagnosis Services (⚠️ DUPLICATION ISSUE)              │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │ services/diagnosis/unified_service.py              │ │   │
│  │  │ - IDiagnosisService implementation                 │ │   │
│  │  │ - Orchestrates multiple diagnosis backends         │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │ diagnosis/comprehensive_diagnosis.py (1,452 lines) │ │   │
│  │  │ diagnosis/enhanced_diagnosis.py (1,436 lines)      │ │   │
│  │  │ diagnosis/differential_diagnosis.py (1,366 lines)  │ │   │
│  │  │ diagnosis/enterprise_multimodal_pipeline.py        │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Memory Services                                         │   │
│  │  - memory/enhanced_memory_system.py (1,118 lines)      │   │
│  │  - memory/semantic_memory/semantic_memory_manager.py   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LLM Services                                            │   │
│  │  - models/llm.py (base interface)                       │   │
│  │  - models/gemini_llm.py                                 │   │
│  │  - providers/llm/gemini_provider.py                     │   │
│  │  - providers/llm/openai_provider.py                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Knowledge Services                                      │   │
│  │  - knowledge/therapeutic/knowledge_base.py              │   │
│  │  - knowledge/clinical/clinical_guidelines_db.py         │   │
│  │  - clinical_decision_support/*                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Dependency Injection                                    │   │
│  │  - infrastructure/di/container.py                        │   │
│  │  - infrastructure/di/diagnosis_registration.py          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Data Persistence                                        │   │
│  │  - database/central_vector_db.py (ChromaDB)            │   │
│  │  - database/conversation_tracker.py                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Configuration                                           │   │
│  │  - config/settings.py (AppConfig)                       │   │
│  │  - config/security.py (SecurityConfig)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Logging & Monitoring                                    │   │
│  │  - utils/logger.py                                       │   │
│  │  - monitoring/health_monitor.py                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### **Key Dependencies**

| Module | Depends On | Purpose |
|--------|-----------|---------|
| `api_server.py` | `main.py`, `auth/*`, `middleware/*` | REST API server |
| `main.py` | `config/settings.py`, `components/*`, `utils/*` | Application bootstrap |
| `agent_orchestrator.py` | All agents, `services/diagnosis/*`, `database/*` | Agent coordination |
| `base_agent.py` | `utils/memory_factory.py`, `security/*` | Agent base class |
| All agents | `models/llm.py`, `config/settings.py` | LLM integration |
| `enhanced_memory_system.py` | `database/central_vector_db.py`, `models/llm.py` | Memory management |
| `diagnosis/comprehensive_diagnosis.py` | `memory/*`, `database/*`, `utils/*` | Diagnosis logic |
| `services/diagnosis/unified_service.py` | All diagnosis implementations | Diagnosis facade |

---

## 4. AGENT SYSTEM ARCHITECTURE

### **Agent Hierarchy**

```
┌───────────────────────────────────────────────────────────────────────┐
│                         BASE AGENT                                     │
│  agents/base/base_agent.py                                            │
│  - Extends Agno Agent framework                                       │
│  - Memory factory integration                                         │
│  - Security validation (optional - ⚠️ ISSUE)                         │
│  - Process method with context management                             │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ inherits
                                  ↓
┌───────────────────────────────────────────────────────────────────────┐
│                        SPECIALIZED AGENTS                              │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  CORE AGENTS (agents/core/)                                     │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  EmotionAgent (emotion_agent.py)                          │ │ │
│  │  │  - Sentiment analysis (TextBlob, transformers)            │ │ │
│  │  │  - Emotion classification (joy, sadness, anger, fear)     │ │ │
│  │  │  - Empathetic response generation                         │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  SafetyAgent (safety_agent.py)                            │ │ │
│  │  │  - Crisis keyword detection                               │ │ │
│  │  │  - Suicide/self-harm risk assessment                      │ │ │
│  │  │  - Crisis resource provision                              │ │ │
│  │  │  - Escalation protocol                                    │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  ChatAgent (chat_agent.py)                                │ │ │
│  │  │  - Conversational flow management                         │ │ │
│  │  │  - Context-aware responses                                │ │ │
│  │  │  - Personality integration                                │ │ │
│  │  │  - LLM interaction                                        │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  PersonalityAgent (personality_agent.py)                  │ │ │
│  │  │  - Big Five trait analysis                                │ │ │
│  │  │  - MBTI type assessment                                   │ │ │
│  │  │  - Communication style adaptation                         │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  CLINICAL AGENTS (agents/clinical/)                             │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  TherapyAgent (therapy_agent.py)                          │ │ │
│  │  │  - CBT technique application                              │ │ │
│  │  │  - Mindfulness exercises                                  │ │ │
│  │  │  - Solution-focused brief therapy                         │ │ │
│  │  │  - Motivational interviewing                              │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  DiagnosisAgent (diagnosis_agent.py) - LEGACY             │ │ │
│  │  │  ⚠️ Not used in main workflows                           │ │ │
│  │  │  - Replaced by services/diagnosis/*                       │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  THERAPEUTIC FRICTION AGENTS (agents/therapeutic_friction/)     │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  BreakthroughDetectionAgent                               │ │ │
│  │  │  - Detects therapeutic breakthroughs                      │ │ │
│  │  │  - Identifies insight moments                             │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  FrictionCoordinator                                      │ │ │
│  │  │  - Manages therapeutic resistance                         │ │ │
│  │  │  - Coordinates friction agents                            │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  ReadinessAssessmentAgent                                 │ │ │
│  │  │  - Assesses user readiness for change                     │ │ │
│  │  │  - Stages of change model                                 │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  SUPPORT AGENTS (agents/support/)                               │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  SearchAgent (search_agent.py)                            │ │ │
│  │  │  - Web search for mental health resources                 │ │ │
│  │  │  - Evidence-based information retrieval                   │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │  ┌───────────────────────────────────────────────────────────┐ │ │
│  │  │  CrawlerAgent (crawler_agent.py)                          │ │ │
│  │  │  - Crawls trusted mental health websites                  │ │ │
│  │  │  - Updates knowledge base                                 │ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ supervised by
                                  ↓
┌───────────────────────────────────────────────────────────────────────┐
│                       SUPERVISOR AGENT                                 │
│  agents/orchestration/supervisor_agent.py (917 lines)                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Validation Levels                                              │ │
│  │  - PASS: Response meets all standards                           │ │
│  │  - WARNING: Minor issues detected                              │ │
│  │  - CRITICAL: Significant problems                              │ │
│  │  - BLOCKED: Response must be rejected                          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Validation Types                                               │ │
│  │  1. Content Validation (harmful content, boundary violations)   │ │
│  │  2. Clinical Risk Assessment (5 risk levels)                    │ │
│  │  3. Ethical Concerns (6 concern types)                          │ │
│  │  4. Response Quality (coherence, relevance, empathy)            │ │
│  │  5. Therapeutic Alignment (evidence-based practices)            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ⚠️ ISSUES:                                                          │
│  - Regex-based validation (limited)                                  │
│  - Simple sentiment analyzer (not production-ready)                  │
└───────────────────────────────────────────────────────────────────────┘
```

### **Agent Communication Flow**

```
User Message
     │
     ↓
┌────────────────────────────────────────────────┐
│  Agent Orchestrator                            │
│  1. Receives message                           │
│  2. Loads context from vector DB               │
│  3. Selects workflow ("enhanced_empathetic")   │
│  4. Initializes agent sequence                 │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Safety Agent                                  │
│  - Crisis detection                            │
│  - Risk assessment                             │
│  - If high risk → immediate intervention       │
│  - Else → continue workflow                    │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Emotion Agent                                 │
│  - Sentiment analysis                          │
│  - Emotion classification                      │
│  - Context: emotional state                    │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Personality Agent                             │
│  - Retrieves user personality profile          │
│  - Adapts communication style                  │
│  - Context: personality traits                 │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Diagnosis Service (if assessment needed)      │
│  - Comprehensive mental health assessment      │
│  - PHQ-9, GAD-7 scoring                        │
│  - Symptom analysis                            │
│  - Context: diagnosis insights                 │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Therapy Agent                                 │
│  - Selects therapeutic approach                │
│  - Applies techniques (CBT, mindfulness, etc.) │
│  - Context: therapeutic strategy               │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Chat Agent                                    │
│  - Generates response using LLM                │
│  - Integrates all context                      │
│  - Produces empathetic response                │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Supervisor Agent                              │
│  - Validates response quality                  │
│  - Checks clinical risk                        │
│  - Verifies ethical compliance                 │
│  - If BLOCKED → regenerate                     │
│  - If PASS → continue                          │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Memory System                                 │
│  - Store conversation turn                     │
│  - Extract therapeutic insights                │
│  - Update user profile                         │
│  - Update vector database                      │
└────────────────────────────────────────────────┘
     │
     ↓
┌────────────────────────────────────────────────┐
│  Return Response to User                       │
│  - Text response                               │
│  - Emotion metadata                            │
│  - Suggestions/recommendations                 │
└────────────────────────────────────────────────┘
```

---

## 5. DATA FLOW DIAGRAMS

### **5.1 User Message Processing Flow**

```
┌─────────────┐
│ User Input  │
│ (Text/Voice)│
└──────┬──────┘
       │
       ↓
┌──────────────────────────────────┐
│ Input Processing                 │
│ - Voice → Text (Whisper ASR)     │
│ - Security validation            │
│ - Input sanitization             │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Context Loading                  │
│ - Vector DB query                │
│ - Load user profile              │
│ - Load conversation history      │
│ - Load personality data          │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Agent Orchestrator               │
│ - Select workflow                │
│ - Initialize agent sequence      │
│ - Manage message bus             │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Agent Execution (Sequential)     │
│ 1. Safety check                  │
│ 2. Emotion analysis              │
│ 3. Personality adaptation        │
│ 4. Diagnosis (if needed)         │
│ 5. Therapy technique selection   │
│ 6. Response generation           │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Supervisor Validation            │
│ - Quality check                  │
│ - Risk assessment                │
│ - Ethics verification            │
│ - If blocked → retry             │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Memory Storage                   │
│ - Store conversation             │
│ - Update therapeutic insights    │
│ - Update emotion tracking        │
│ - Vector DB embedding            │
└──────┬───────────────────────────┘
       │
       ↓
┌──────────────────────────────────┐
│ Response Delivery                │
│ - Text response                  │
│ - Voice synthesis (TTS)          │
│ - Emotion metadata               │
│ - Recommendations                │
└──────┬───────────────────────────┘
       │
       ↓
┌─────────────┐
│ User Output │
└─────────────┘
```

### **5.2 Diagnosis Flow (⚠️ Current Implementation)**

```
User requests assessment
       │
       ↓
┌──────────────────────────────────────────┐
│ Agent Orchestrator                       │
│ - Detects diagnosis intent               │
│ - ⚠️ UNCLEAR which implementation to use│
└──────┬───────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────┐
│ Multiple Diagnosis Paths (ISSUE!)       │
│                                          │
│ Path 1: services/diagnosis/              │
│   unified_service.py                     │
│   ↓                                      │
│   orchestrator.py                        │
│   ↓                                      │
│   comprehensive_diagnosis.py             │
│                                          │
│ Path 2: diagnosis/                       │
│   enhanced_diagnosis.py (standalone)     │
│                                          │
│ Path 3: diagnosis/                       │
│   enterprise_multimodal_pipeline.py      │
│                                          │
│ Path 4: agents/clinical/                 │
│   diagnosis_agent.py (LEGACY)            │
│                                          │
│ ⚠️ No clear selection logic!            │
└──────┬───────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────┐
│ Diagnosis Processing (varies by path)   │
│ - Symptom extraction                     │
│ - PHQ-9/GAD-7 scoring                    │
│ - Condition matching                     │
│ - Confidence scoring                     │
│ - Voice emotion analysis (some paths)    │
│ - Cultural sensitivity (some paths)      │
│ - Temporal analysis (some paths)         │
└──────┬───────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────┐
│ Memory Integration                       │
│ - Store diagnosis insights               │
│ - Update user profile                    │
│ - Vector DB storage                      │
│ - ⚠️ Some paths skip this!              │
└──────┬───────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────┐
│ Return Diagnosis Result                  │
│ - Primary diagnosis                      │
│ - Confidence level                       │
│ - Recommendations                        │
│ - Treatment suggestions                  │
└──────────────────────────────────────────┘
```

### **5.3 Memory System Flow**

```
┌─────────────────────────────────────────────┐
│ Conversation Turn                           │
│ - User message + agent responses            │
└──────┬──────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────┐
│ Enhanced Memory System                       │
│ (memory/enhanced_memory_system.py)           │
│                                              │
│ ┌──────────────────────────────────────────┐ │
│ │ Therapeutic Insight Extraction           │ │
│ │ - Breakthrough moments                   │ │
│ │ - Coping mechanisms                      │ │
│ │ - Emotional patterns                     │ │
│ │ - Cognitive distortions                  │ │
│ │ - Support systems                        │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ ┌──────────────────────────────────────────┐ │
│ │ Progress Milestone Detection             │ │
│ │ - Improvement indicators                 │ │
│ │ - Setback patterns                       │ │
│ │ - Skill acquisition                      │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ ┌──────────────────────────────────────────┐ │
│ │ Session Continuity Context               │ │
│ │ - Previous session summary               │ │
│ │ - Open issues                            │ │
│ │ - Action items                           │ │
│ └──────────────────────────────────────────┘ │
└──────┬───────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────┐
│ Vector Database Integration                  │
│ (database/central_vector_db.py)              │
│                                              │
│ ┌──────────────────────────────────────────┐ │
│ │ Embedding Generation                     │ │
│ │ - Text → Vector (sentence-transformers)  │ │
│ │ - Namespace selection                    │ │
│ └──────────────────────────────────────────┘ │
│                                              │
│ ┌──────────────────────────────────────────┐ │
│ │ Storage                                  │ │
│ │ - user_profile collection                │ │
│ │ - conversation collection                │ │
│ │ - diagnostic_data collection             │ │
│ │ - therapy_resource collection            │ │
│ │ - emotion_record collection              │ │
│ └──────────────────────────────────────────┘ │
└──────┬───────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────┐
│ Persistence Layer                            │
│ ⚠️ ISSUE: Uses pickle (security risk)       │
│                                              │
│ Pickle files in src/data/memory_system/:    │
│ - therapeutic_insights.pkl                   │
│ - progress_milestones.pkl                    │
│ - session_continuity.pkl                     │
│ - recurring_themes.pkl                       │
└──────────────────────────────────────────────┘
```

---

## 6. API ENDPOINTS MAP

### **API Structure** (api_server.py)

```
FastAPI Application
├── Security Middleware (in order)
│   ├── 1. SecurityHeadersMiddleware
│   ├── 2. RequestLoggingMiddleware
│   ├── 3. ContentTypeValidationMiddleware
│   ├── 4. IPFilterMiddleware
│   ├── 5. SlowAPIMiddleware (rate limiting)
│   └── 6. CORSMiddleware
│
├── Authentication Endpoints
│   ├── POST /auth/register
│   ├── POST /auth/login
│   ├── POST /auth/refresh
│   ├── POST /auth/logout
│   ├── POST /auth/password-reset
│   ├── POST /auth/password-reset/confirm
│   └── PUT /auth/password-change
│
├── Chat Endpoints
│   ├── POST /chat/message
│   │   - Body: ChatRequestSecure (message, user_id, session_id)
│   │   - Auth: Required (JWT)
│   │   - Returns: ChatResponse (response, emotion, metadata)
│   │   - Rate limit: 10/minute
│   │
│   └── GET /chat/history/{user_id}
│       - Auth: Required
│       - Returns: List of conversation history
│       - Rate limit: 20/minute
│
├── Assessment Endpoints
│   ├── POST /assessment/start
│   │   - Start new assessment session
│   │   - Auth: Required
│   │   - Returns: Assessment session ID
│   │
│   ├── POST /assessment/question
│   │   - Submit answer to assessment question
│   │   - Body: AssessmentQuestionResponse
│   │   - Auth: Required
│   │
│   ├── POST /assessment/complete
│   │   - Complete assessment and get results
│   │   - Auth: Required
│   │   - Returns: Diagnosis result
│   │
│   └── GET /assessment/history/{user_id}
│       - Get past assessment results
│       - Auth: Required
│
├── Voice Endpoints
│   ├── POST /voice/process
│   │   - Upload: Audio file (WAV, MP3)
│   │   - Auth: Required
│   │   - Returns: Transcription + emotion
│   │   - Rate limit: 5/minute
│   │
│   └── POST /voice/synthesize
│       - Body: Text to synthesize
│       - Auth: Required
│       - Returns: Audio file
│
├── User Profile Endpoints
│   ├── GET /profile/{user_id}
│   │   - Auth: Required (self or admin)
│   │   - Returns: User profile data
│   │
│   ├── PUT /profile/{user_id}
│   │   - Update user profile
│   │   - Auth: Required (self or admin)
│   │   - Body: UserUpdate
│   │
│   └── GET /profile/{user_id}/insights
│       - Get therapeutic insights
│       - Auth: Required (self or therapist)
│       - Returns: Insights, progress, patterns
│
├── Supervision Endpoints (Admin only)
│   ├── GET /supervision/status
│   │   - Get supervision system status
│   │   - Auth: Admin required
│   │   - Returns: SupervisionStatusResponse
│   │
│   ├── GET /supervision/summary
│   │   - Get supervision summary
│   │   - Auth: Admin required
│   │   - Query: time_window_hours
│   │   - Returns: Metrics, audits, quality data
│   │
│   ├── GET /supervision/agent-quality/{agent_name}
│   │   - Get agent quality report
│   │   - Auth: Admin required
│   │   - Returns: Performance metrics
│   │
│   └── GET /supervision/audit-trail
│       - Get audit trail
│       - Auth: Admin required
│       - Query: start_time, end_time
│
├── Admin Endpoints
│   ├── GET /admin/users
│   │   - List all users
│   │   - Auth: Admin required
│   │
│   ├── PUT /admin/users/{user_id}/role
│   │   - Update user role
│   │   - Auth: Admin required
│   │
│   └── GET /admin/metrics
│       - System metrics
│       - Auth: Admin required
│
└── Health Endpoints
    ├── GET /health
    │   - Basic health check
    │   - No auth required
    │
    └── GET /health/detailed
        - Detailed system health
        - Auth: Admin required
        - Returns: All service statuses
```

### **API Request/Response Models**

| Model | Fields | Purpose |
|-------|--------|---------|
| `UserCreate` | username, email, password, role | User registration |
| `UserLogin` | username, password | Authentication |
| `Token` | access_token, token_type, refresh_token | JWT tokens |
| `ChatRequestSecure` | message, user_id, session_id, context | Chat message |
| `ChatResponse` | response, emotion, metadata, timestamp | Chat response |
| `DiagnosticAssessmentRequestSecure` | responses, user_id, assessment_type | Assessment |
| `SupervisionStatusResponse` | supervision_enabled, metrics, status | Supervision status |
| `AgentQualityReportResponse` | agent_name, performance_summary | Agent quality |

---

## 7. CONFIGURATION STRUCTURE

### **Configuration Files**

```
Configuration Hierarchy
│
├── 📄 .env (Root - NOT in git)
│   ├── GEMINI_API_KEY=xxx
│   ├── OPENAI_API_KEY=xxx
│   ├── MODEL_NAME=gemini-1.5-pro
│   ├── EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
│   ├── LLM_PROVIDER=gemini
│   ├── DEBUG=False
│   ├── LOG_LEVEL=INFO
│   ├── USER_ID=default_user
│   └── ... (secrets)
│
├── 📄 src/config/settings.py
│   │
│   ├── AppConfig (Main Configuration Class)
│   │   ├── APP_NAME = "Mental Health Support Bot"
│   │   ├── APP_VERSION = "1.0.0"
│   │   ├── DEBUG (from env)
│   │   │
│   │   ├── Paths
│   │   │   ├── BASE_DIR = src/
│   │   │   ├── DATA_DIR = src/data/
│   │   │   ├── MODEL_DIR = src/models/
│   │   │   └── VECTOR_STORE_PATH = src/data/vector_store/
│   │   │
│   │   ├── LLM_CONFIG
│   │   │   ├── provider (gemini/openai)
│   │   │   ├── model (from env)
│   │   │   ├── api_key (from env)
│   │   │   ├── temperature (0.7)
│   │   │   ├── top_p (0.9)
│   │   │   ├── top_k (50)
│   │   │   └── max_tokens (2000)
│   │   │
│   │   ├── VECTOR_DB_CONFIG
│   │   │   ├── engine: "faiss"
│   │   │   ├── dimension: 768
│   │   │   ├── index_type: "L2"
│   │   │   ├── metric_type: "cosine"
│   │   │   ├── retention_days: 180
│   │   │   └── namespaces: [user_profile, conversation,
│   │   │       knowledge, therapy_resource, diagnostic_data,
│   │   │       personality_assessment, emotion_record]
│   │   │
│   │   ├── SAFETY_CONFIG
│   │   │   ├── max_toxicity: 0.7
│   │   │   ├── blocked_categories: [harmful, unsafe, toxic...]
│   │   │   ├── content_filters: {profanity, personal_info...}
│   │   │   └── fallback_responses: {...}
│   │   │
│   │   ├── PERSONALITY_CONFIG
│   │   │   ├── big_five: {enabled, num_questions, traits}
│   │   │   └── mbti: {enabled, num_questions, dimensions}
│   │   │
│   │   ├── VOICE_CONFIG
│   │   │   ├── stt_model (from env)
│   │   │   ├── tts_model (from env)
│   │   │   ├── use_gpu (True)
│   │   │   └── voice_styles: {default, male, female, warm}
│   │   │
│   │   ├── Assessment Questions
│   │   │   ├── ASSESSMENT_QUESTIONS (general)
│   │   │   ├── PHQ9_QUESTIONS (depression - 9 questions)
│   │   │   └── GAD7_QUESTIONS (anxiety - 7 questions)
│   │   │
│   │   └── CRISIS_RESOURCES
│   │       - National Crisis Hotline: 988
│   │       - Emergency Services: 911
│   │       - Crisis Text Line: HOME to 741741
│   │
│   └── Methods
│       ├── get_vector_store_config()
│       ├── get_crawler_config()
│       ├── get_model_config()
│       ├── get_optimized_model_config(agent_name)
│       ├── validate_config()
│       ├── validate_security()
│       └── require_secure_config()
│
├── 📄 src/config/security.py
│   └── SecurityConfig
│       ├── JWT_SECRET_KEY (from env)
│       ├── JWT_ALGORITHM = "HS256"
│       ├── ACCESS_TOKEN_EXPIRE_MINUTES = 30
│       ├── REFRESH_TOKEN_EXPIRE_DAYS = 7
│       ├── ALLOWED_ORIGINS (CORS)
│       ├── RATE_LIMITS
│       └── is_development()
│
├── 📄 src/config/supervision_config.py
│   └── Supervision settings
│       ├── Validation thresholds
│       ├── Risk assessment levels
│       └── Audit trail settings
│
├── 📄 src/config/feature_flags.py
│   └── Feature toggles
│       ├── SUPERVISION_ENABLED
│       ├── DIAGNOSIS_ENHANCED_MODE
│       ├── VOICE_ENABLED
│       └── ENTERPRISE_FEATURES
│
└── 📄 src/config/optimization_config.py
    └── Performance settings
        ├── PROFILING_ENABLED
        ├── CACHE_SETTINGS
        └── BATCH_SIZES
```

### **Agent-Specific Configuration**

```python
# Agent configuration resolution
AppConfig.get_optimized_model_config(agent_name)

# Critical agents → Full model
- chat_agent       → gemini-1.5-pro (temp: 0.7)
- therapy_agent    → gemini-1.5-pro (temp: 0.7)
- diagnosis_agent  → gemini-1.5-pro (temp: 0.3)

# Standard agents → Standard model
- emotion_agent    → gemini-1.5-pro (temp: 0.5)
- personality_agent→ gemini-1.5-pro (temp: 0.5)
- safety_agent     → gemini-1.5-pro (temp: 0.3)

# Support agents → Lighter model (cost optimization)
- search_agent     → gemini-1.5-flash (temp: 0.5)
- crawler_agent    → gemini-1.5-flash (temp: 0.5)
```

---

## 8. INTEGRATION POINTS

### **8.1 External Integrations**

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                         │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  LLM Providers                                        │  │
│  │  ┌──────────────────┐  ┌──────────────────┐          │  │
│  │  │ Google Gemini    │  │ OpenAI GPT       │          │  │
│  │  │ - Gemini 1.5 Pro │  │ - GPT-4, GPT-3.5 │          │  │
│  │  │ - API key auth   │  │ - API key auth   │          │  │
│  │  └────────┬─────────┘  └────────┬─────────┘          │  │
│  │           │                      │                    │  │
│  │           └──────────┬───────────┘                    │  │
│  │                      ↓                                │  │
│  │           ┌──────────────────────┐                    │  │
│  │           │ models/llm.py        │                    │  │
│  │           │ - Abstract interface │                    │  │
│  │           │ - Provider factory   │                    │  │
│  │           └──────────────────────┘                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Voice Services                                       │  │
│  │  ┌──────────────────┐  ┌──────────────────┐          │  │
│  │  │ Whisper V3 Turbo │  │ TTS Engine       │          │  │
│  │  │ - Speech-to-text │  │ - Text-to-speech │          │  │
│  │  │ - Multi-language │  │ - Voice styles   │          │  │
│  │  └────────┬─────────┘  └────────┬─────────┘          │  │
│  │           │                      │                    │  │
│  │           └──────────┬───────────┘                    │  │
│  │                      ↓                                │  │
│  │           ┌──────────────────────┐                    │  │
│  │           │ utils/whisper_asr.py │                    │  │
│  │           │ utils/dia_tts.py     │                    │  │
│  │           └──────────────────────┘                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Vector Database                                      │  │
│  │  ┌──────────────────┐                                 │  │
│  │  │ ChromaDB         │                                 │  │
│  │  │ - Embeddings     │                                 │  │
│  │  │ - Semantic search│                                 │  │
│  │  └────────┬─────────┘                                 │  │
│  │           │                                           │  │
│  │           ↓                                           │  │
│  │  ┌──────────────────────┐                            │  │
│  │  │ database/            │                            │  │
│  │  │ central_vector_db.py │                            │  │
│  │  └──────────────────────┘                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### **8.2 Internal Integration Points**

```
┌─────────────────────────────────────────────────────────────┐
│                 DEPENDENCY INJECTION                         │
│                                                              │
│  infrastructure/di/container.py                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  DIContainer                                          │  │
│  │  - Service registration (transient, singleton, scoped)│  │
│  │  - Automatic dependency resolution                    │  │
│  │  - Lifecycle management                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Registered Services:                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  IDiagnosisOrchestrator → DiagnosisOrchestrator       │  │
│  │  IDiagnosisService → UnifiedDiagnosisService          │  │
│  │  IMemoryService → EnhancedMemorySystem                │  │
│  │  IVectorDatabase → CentralVectorDB                    │  │
│  │  ILLM → GeminiLLM / OpenAILLM                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Usage:                                                      │
│  from infrastructure.di.container import get_container      │
│  container = get_container()                                │
│  diagnosis_service = container.resolve(IDiagnosisService)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     EVENT BUS                                │
│                                                              │
│  integration/event_bus.py                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MessageBus                                           │  │
│  │  - Publish/subscribe pattern                          │  │
│  │  - Event types: agent_started, agent_completed,       │  │
│  │    validation_failed, diagnosis_complete, etc.        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Subscribers:                                                │
│  - SupervisorAgent (all agent events)                        │
│  - PerformanceMonitor (performance events)                   │
│  - AuditSystem (security events)                             │
│  - MemorySystem (conversation events)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   MEMORY FACTORY                             │
│                                                              │
│  utils/memory_factory.py                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  get_or_create_memory(memory)                         │  │
│  │  - Creates memory instances                            │  │
│  │  - Ensures singleton per user                         │  │
│  │  - Integrates with vector DB                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ⚠️ ISSUE: Not consistently used across all agents         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               VECTOR DB INTEGRATION                          │
│                                                              │
│  utils/vector_db_integration.py                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Helper Functions                                     │  │
│  │  - search_relevant_data(query, namespaces, limit)     │  │
│  │  - add_data_to_vector_db(data, namespace, user_id)   │  │
│  │  - get_conversation_tracker()                         │  │
│  │  - get_user_data(user_id)                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Used by:                                                    │
│  - Agent Orchestrator (context loading)                      │
│  - Memory System (insight storage)                           │
│  - Diagnosis Services (historical data)                      │
│  - Therapeutic Friction (pattern detection)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. MEMORY ARCHITECTURE

### **Memory Hierarchy**

```
┌───────────────────────────────────────────────────────────────┐
│                  MEMORY ARCHITECTURE                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 1: Short-Term Memory (Conversation Context)     │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  ConversationMemory                               │  │  │
│  │  │  - Last N conversation turns (configurable)       │  │  │
│  │  │  - Current session context                        │  │  │
│  │  │  - Working memory for immediate responses         │  │  │
│  │  │  - Retention: Session lifetime                    │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 2: Enhanced Memory System (Therapeutic Insights)│  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  EnhancedMemorySystem                             │  │  │
│  │  │  (memory/enhanced_memory_system.py - 1,118 lines) │  │  │
│  │  │                                                   │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ Therapeutic Insights                        │ │  │  │
│  │  │  │ - Breakthrough moments                      │ │  │  │
│  │  │  │ - Coping mechanisms discovered              │ │  │  │
│  │  │  │ - Emotional patterns identified             │ │  │  │
│  │  │  │ - Cognitive distortions detected            │ │  │  │
│  │  │  │ - Support systems recognized                │ │  │  │
│  │  │  │ - Retention: 365 days                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │                                                   │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ Progress Milestones                         │ │  │  │
│  │  │  │ - Improvement indicators                    │ │  │  │
│  │  │  │ - Setback tracking                          │ │  │  │
│  │  │  │ - Skill acquisition markers                 │ │  │  │
│  │  │  │ - Goal achievement tracking                 │ │  │  │
│  │  │  │ - Retention: 365 days                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │                                                   │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ Session Continuity Context                  │ │  │  │
│  │  │  │ - Previous session summary                  │ │  │  │
│  │  │  │ - Open issues/unresolved topics             │ │  │  │
│  │  │  │ - Homework/action items                     │ │  │  │
│  │  │  │ - Follow-up reminders                       │ │  │  │
│  │  │  │ - Retention: 180 days                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │                                                   │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ Recurring Themes                            │ │  │  │
│  │  │  │ - Identified patterns                       │ │  │  │
│  │  │  │ - Trigger identification                    │ │  │  │
│  │  │  │ - Response patterns                         │ │  │  │
│  │  │  │ - Retention: 365 days                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │                                                   │  │  │
│  │  │  Storage: ⚠️ Pickle files (SECURITY ISSUE)      │  │  │
│  │  │  - src/data/memory_system/*.pkl                  │  │  │
│  │  │  - Unencrypted                                    │  │  │
│  │  │  - No schema versioning                           │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 3: Semantic Memory (Long-Term Knowledge)        │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  SemanticMemoryManager                            │  │  │
│  │  │  (memory/semantic_memory/semantic_memory_manager) │  │  │
│  │  │                                                   │  │  │
│  │  │  - Abstract concepts and knowledge                │  │  │
│  │  │  - User beliefs and values                        │  │  │
│  │  │  - Life narrative elements                        │  │  │
│  │  │  - Long-term goals and aspirations                │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 4: Vector Database (Persistent Storage)         │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  CentralVectorDB (ChromaDB)                       │  │  │
│  │  │  (database/central_vector_db.py)                  │  │  │
│  │  │                                                   │  │  │
│  │  │  Namespaces (Collections):                        │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 1. user_profile                             │ │  │  │
│  │  │  │    - Demographics, preferences              │ │  │  │
│  │  │  │    - Personality assessment results         │ │  │  │
│  │  │  │    - User preferences                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 2. conversation                             │ │  │  │
│  │  │  │    - Full conversation history              │ │  │  │
│  │  │  │    - Message embeddings                     │ │  │  │
│  │  │  │    - Context vectors                        │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 3. diagnostic_data                          │ │  │  │
│  │  │  │    - Assessment results (PHQ-9, GAD-7)      │ │  │  │
│  │  │  │    - Diagnosis history                      │ │  │  │
│  │  │  │    - Symptom tracking                       │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 4. therapy_resource                         │ │  │  │
│  │  │  │    - Therapeutic techniques used            │ │  │  │
│  │  │  │    - Resources provided                     │ │  │  │
│  │  │  │    - Homework assignments                   │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 5. personality_assessment                   │ │  │  │
│  │  │  │    - Big Five scores                        │ │  │  │
│  │  │  │    - MBTI type                              │ │  │  │
│  │  │  │    - Personality evolution over time        │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 6. emotion_record                           │ │  │  │
│  │  │  │    - Emotional states over time             │ │  │  │
│  │  │  │    - Sentiment trends                       │ │  │  │
│  │  │  │    - Emotional triggers                     │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │  ┌─────────────────────────────────────────────┐ │  │  │
│  │  │  │ 7. knowledge                                │ │  │  │
│  │  │  │    - Mental health knowledge base           │ │  │  │
│  │  │  │    - Clinical guidelines                    │ │  │  │
│  │  │  │    - Evidence-based resources               │ │  │  │
│  │  │  └─────────────────────────────────────────────┘ │  │  │
│  │  │                                                   │  │  │
│  │  │  Configuration:                                   │  │  │
│  │  │  - Embedding dimension: 768                       │  │  │
│  │  │  - Metric: Cosine similarity                      │  │  │
│  │  │  - Retention: 180 days (configurable)             │  │  │
│  │  │  - Index: FAISS (fast approximate search)         │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### **Memory Retrieval Flow**

```
User asks question
       │
       ↓
┌──────────────────────────────────────┐
│ Agent Orchestrator                   │
│ - Receives message                   │
│ - Needs context                      │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│ Query Preparation                    │
│ - Extract query embedding            │
│ - Determine relevant namespaces      │
│ - Set similarity threshold           │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│ Vector DB Query                      │
│ - Search user_profile (user info)    │
│ - Search conversation (past talks)   │
│ - Search diagnostic_data (symptoms)  │
│ - Search emotion_record (patterns)   │
│ - Return top K similar results       │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│ Context Assembly                     │
│ - Combine results from all namespaces│
│ - Rank by relevance                  │
│ - Add session continuity context     │
│ - Add therapeutic insights           │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│ Provide to Agent                     │
│ - Enriched context object            │
│ - Agent uses for informed response   │
└──────────────────────────────────────┘
```

---

## 10. SECURITY & COMPLIANCE

### **Security Layers**

```
┌───────────────────────────────────────────────────────────────┐
│                      SECURITY LAYERS                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 1: API Security (middleware/security.py)         │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  SecurityHeadersMiddleware                        │  │  │
│  │  │  - X-Content-Type-Options: nosniff                │  │  │
│  │  │  - X-Frame-Options: DENY                          │  │  │
│  │  │  - X-XSS-Protection: 1; mode=block                │  │  │
│  │  │  - Strict-Transport-Security (HSTS)               │  │  │
│  │  │  - Content-Security-Policy                        │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  RequestLoggingMiddleware                         │  │  │
│  │  │  - Logs all requests for audit                    │  │  │
│  │  │  - Redacts sensitive information                  │  │  │
│  │  │  - Tracks request IDs                             │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  ContentTypeValidationMiddleware                  │  │  │
│  │  │  - Validates Content-Type headers                 │  │  │
│  │  │  - Rejects suspicious content types               │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  IPFilterMiddleware                               │  │  │
│  │  │  - IP whitelist/blacklist for admin endpoints     │  │  │
│  │  │  - Geolocation filtering                          │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  SlowAPIMiddleware (Rate Limiting)                │  │  │
│  │  │  - Per-endpoint rate limits                       │  │  │
│  │  │  - Per-user rate limits                           │  │  │
│  │  │  - Prevents DoS attacks                           │  │  │
│  │  │  - Chat: 10/min, Voice: 5/min, etc.               │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 2: Authentication (auth/)                        │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  JWT Token System (auth/jwt_utils.py)            │  │  │
│  │  │  - Access tokens (30 min expiry)                  │  │  │
│  │  │  - Refresh tokens (7 day expiry)                  │  │  │
│  │  │  - HS256 algorithm                                │  │  │
│  │  │  - Token revocation support                       │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  Role-Based Access Control (auth/dependencies.py)│  │  │
│  │  │  - User roles: user, therapist, admin             │  │  │
│  │  │  - Endpoint-level permissions                     │  │  │
│  │  │  - require_admin()                                │  │  │
│  │  │  - require_therapist_or_admin()                   │  │  │
│  │  │  - require_chat_access()                          │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  Password Security                                │  │  │
│  │  │  - Bcrypt hashing                                 │  │  │
│  │  │  - Password reset flow                            │  │  │
│  │  │  - Password complexity requirements               │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 3: Input Validation (security/input_validator.py)│  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  InputValidator                                   │  │  │
│  │  │  - SQL injection detection                        │  │  │
│  │  │  - XSS (cross-site scripting) detection           │  │  │
│  │  │  - Command injection detection                    │  │  │
│  │  │  - Path traversal detection                       │  │  │
│  │  │  - LDAP injection detection                       │  │  │
│  │  │  - XML injection detection                        │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ⚠️ ISSUE: Optional in base_agent.py - can be skipped  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 4: HIPAA Compliance (compliance/hipaa_validator)│  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  PHIDetector (Protected Health Information)       │  │  │
│  │  │  - SSN detection                                  │  │  │
│  │  │  - Phone number detection                         │  │  │
│  │  │  - Email detection                                │  │  │
│  │  │  - Date of birth detection                        │  │  │
│  │  │  - Medical record number detection                │  │  │
│  │  │  - Health insurance number detection              │  │  │
│  │  │  - Address detection                              │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  ComplianceValidator                              │  │  │
│  │  │  - Validates data handling practices              │  │  │
│  │  │  - Ensures encryption requirements                │  │  │
│  │  │  - Checks access controls                         │  │  │
│  │  │  - Audit trail requirements                       │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │  ⚠️ ISSUE: Detection only - no automatic redaction    │  │
│  │  ⚠️ ISSUE: Not consistently enforced                  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 5: Secrets Management (security/secrets_manager)│  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  SecretsManager                                   │  │  │
│  │  │  - Environment variable validation                │  │  │
│  │  │  - API key rotation                               │  │  │
│  │  │  - Secret encryption at rest                      │  │  │
│  │  │  - Access auditing                                │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LAYER 6: Audit System (auditing/audit_system.py)      │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  AuditLogger                                      │  │  │
│  │  │  - Logs all security events                       │  │  │
│  │  │  - Logs authentication attempts                   │  │  │
│  │  │  - Logs data access                               │  │  │
│  │  │  - Logs configuration changes                     │  │  │
│  │  │  - Tamper-proof audit trail                       │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### **HIPAA Compliance Map**

| HIPAA Requirement | Implementation | Status |
|-------------------|----------------|--------|
| **Access Control** | JWT tokens, RBAC | ✅ Implemented |
| **Audit Controls** | AuditSystem, RequestLogging | ✅ Implemented |
| **Integrity** | HMAC signatures | ⚠️ Partial |
| **Person Authentication** | Password + JWT | ✅ Implemented |
| **Transmission Security** | HTTPS/TLS | ✅ Implemented |
| **Encryption at Rest** | NOT implemented | ❌ Critical gap |
| **PHI Detection** | PHIDetector | ⚠️ Detection only, no redaction |
| **Minimum Necessary** | NOT enforced | ❌ Critical gap |
| **Breach Notification** | NOT implemented | ❌ Critical gap |

---

## 11. ENTRY POINTS & WORKFLOWS

### **Application Entry Points**

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRY POINTS                             │
│                                                              │
│  1. API Server (Production)                                 │
│     python api_server.py                                    │
│     ├─ Starts FastAPI server on port 8000                   │
│     ├─ Loads Application from src/main.py                   │
│     ├─ Initializes all middleware                           │
│     ├─ Registers routes                                     │
│     └─ Serves REST API                                      │
│                                                              │
│  2. Main Application (Development)                          │
│     python -m src.main                                      │
│     ├─ Application class initialization                     │
│     ├─ Module manager setup                                 │
│     ├─ Device detection (CPU/GPU)                           │
│     ├─ Performance profiling setup                          │
│     └─ Component initialization                             │
│                                                              │
│  3. CLI Voice Chat                                          │
│     python -m src.cli.voice_chat                            │
│     ├─ Voice input/output                                   │
│     ├─ Whisper ASR                                          │
│     ├─ TTS synthesis                                        │
│     └─ Console interface                                    │
│                                                              │
│  4. Test Runner                                             │
│     pytest                                                  │
│     └─ Runs unit and integration tests                      │
└─────────────────────────────────────────────────────────────┘
```

### **Predefined Workflows**

```
AgentOrchestrator Workflows (12 total)

1. "basic_chat"
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘

2. "empathetic_support"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘

3. "enhanced_empathetic_chat" (DEFAULT)
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │Personality│ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │   Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

4. "therapeutic_session"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │ Therapy  │ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │  Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

5. "comprehensive_diagnosis"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │Diagnosis │ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │ Service  │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

6. "crisis_intervention"
   ┌──────────┐    ┌──────────────────────┐    ┌──────────┐
   │  Safety  │ →  │  Crisis Response      │ →  │ Response │
   │  Agent   │    │  (immediate help)     │    │          │
   └──────────┘    └──────────────────────┘    └──────────┘

7. "personality_assessment"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │Personality│ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘

8. "research_assisted"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │  Search  │ →  │   Chat   │ →  │ Response │
   │  Agent   │    │  Agent   │    │  Agent   │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘

9. "breakthrough_detection"
   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │ Breakthrough │ →  │ Response │
   │  Agent   │    │  Agent   │    │   Detection  │    │          │
   └──────────┘    └──────────┘    └──────────────┘    └──────────┘

10. "readiness_assessment"
   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │  Readiness   │ →  │ Response │
   │  Agent   │    │  Agent   │    │  Assessment  │    │          │
   └──────────┘    └──────────┘    └──────────────┘    └──────────┘

11. "friction_guided"
   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │   Friction   │ →  │ Response │
   │  Agent   │    │  Agent   │    │ Coordinator  │    │          │
   └──────────┘    └──────────┘    └──────────────┘    └──────────┘

12. "full_therapeutic_pipeline"
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Safety  │ →  │ Emotion  │ →  │Personality│ →  │ Therapy  │ →  │Diagnosis │
   │  Agent   │    │  Agent   │    │  Agent   │    │  Agent   │    │ Service  │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                           ↓
   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐
   │Breakthrough│ →│ Friction │ →  │    Chat      │ →  │      Response        │
   │ Detection  │  │Coordinator│   │   Agent      │    │                      │
   └──────────┘    └──────────┘    └──────────────┘    └──────────────────────┘

Each workflow includes:
- Supervisor validation after each agent
- Context updates between agents
- Memory storage after completion
- Performance metrics collection
```

---

## 12. SERVICE LAYER MAP

### **Service Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   DIAGNOSIS SERVICES                         │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Interface Layer (services/diagnosis/interfaces.py)  │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  IDiagnosisService (Abstract)                   │ │  │
│  │  │  - diagnose(request) → DiagnosisResult          │ │  │
│  │  │  - validate_request(request) → bool             │ │  │
│  │  │  - supports_diagnosis_type(type) → bool         │ │  │
│  │  │  - get_service_health() → Dict                  │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  IEnhancedDiagnosisService (extends above)      │ │  │
│  │  │  - get_comprehensive_diagnosis()                │ │  │
│  │  │  - get_temporal_analysis()                      │ │  │
│  │  │  - get_cultural_adaptations()                   │ │  │
│  │  │  - get_personalized_recommendations()           │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  IDiagnosisOrchestrator                         │ │  │
│  │  │  - orchestrate_diagnosis(request)               │ │  │
│  │  │  - register_diagnosis_service(name, service)    │ │  │
│  │  │  - get_available_services()                     │ │  │
│  │  │  - get_orchestrator_health()                    │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  IDiagnosisAgentAdapter                         │ │  │
│  │  │  - adapt_agent_request(input, context)          │ │  │
│  │  │  - adapt_diagnosis_response(result, format)     │ │  │
│  │  │  - get_supported_agents()                       │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ↓                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Implementation Layer                                 │  │
│  │                                                       │  │
│  │  services/diagnosis/unified_service.py (810 lines)   │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  UnifiedDiagnosisService                        │ │  │
│  │  │  - Facade pattern                               │ │  │
│  │  │  - Coordinates multiple diagnosis backends      │ │  │
│  │  │  - Strategy selection logic                     │ │  │
│  │  │  - Memory integration                           │ │  │
│  │  │  - Vector DB integration                        │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  services/diagnosis/orchestrator.py                  │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  DiagnosisOrchestrator                          │ │  │
│  │  │  - Service registry                             │ │  │
│  │  │  - Diagnosis workflow management                │ │  │
│  │  │  - Result aggregation                           │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  services/diagnosis/agent_adapter.py                 │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  DiagnosisAgentAdapter                          │ │  │
│  │  │  - Adapts legacy diagnosis agents               │ │  │
│  │  │  - Format conversion                            │ │  │
│  │  │  - Backward compatibility                       │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  services/diagnosis/memory_integration.py            │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  MemoryIntegrationService                       │ │  │
│  │  │  - Store diagnosis insights                     │ │  │
│  │  │  - Retrieve historical context                  │ │  │
│  │  │  - Session continuity                           │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ↓                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Backend Implementations (⚠️ DUPLICATION)            │  │
│  │                                                       │  │
│  │  diagnosis/comprehensive_diagnosis.py (1,452 lines)  │  │
│  │  - Main comprehensive diagnosis                      │  │
│  │  - Vector DB RAG                                     │  │
│  │  - Voice emotion analysis                            │  │
│  │                                                       │  │
│  │  diagnosis/enhanced_diagnosis.py (1,436 lines)       │  │
│  │  - Extended diagnosis with more conditions           │  │
│  │  - Multimodal analysis                               │  │
│  │                                                       │  │
│  │  diagnosis/differential_diagnosis.py (1,366 lines)   │  │
│  │  - Differential diagnosis support                    │  │
│  │  - Condition differentiation                         │  │
│  │                                                       │  │
│  │  diagnosis/enterprise_multimodal_pipeline.py         │  │
│  │  (1,620 lines)                                       │  │
│  │  - Enterprise version                                │  │
│  │  - Bayesian models                                   │  │
│  │  - Fusion logic                                      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 CONCLUSION

This project map provides a comprehensive view of the Solace-AI mental health chatbot system. Key takeaways:

### **✅ Strengths:**
1. Sophisticated multi-agent architecture
2. Comprehensive security middleware
3. Well-organized module structure
4. Advanced therapeutic capabilities
5. Strong dependency injection pattern

### **⚠️ Critical Issues:**
1. **Diagnosis duplication** - 8 implementations with no clear selection logic
2. **Security is optional** - Base agent can skip security validation
3. **Memory persistence risks** - Unencrypted pickle files
4. **Integration gaps** - Enterprise features not integrated
5. **Testing gaps** - Minimal test coverage

### **📋 Next Steps:**
1. Implement the proposed clean architecture (see improvements.md)
2. Consolidate diagnosis module
3. Make security mandatory
4. Fix memory encryption
5. Add comprehensive tests

**For implementation details, see:**
- [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) - Performance optimizations
- [improvements.md](improvements.md) - Suggested improvements
- [README.md](README.md) - User documentation

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Maintained By**: Development Team
