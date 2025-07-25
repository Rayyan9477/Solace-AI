# Contextual-Chatbot Refactoring Plan

## Current Issues
1. **Excessive sys.path manipulation** - 22 files with path hacks
2. **Duplicate/redundant components** - _new versions, similar functionality
3. **Inconsistent import structures**
4. **Mixed entry points** - Multiple main files with overlapping functionality
5. **Scattered configuration and utilities**

## Proposed Structure

```
contextual_chatbot/
├── main.py                          # Single entry point
├── pyproject.toml                   # Modern Python packaging
├── README.md
├── requirements/
│   ├── base.txt                     # Core dependencies
│   ├── voice.txt                    # Voice processing
│   ├── dev.txt                      # Development tools
│   └── prod.txt                     # Production extras
├── contextual_chatbot/              # Main package (proper Python module)
│   ├── __init__.py
│   ├── core/                        # Core functionality
│   │   ├── __init__.py
│   │   ├── application.py           # Main application class
│   │   ├── module_manager.py        # Module system
│   │   └── config.py                # Configuration management
│   ├── agents/                      # AI agents
│   │   ├── __init__.py
│   │   ├── base.py                  # Base agent class
│   │   ├── chat.py                  # Chat agent
│   │   ├── emotion.py               # Emotion analysis
│   │   ├── safety.py                # Safety monitoring
│   │   ├── therapy.py               # Therapy recommendations
│   │   └── orchestrator.py          # Agent coordination
│   ├── models/                      # LLM and ML models
│   │   ├── __init__.py
│   │   ├── llm.py                   # Language model interface
│   │   ├── embeddings.py            # Embedding models
│   │   └── voice/                   # Voice models
│   │       ├── __init__.py
│   │       ├── stt.py               # Speech-to-text
│   │       └── tts.py               # Text-to-speech
│   ├── memory/                      # Memory systems
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Vector database
│   │   ├── conversation.py          # Conversation memory
│   │   └── semantic.py              # Semantic memory
│   ├── assessment/                  # Psychological assessments
│   │   ├── __init__.py
│   │   ├── personality.py           # Personality tests
│   │   ├── mental_health.py         # Mental health screening
│   │   └── questionnaires/          # Question data
│   ├── therapy/                     # Therapeutic knowledge
│   │   ├── __init__.py
│   │   ├── techniques.py            # CBT, mindfulness, etc.
│   │   ├── resources.py             # Therapy resources
│   │   └── data/                    # Therapeutic data files
│   ├── interfaces/                  # User interfaces
│   │   ├── __init__.py
│   │   ├── cli.py                   # Command line interface
│   │   ├── api.py                   # REST API
│   │   └── voice.py                 # Voice interface
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py               # Logging setup
│   │   ├── device.py                # Device management
│   │   └── helpers.py               # Helper functions
│   └── data/                        # Data storage
│       ├── conversations/
│       ├── profiles/
│       └── vector_store/
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/                         # Utility scripts
│   ├── setup_environment.sh
│   └── migrate_data.py
└── docs/                           # Documentation
    ├── api.md
    ├── architecture.md
    └── deployment.md
```

## Key Improvements

### 1. Proper Python Package Structure
- Remove all `sys.path.append()` calls
- Use proper relative imports
- Single `__init__.py` files for package discovery
- Use `pyproject.toml` for modern packaging

### 2. Consolidate Redundant Components
- Remove duplicate files (_new versions)
- Merge similar functionality
- Single source of truth for each feature

### 3. Clean Entry Points
- Single `main.py` entry point
- CLI/API as interface modules
- Clear separation of concerns

### 4. Modular Architecture
- Clear separation by functionality
- Proper dependency injection
- Interface-based design

### 5. Configuration Management
- Single configuration source
- Environment-specific configs
- Type-safe configuration classes

## Migration Steps

1. Create new package structure
2. Consolidate and refactor core modules
3. Remove duplicate files
4. Update all imports to use proper package structure
5. Create single entry point
6. Update configuration system
7. Add proper testing infrastructure
8. Update documentation

## Benefits

- **Maintainability**: Clear module boundaries
- **Scalability**: Easy to add new features
- **Testability**: Proper module isolation
- **Professional**: Follows Python best practices
- **Deployment**: Easier packaging and distribution