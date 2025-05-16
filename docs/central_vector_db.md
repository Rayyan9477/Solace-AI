# Central Vector Database Integration Guide

## Overview

The Central Vector Database provides unified storage for all application data using vector embeddings for efficient semantic search and retrieval. This allows for more intelligent context-building, data retrieval, and improved user experiences.

## Key Components

1. **Central Vector Database (`CentralVectorDB`)**: 
   - Provides namespace-based storage for different data types
   - Uses Faiss for vector storage and retrieval
   - Manages user profiles, conversations, diagnostics, personality assessments, etc.

2. **Central Vector Database Module (`CentralVectorDBModule`)**:
   - Integrates the vector database into the module system
   - Exposes services for other components to access

3. **Vector Database Integration Utilities**:
   - Makes the database accessible throughout the application
   - Provides helper functions like `get_user_data`, `add_user_data`, etc.

4. **Migration Utilities**:
   - Helps migrate existing data to the central vector database

## Using the Central Vector Database

### In Agent Code

```python
from utils.vector_db_integration import add_user_data, get_user_data, search_relevant_data

# Store data
doc_id = add_user_data("personality", personality_data)

# Retrieve data
user_profile = get_user_data("profile")
latest_diagnosis = get_user_data("diagnosis")

# Search for relevant data
results = search_relevant_data("anxiety symptoms", ["conversation", "diagnosis"], limit=5)
```

### Integration with Agent Orchestrator

The Agent Orchestrator now automatically:
- Enhances context with relevant past conversations
- Adds the latest diagnosis and personality profile to context
- Stores conversations in the central vector database

### Integration with Agents

All agents now implement the `store_to_vector_db` method to store their results in the central database:

- **Personality Agent**: Stores personality assessments
- **Diagnosis Agent**: Stores diagnostic assessments
- **Emotion Agent**: Stores emotion analysis data
- **Chat Agent**: Stores conversation data

## Data Migration

To migrate existing data to the central vector database, run:

```bash
python src/main.py --migrate-data
```

To specify a user ID for migration:

```bash
python src/main.py --migrate-data --user-id=your_user_id
```

## Configuration Settings

The Central Vector Database configuration is defined in `src/config/settings.py`:

```python
VECTOR_DB_CONFIG = {
    "engine": "faiss",
    "dimension": 768,
    "index_type": "L2",
    "metric_type": "cosine",
    "index_path": str(VECTOR_STORE_PATH),
    "collection_name": "mental_health_kb",
    "central_data_enabled": True,
    "retention_days": 180,
    "embedder_model": "all-MiniLM-L6-v2",
    "namespaces": [
        "user_profile",
        "conversation",
        "knowledge",
        "therapy_resource",
        "diagnostic_data",
        "personality_assessment",
        "emotion_record"
    ]
}
```

## Benefits of the Central Vector Database

1. **Unified Data Storage**: All data is stored in one central location with consistent access patterns
2. **Semantic Search**: Find related information across different data types
3. **Enhanced Context**: Automatically build relevant context for agent interactions
4. **Improved User Experience**: More personalized responses based on user history
5. **Better Data Analysis**: Analyze patterns across different data types

## Next Steps for Further Integration

1. Add more specialized search functions for each data type
2. Implement data retention and privacy policies
3. Add data export and backup functionality
4. Create analytics dashboards using the centralized data
5. Implement user settings and preferences storage in the central database
