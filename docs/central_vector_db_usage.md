# Central Vector Database

The Central Vector Database provides a unified storage solution for all data used by the Contextual-Chatbot. It brings together user profiles, conversations, diagnostic assessments, personality assessments, knowledge base items, and therapy resources in a single, consistent interface.

## Benefits

- **Unified Data Storage**: All application data is stored with consistent patterns and interfaces
- **Semantic Search**: Find related information across different data types using vector similarity
- **Improved Context**: Build relevant context for more personalized responses
- **Simplified Integration**: Single integration point for all data operations
- **Enhanced Privacy**: Centralized control of data retention and privacy settings

## Key Components

1. **CentralVectorDB**: Core database class in `src/database/central_vector_db.py`
2. **CentralVectorDBModule**: Module wrapper in `src/components/central_vector_db_module.py`
3. **Vector DB Integration**: Utility functions in `src/utils/vector_db_integration.py`
4. **Migration Utilities**: Tools to migrate existing data in `src/utils/migration_utils.py`

## Getting Started

### Basic Usage

```python
from utils.vector_db_integration import add_user_data, get_user_data, search_relevant_data

# Store data
doc_id = add_user_data("profile", profile_data)

# Retrieve data
user_profile = get_user_data("profile")
latest_diagnosis = get_user_data("diagnosis")

# Search for relevant data
results = search_relevant_data("anxiety symptoms", ["conversation", "diagnosis"], limit=5)
```

### Migrating Existing Data

To migrate existing data to the central vector database:

```bash
python src/main.py --migrate-data
```

To specify a user ID for migration:

```bash
python src/main.py --migrate-data --user-id=your_user_id
```

### Testing the Integration

Run the test script to verify the central vector database functionality:

```bash
python test_vector_db.py
```

## Data Types

The central vector database supports the following data types:

| Type | Description | Examples |
|------|-------------|----------|
| `user_profile` | User information and preferences | Personal details, app settings, interface preferences |
| `conversation` | Chat history and interactions | User messages, assistant responses, emotion data |
| `diagnostic_data` | Mental health assessments | Symptom analyses, condition assessments, severity levels |
| `personality_assessment` | Personality profiles | Big Five traits, MBTI type, personality insights |
| `knowledge` | Knowledge base items | Mental health information, coping strategies, educational content |
| `therapy_resource` | Therapeutic materials | Exercises, scripts, techniques, worksheets |
| `emotion_record` | Emotion tracking data | Mood logs, emotion analysis, intensity tracking |

## Advanced Usage

### Direct Database Access

For advanced operations, you can directly access the database:

```python
from database.central_vector_db import CentralVectorDB

# Initialize the database
db = CentralVectorDB(user_id="your_user_id")

# Perform operations
profile_id = db.add_user_profile(profile_data)
diagnosis = db.get_latest_diagnosis()
search_results = db.search_documents("anxiety management", limit=10)
```

### Custom Document Storage

For data types not covered by the standard categories:

```python
from utils.vector_db_integration import add_user_data

# Store custom data
custom_data = {
    "content": "Custom document content",
    "metadata": {"key": "value"}
}

doc_id = add_user_data("custom_category", custom_data)
```

## Configuration

The database configuration is defined in `src/config/settings.py`:

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

## Further Development

Future enhancements planned for the central vector database:

1. Data retention and privacy controls
2. Export and backup functionality
3. Analytics tools to identify patterns across data types
4. Integration with additional data sources
5. Enhanced security features
