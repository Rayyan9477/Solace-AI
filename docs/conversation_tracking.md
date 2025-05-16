# Conversation Tracking with Faiss Vector Database

This document describes the conversation tracking functionality that has been implemented in the Contextual-Chatbot system using Faiss vector storage.

## Overview

The conversation tracking system stores all interactions between users and the chatbot, including:
- User messages
- Assistant responses
- Timestamps
- Emotion data
- Additional metadata

This data is stored using Faiss, an efficient similarity search library, allowing for semantic search and retrieval of past conversations.

## Key Components

### 1. ConversationTracker Class

Located in `src/database/conversation_tracker.py`, this class provides:

- Storage of conversations with timestamps and emotion data
- Semantic search of past conversations
- Filtering conversations by date, emotion, and other metadata
- Analysis of emotion patterns over time
- Export functionality for external analysis

### 2. Agent Orchestrator Integration

The `AgentOrchestrator` class has been extended to automatically track conversations in the Faiss database. This ensures that all interactions are recorded for future reference and analysis.

### 3. Analysis Tools

Two utility scripts have been provided for analyzing conversation data:

- `conversation_analysis.py`: Command-line tool for accessing, analyzing and exporting conversation history
- `emotion_analysis.py`: Visualization tool for analyzing emotion patterns over time

## Usage

### Tracking Conversations

Conversations are automatically tracked when processed through the `AgentOrchestrator`. No additional code is required to enable tracking.

```python
# The orchestrator will automatically track this conversation
result = await orchestrator.process_message(message)
```

### Accessing Conversation History

Use the `ConversationTracker` class directly to access conversation history:

```python
from src.database.conversation_tracker import ConversationTracker

# Initialize the tracker
tracker = ConversationTracker(user_id="user123")

# Get recent conversations
recent = tracker.get_recent_conversations(limit=5)

# Search for relevant conversations
results = tracker.search_conversations(
    query="feeling anxious", 
    limit=3
)

# Get conversations by emotion
anxious_convs = tracker.get_conversations_by_emotion(
    emotion="anxious", 
    limit=5
)
```

### Command-line Analysis

Use the included utility scripts for command-line analysis:

```bash
# Show recent conversations
python conversation_analysis.py --user user123 --action recent

# View emotion distribution
python conversation_analysis.py --user user123 --action emotion

# Search conversations
python conversation_analysis.py --user user123 --action search --query "feeling anxious"

# Export conversations to JSON
python conversation_analysis.py --user user123 --action export --output conversations.json

# View statistics
python conversation_analysis.py --user user123 --action stats
```

### Emotion Analysis Visualization

Visualize emotion patterns over time:

```bash
# Create emotion analysis visualization
python emotion_analysis.py --user user123 --days 30 --output emotion_analysis.png
```

## Data Storage

Conversations are stored in the following locations:

- Vector embeddings: In-memory Faiss index with serialization to disk
- Metadata: JSON file in `src/data/conversations/<user_id>/metadata.json`
- Exports: JSON files in `src/data/conversations/<user_id>/`

## Dependencies

The conversation tracking system depends on:

- `faiss-cpu`: Efficient similarity search
- `sentence-transformers`: Text embedding generation
- `numpy`: Numerical operations
- `matplotlib` and `pandas`: For emotion analysis visualization

These dependencies are already included in the project's requirements.txt file.

## Future Enhancements

Potential future enhancements to the conversation tracking system:

1. **Persistent Faiss Index**: Implement proper persistence of the Faiss index to support larger conversation histories
2. **Multi-user Scaling**: Optimize for multiple concurrent users
3. **Topic Clustering**: Add automatic topic clustering of conversations
4. **Anomaly Detection**: Identify unusual patterns in emotional responses
5. **Integration with External Analytics**: Export data to external analytics platforms
