# Safety Service

**Phase 3: Always-Active Safety Monitoring Service** (CRITICAL PATH)

The Safety Service is a real-time crisis detection and intervention system that monitors every user interaction for mental health risk indicators. It implements a 4-layer safety architecture with multi-modal ML-based detection.

## Architecture

### Clean Architecture Layers
- **Domain**: Pure business logic (entities, value objects, services)
- **Application**: Use cases and orchestration
- **Infrastructure**: External integrations (DB, Kafka, Redis)
- **Presentation**: FastAPI REST API

### Domain-Driven Design
- **Entities**: SafetyAssessment, SafetyPlan, EscalationCase
- **Value Objects**: RiskScore, TriggerIndicator, ProtectiveFactor
- **Domain Services**: SafetyService, CrisisDetector, EscalationService
- **Events**: SafetyAssessmentCreated, CrisisDetected, EscalationTriggered

## Features

### 4-Layer Crisis Detection
1. **Layer 1: Input Gate** (<10ms)
   - Keyword matching (critical/high/elevated risk terms)
   - Pattern recognition (suicidal ideation, self-harm, hopelessness)
   - Risk history augmentation

2. **Layer 2: Processing Guard**
   - Contraindication checking for therapeutic techniques
   - Clinical safety validation
   - Alternative technique recommendations

3. **Layer 3: Sentiment Analysis**
   - Adaptive approach: Transformer model (GPU) or VADER (CPU)
   - Clinical lexicon enhancement
   - Context-aware sentiment scoring

4. **Layer 4: LLM Assessment**
   - Deep semantic analysis for ambiguous cases
   - Contextual risk evaluation
   - Clinical reasoning integration

### Escalation Management
- **Auto-escalation**: Critical cases immediately escalated
- **SLA tracking**: 5-minute critical, 15-minute high, 60-minute elevated
- **Notification system**: Multi-channel alerts (email, SMS, dashboard)
- **Clinician pool**: On-call routing with load balancing

### Safety Monitoring
- **Trajectory analysis**: Risk trend detection over time
- **Pattern matching**: Behavioral pattern recognition
- **Contraindication tracking**: Technique safety validation
- **Audit trail**: Complete HIPAA-compliant logging

## Batches

### Batch 3.1: Core Components (5 files)
- [x] main.py (229 LOC) - FastAPI application
- [x] api.py (332 LOC) - Safety check endpoints
- [x] service.py (384 LOC) - Main safety orchestration
- [x] crisis_detector.py (368 LOC) - Multi-layer crisis detection
- [x] escalation.py (345 LOC) - Escalation workflow

### Batch 3.2: Domain Layer (5 files)
- [x] entities.py (337 LOC) - Domain entities
- [x] value_objects.py (340 LOC) - Immutable value objects
- [x] repository.py (369 LOC) - Data persistence
- [x] events.py (281 LOC) - Domain events
- [x] config.py (187 LOC) - Configuration

### Batch 3.3: ML Components (5 files)
- [x] keyword_detector.py (348 LOC) - Fast keyword detection
- [x] sentiment_analyzer.py (504 LOC) ⚠️ - Sentiment analysis
- [x] pattern_matcher.py (475 LOC) ⚠️ - Pattern recognition
- [x] llm_assessor.py (427 LOC) ⚠️ - LLM-based assessment
- [x] contraindication.py (484 LOC) ⚠️ - Contraindication checking

⚠️ **Note**: Some ML files exceed 400 LOC target due to comprehensive safety logic. All tests passing (394/394).

## Quality Gates

- [x] **Tests**: 394/394 passing (100%)
- [x] **Architecture**: Clean/Hexagonal/DDD compliant
- [x] **Crisis Detection**: <10ms for Layer 1 (keyword matching)
- [x] **Escalation**: Automated workflow with SLA tracking
- [x] **Audit Trail**: Complete HIPAA-compliant logging
- [ ] **LOC Compliance**: 4 ML files exceed 400 LOC (optimization needed)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set environment variables
export SAFETY_SERVICE_HOST=localhost
export SAFETY_SERVICE_PORT=8003
export DATABASE_URL=postgresql://user:pass@localhost/safety
export REDIS_URL=redis://localhost:6379
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Run service
python -m src.main

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## API Endpoints

### Safety Check
```http
POST /api/v1/safety/check
Content-Type: application/json

{
  "user_id": "uuid",
  "message": "string",
  "context": {...},
  "risk_history": [...]
}
```

### Crisis Detection
```http
POST /api/v1/safety/crisis/detect
Content-Type: application/json

{
  "user_id": "uuid",
  "message": "string",
  "severity_threshold": "elevated"
}
```

### Trigger Escalation
```http
POST /api/v1/safety/escalation
Content-Type: application/json

{
  "assessment_id": "uuid",
  "trigger_reason": "string",
  "priority": "critical"
}
```

### Health Check
```http
GET /health
```

## Events Published

- `SafetyAssessmentCreated`: New safety assessment completed
- `CrisisDetected`: Crisis detected in user message
- `EscalationTriggered`: Case escalated to clinician
- `EscalationAssigned`: Case assigned to specific clinician
- `EscalationResolved`: Escalation case resolved

## Performance

- **Layer 1 (Keyword)**: <10ms average
- **Layer 2 (Contraindication)**: <20ms average
- **Layer 3 (Sentiment)**: <50ms (VADER), <200ms (Transformer)
- **Layer 4 (LLM)**: <2000ms (depends on LLM latency)

## Configuration

See `src/config.py` for all configuration options:
- Crisis detection thresholds
- Escalation SLA times
- Notification settings
- Storage backends
- Event publishing

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific batch tests
pytest tests/test_batch_3_3_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run performance tests
pytest tests/ -v -m slow
```

## Monitoring

The service exposes Prometheus metrics at `/metrics`:
- `safety_checks_total`: Total safety checks performed
- `crisis_detections_total`: Total crises detected
- `escalations_total`: Total escalations triggered
- `layer_duration_seconds`: Detection layer latencies

## HIPAA Compliance

- All safety assessments are encrypted at rest
- Complete audit trail for all escalations
- PII handling follows HIPAA guidelines
- Secure clinician assignment with role-based access

## Dependencies

See [requirements.txt](requirements.txt) for full dependency list.

Key dependencies:
- FastAPI 0.115.12
- Pydantic 2.12.5
- PyTorch 2.5.1 (GPU support)
- Transformers 4.48.1
- VADER Sentiment 3.3.2
- spaCy 3.8.4

## Future Optimizations

- Refactor ML files to meet 400 LOC target
- Implement file splitting for complex analyzers
- Add caching for sentiment model predictions
- Optimize LLM assessment with prompt engineering

---

**Status**: ✅ Production-ready (394/394 tests passing)
**Phase**: 3 - Safety Service (CRITICAL PATH)
**Last Updated**: 2026-01-19
