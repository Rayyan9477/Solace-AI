from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY
import time
from typing import Dict, Any

class Metrics:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Metrics, cls).__new__(cls)
            cls._instance._initialize_metrics()
        return cls._instance

    def _initialize_metrics(self):
        # Check if metrics are already registered
        if 'chat_interaction_types' not in REGISTRY._names_to_collectors:
            self.INTERACTION_TYPES = Counter(
                'chat_interaction_types',
                'Count of different interaction types',
                ['type']
            )

        if 'response_time_seconds' not in REGISTRY._names_to_collectors:
            self.RESPONSE_TIME = Histogram(
                'response_time_seconds',
                'Time taken to generate responses',
                buckets=[0.1, 0.5, 1, 2, 5]
            )

        if 'user_emotion_intensity' not in REGISTRY._names_to_collectors:
            self.EMOTION_GAUGE = Gauge(
                'user_emotion_intensity',
                'Intensity of detected emotions',
                ['emotion']
            )

        if 'safety_flag_triggers' not in REGISTRY._names_to_collectors:
            self.SAFETY_FLAGS = Counter(
                'safety_flag_triggers',
                'Count of triggered safety flags',
                ['severity_level']
            )

        if 'embedding_generation_time' not in REGISTRY._names_to_collectors:
            self.EMBEDDING_TIME = Summary(
                'embedding_generation_time',
                'Time spent generating text embeddings'
            )

metrics = Metrics()

def track_metric(metric_name: str, labels: Dict[str, Any] = None):
    """Generic metric tracking decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            
            # Track execution time
            duration = time.time() - start_time
            if metric_name == 'embedding':
                metrics.EMBEDDING_TIME.observe(duration)
            elif metric_name == 'response':
                metrics.RESPONSE_TIME.observe(duration)
            
            return result
        return wrapper
    return decorator

def log_emotion(emotion_data: Dict):
    """Log emotion metrics"""
    metrics.EMOTION_GAUGE.labels(
        emotion=emotion_data.get('primary_emotion', 'unknown')
    ).set(emotion_data.get('intensity', 0))

def log_safety_event(assessment: Dict):
    """Log safety events"""
    if not assessment.get('safe', True):
        metrics.SAFETY_FLAGS.labels(
            severity_level=str(assessment.get('severity', 0))
        ).inc()

def track_interaction(interaction_type: str):
    """Track interaction types"""
    metrics.INTERACTION_TYPES.labels(type=interaction_type).inc()