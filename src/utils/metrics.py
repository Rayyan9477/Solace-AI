# Modified track_metric to update appropriate counters instead of acting as a decorator

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
        
        # New counter for assessment completions
        if 'assessment_completed' not in REGISTRY._names_to_collectors:
            self.ASSESSMENT_COMPLETED = Counter(
                'assessment_completed',
                'Number of completed assessments'
            )

        if 'embedding_generation_time' not in REGISTRY._names_to_collectors:
            self.EMBEDDING_TIME = Summary(
                'embedding_generation_time',
                'Time spent generating text embeddings'
            )

metrics = Metrics()

def track_metric(metric_name: str, value: float):
    if metric_name == "embedding":
        metrics.EMBEDDING_TIME.observe(value)
    elif metric_name == "response":
        metrics.RESPONSE_TIME.observe(value)
    elif metric_name == "assessment_completed":
        metrics.ASSESSMENT_COMPLETED.inc(value)
    elif metric_name == "safety_flag_raised":
        metrics.SAFETY_FLAGS.labels(severity_level="raised").inc(value)