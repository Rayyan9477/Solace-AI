from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY
import time
from typing import Dict, Any

class Metrics:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Metrics, cls).__new__(cls)
        # (Re)initialize metrics to include new counters if added later
        cls._instance._initialize_metrics()
        return cls._instance

    def _initialize_metrics(self):
        # Initialize INTERACTION_TYPES Counter
        try:
            self.INTERACTION_TYPES = Counter(
                'chat_interaction_types',
                'Count of different interaction types',
                ['type']
            )
        except ValueError:
            self.INTERACTION_TYPES = REGISTRY._names_to_collectors['chat_interaction_types']

        # Initialize RESPONSE_TIME Histogram
        try:
            self.RESPONSE_TIME = Histogram(
                'response_time_seconds',
                'Time taken to generate responses',
                buckets=[0.1, 0.5, 1, 2, 5]
            )
        except ValueError:
            self.RESPONSE_TIME = REGISTRY._names_to_collectors['response_time_seconds']

        # Initialize EMOTION_GAUGE Gauge
        try:
            self.EMOTION_GAUGE = Gauge(
                'user_emotion_intensity',
                'Intensity of detected emotions',
                ['emotion']
            )
        except ValueError:
            self.EMOTION_GAUGE = REGISTRY._names_to_collectors['user_emotion_intensity']

        # Initialize SAFETY_FLAGS Counter
        try:
            self.SAFETY_FLAGS = Counter(
                'safety_flag_triggers',
                'Count of triggered safety flags',
                ['severity_level']
            )
        except ValueError:
            self.SAFETY_FLAGS = REGISTRY._names_to_collectors['safety_flag_triggers']

        # Initialize ASSESSMENT_COMPLETED Counter
        try:
            self.ASSESSMENT_COMPLETED = Counter(
                'assessment_completed',
                'Number of completed assessments'
            )
        except ValueError:
            self.ASSESSMENT_COMPLETED = REGISTRY._names_to_collectors['assessment_completed']

        # Initialize EMBEDDING_TIME Summary
        try:
            self.EMBEDDING_TIME = Summary(
                'embedding_generation_time',
                'Time spent generating text embeddings'
            )
        except ValueError:
            self.EMBEDDING_TIME = REGISTRY._names_to_collectors['embedding_generation_time']

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