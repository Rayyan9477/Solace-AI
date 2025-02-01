# metrics.py
from prometheus_client import Counter, Gauge, Histogram
import logging
from typing import Any, Dict
import time

# Initialize metrics
INTERACTION_COUNTER = Counter(
    'chat_interactions_total',
    'Total number of chat interactions'
)
RESPONSE_TIME_HISTOGRAM = Histogram(
    'response_time_seconds',
    'Time taken to generate responses'
)
SEVERITY_GAUGE = Gauge(
    'user_severity_level',
    'Current severity level of user mental state'
)
EMOTION_INTENSITY_GAUGE = Gauge(
    'emotion_intensity',
    'Intensity of user emotions'
)

def track_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """
    Track a metric using Prometheus
    """
    try:
        if metric_name == "chat_interaction":
            INTERACTION_COUNTER.inc()
        elif metric_name == "response_time":
            RESPONSE_TIME_HISTOGRAM.observe(value)
        elif metric_name == "severity_level":
            SEVERITY_GAUGE.set(value)
        elif metric_name == "emotion_intensity":
            EMOTION_INTENSITY_GAUGE.set(value)
    except Exception as e:
        logging.error(f"Error tracking metric {metric_name}: {str(e)}")