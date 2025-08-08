try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY
except ImportError:  # Allow running without prometheus_client installed
    Counter = Gauge = Histogram = Summary = None
    REGISTRY = None
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
# from agno.metrics import MetricsCollector
# from agno.utils import TokenManager
from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

class Metrics:
    
    def __init__(self):
        """Initialize metrics tracking system"""
        self.initialized = False
        self.prometheus_enabled = False
        try:
            from prometheus_client import Counter, Histogram, Gauge, REGISTRY
            self.prometheus_enabled = True
            self.REGISTRY = REGISTRY
            self._initialize_metrics()
            self.initialized = True
        except ImportError:
            # Prometheus is not installed, provide fallback tracking
            self.metrics_store = {
                'interaction_types': {},
                'response_times': [],
                'emotion_intensities': {},
                'safety_flags': 0
            }
            
    def _initialize_metrics(self):
        # Initialize INTERACTION_TYPES Counter
        try:
            from prometheus_client import Counter, Histogram, Gauge
            self.INTERACTION_TYPES = Counter(
                'chat_interaction_types',
                'Count of different interaction types',
                ['type']
            )
        except ValueError:
            self.INTERACTION_TYPES = self.REGISTRY._names_to_collectors['chat_interaction_types']

        # Initialize RESPONSE_TIME Histogram
        try:
            self.RESPONSE_TIME = Histogram(
                'response_time_seconds',
                'Time taken to generate responses',
                buckets=[0.1, 0.5, 1, 2, 5]
            )
        except ValueError:
            self.RESPONSE_TIME = self.REGISTRY._names_to_collectors['response_time_seconds']

        # Initialize EMOTION_GAUGE Gauge
        try:
            self.EMOTION_GAUGE = Gauge(
                'user_emotion_intensity',
                'Intensity of detected emotions',
                ['emotion']
            )
        except ValueError:
            self.EMOTION_GAUGE = self.REGISTRY._names_to_collectors['user_emotion_intensity']

        # Initialize SAFETY_FLAGS Counter
        try:
            self.SAFETY_FLAGS = Counter(
                'safety_flags',
                'Count of safety flags raised during conversations',
                ['severity']
            )
        except ValueError:
            self.SAFETY_FLAGS = self.REGISTRY._names_to_collectors['safety_flags']
    
    def track_interaction(self, interaction_type: str):
        """Track interaction type"""
        if self.prometheus_enabled and self.initialized:
            self.INTERACTION_TYPES.labels(type=interaction_type).inc()
        else:
            # Fallback tracking
            if interaction_type not in self.metrics_store['interaction_types']:
                self.metrics_store['interaction_types'][interaction_type] = 0
            self.metrics_store['interaction_types'][interaction_type] += 1
    
    def track_response_time(self, time_seconds: float):
        """Track response generation time"""
        if self.prometheus_enabled and self.initialized:
            self.RESPONSE_TIME.observe(time_seconds)
        else:
            # Fallback tracking
            self.metrics_store['response_times'].append(time_seconds)
    
    def track_emotion(self, emotion: str, intensity: float):
        """Track detected emotion intensity"""
        if self.prometheus_enabled and self.initialized:
            self.EMOTION_GAUGE.labels(emotion=emotion).set(intensity)
        else:
            # Fallback tracking
            self.metrics_store['emotion_intensities'][emotion] = intensity
    
    def track_safety_flag(self, severity: str = "warning"):
        """Track safety flag occurrence"""
        if self.prometheus_enabled and self.initialized:
            self.SAFETY_FLAGS.labels(severity=severity).inc()
        else:
            # Fallback tracking
            self.metrics_store['safety_flags'] += 1
    
    def get_metrics_summary(self):
        """Get a summary of tracked metrics (for fallback mode)"""
        if not self.prometheus_enabled:
            return self.metrics_store
        # For prometheus mode, this would require more complex collection logic
        return {"status": "prometheus_enabled"}


# Singleton instance for global access
metrics_manager = Metrics()

def track_metric(metric_type: str, value, **kwargs):
    """Convenience function for tracking metrics"""
    if metric_type == "interaction":
        metrics_manager.track_interaction(value)
    elif metric_type == "response_time":
        metrics_manager.track_response_time(value)
    elif metric_type == "emotion":
        emotion = value
        intensity = kwargs.get("intensity", 5.0)
        metrics_manager.track_emotion(emotion, intensity)
    elif metric_type == "safety_flag":
        severity = kwargs.get("severity", "warning")
        metrics_manager.track_safety_flag(severity)
    else:
        # Unknown metric type, log it
        pass

class MetricsManager:
    """Manages metrics collection and analysis"""
    
    def __init__(self):
        self._metrics = metrics_manager
        self._start_time = datetime.now()
        
    def track_interaction(
        self,
        interaction_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track an interaction with metadata"""
        try:
            # Add basic metadata
            meta = {
                'timestamp': datetime.now().isoformat(),
                'interaction_type': interaction_type,
                **(metadata or {})
            }
            
            # Track latency if provided
            if 'latency' in data:
                self._metrics.track_response_time(data['latency'])
            
            # Track interaction type
            self._metrics.track_interaction(interaction_type)
            
            # Track emotions if present
            if 'emotions' in data:
                for emotion, intensity in data['emotions'].items():
                    self._metrics.track_emotion(emotion, intensity)
            
            # Track safety flags
            if 'safety_flags' in data:
                for flag in data['safety_flags']:
                    self._metrics.track_safety_flag(flag)
            
            return {
                'metrics': {
                    'latency': data.get('latency', 0),
                    'interaction_type': interaction_type,
                    'emotions': data.get('emotions', {}),
                    'safety_flags': data.get('safety_flags', [])
                },
                'metadata': meta
            }
            
        except Exception as e:
            logger.error(f"Failed to track interaction: {str(e)}")
            return {}
            
    def get_metrics_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        interaction_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        try:
            start = datetime.fromisoformat(start_time) if start_time else self._start_time
            end = datetime.fromisoformat(end_time) if end_time else datetime.now()
            
            # Get metrics from Prometheus
            summary = {
                'total_interactions': self._metrics.INTERACTION_TYPES._value.sum(),
                'avg_response_time': float(self._metrics.RESPONSE_TIME.describe()['avg']),
                'total_safety_flags': self._metrics.SAFETY_FLAGS._value.sum(),
                'total_assessments': self._metrics.ASSESSMENT_COMPLETED._value,
                'time_range': {
                    'start': start.isoformat(),
                    'end': end.isoformat()
                }
            }
            
            # Add emotion summaries if available
            emotions = {}
            for emotion in self._metrics.EMOTION_GAUGE._metrics:
                emotions[emotion] = float(self._metrics.EMOTION_GAUGE.labels(emotion=emotion)._value)
            if emotions:
                summary['emotions'] = emotions
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {}
            
    def track_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track an error occurrence"""
        try:
            self.collector.add_error(
                error_type=error_type,
                message=error_message,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'context': context or {}
                }
            )
        except Exception as e:
            logger.error(f"Failed to track error: {str(e)}")
            
    def get_error_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of tracked errors"""
        try:
            # Get filtered errors
            errors = self.collector.get_errors(
                start_time=start_time,
                end_time=end_time
            )
            
            # Group by error type
            error_counts = {}
            for error in errors:
                error_type = error['error_type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
            return {
                'total_errors': len(errors),
                'error_types': error_counts,
                'time_range': {
                    'start': start_time or min(e['metadata']['timestamp'] for e in errors) if errors else None,
                    'end': end_time or max(e['metadata']['timestamp'] for e in errors) if errors else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get error summary: {str(e)}")
            return {}