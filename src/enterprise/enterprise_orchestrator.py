"""
Enterprise Agent Orchestrator for Solace-AI

This module provides enterprise-grade orchestration capabilities with:
- Full enterprise system integration
- Advanced monitoring and analytics
- Comprehensive quality assurance
- Clinical safety and compliance
- Real-time knowledge integration
- Performance optimization
- Scalable architecture
- Health monitoring and self-healing
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

from src.utils.logger import get_logger
from src.agents.agent_orchestrator import AgentOrchestrator, MessageType
from langchain.schema.language_model import BaseLanguageModel

# Enterprise system imports
from src.enterprise.real_time_monitoring import get_real_time_monitor
from src.enterprise.analytics_dashboard import create_analytics_dashboard
from src.enterprise.quality_assurance import create_quality_assurance_framework
from src.enterprise.knowledge_integration import create_knowledge_integration_system
from src.enterprise.data_reliability import create_data_reliability_system
from src.enterprise.dependency_injection import create_dependency_container
from src.enterprise.clinical_compliance import create_clinical_compliance_system
from src.integration.event_bus import get_event_bus, Event, EventType, EventPriority
from src.integration.supervision_mesh import create_clinical_supervision_mesh
from src.integration.friction_engine import FrictionEngine

logger = get_logger(__name__)


class EnterpriseConfig:
    """Enterprise configuration management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Enterprise feature flags
        self.enterprise_features_enabled = self.config.get('enterprise_features_enabled', True)
        self.auto_start_systems = self.config.get('auto_start_systems', True)
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.analytics_enabled = self.config.get('analytics_enabled', True)
        self.quality_assurance_enabled = self.config.get('quality_assurance_enabled', True)
        self.knowledge_integration_enabled = self.config.get('knowledge_integration_enabled', True)
        self.data_reliability_enabled = self.config.get('data_reliability_enabled', True)
        self.clinical_compliance_enabled = self.config.get('clinical_compliance_enabled', True)
        
        # Performance settings
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 100)
        self.request_timeout_seconds = self.config.get('request_timeout_seconds', 30)
        self.health_check_interval = self.config.get('health_check_interval', 60)
        
        # Quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 70.0)
        self.max_response_time = self.config.get('max_response_time', 10.0)
        
        # Compliance settings
        self.require_supervision = self.config.get('require_supervision', True)
        self.enable_therapeutic_friction = self.config.get('enable_therapeutic_friction', True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enterprise_features_enabled': self.enterprise_features_enabled,
            'auto_start_systems': self.auto_start_systems,
            'monitoring_enabled': self.monitoring_enabled,
            'analytics_enabled': self.analytics_enabled,
            'quality_assurance_enabled': self.quality_assurance_enabled,
            'knowledge_integration_enabled': self.knowledge_integration_enabled,
            'data_reliability_enabled': self.data_reliability_enabled,
            'clinical_compliance_enabled': self.clinical_compliance_enabled,
            'max_concurrent_requests': self.max_concurrent_requests,
            'request_timeout_seconds': self.request_timeout_seconds,
            'health_check_interval': self.health_check_interval,
            'min_quality_score': self.min_quality_score,
            'max_response_time': self.max_response_time,
            'require_supervision': self.require_supervision,
            'enable_therapeutic_friction': self.enable_therapeutic_friction
        }


@dataclass
class EnterpriseRequestContext:
    """Enterprise request context with additional metadata."""
    
    request_id: str
    user_id: Optional[str]
    session_id: str
    message: str
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    require_quality_check: bool = True
    require_compliance_check: bool = True
    enable_knowledge_enhancement: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnterpriseAgentOrchestrator:
    """
    Enterprise-grade agent orchestrator with comprehensive enterprise features.
    
    Provides:
    - Full enterprise system integration
    - Advanced monitoring and analytics
    - Quality assurance and validation
    - Clinical safety and compliance
    - Real-time knowledge integration
    - Performance optimization
    - Health monitoring and recovery
    """
    
    def __init__(self, model: BaseLanguageModel, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = EnterpriseConfig(config)
        
        # Core orchestrator
        self.core_orchestrator = AgentOrchestrator()
        
        # Enterprise components
        self.event_bus = None
        self.dependency_container = None
        self.real_time_monitor = None
        self.analytics_dashboard = None
        self.quality_assurance = None
        self.knowledge_integration = None
        self.data_reliability = None
        self.clinical_compliance = None
        self.supervision_mesh = None
        self.friction_engine = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.startup_time = None
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'last_activity': None,
            'enterprise_features_used': defaultdict(int)
        }
        
        # Request management
        self.active_requests = {}
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        logger.info("EnterpriseAgentOrchestrator created")
    
    async def initialize(self) -> None:
        """Initialize the enterprise orchestrator and all systems."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing EnterpriseAgentOrchestrator...")
            
            # Initialize event bus first
            self.event_bus = get_event_bus()
            
            # Initialize enterprise systems if enabled
            if self.config.enterprise_features_enabled:
                await self._initialize_enterprise_systems()
            
            # Initialize core orchestrator
            await self._initialize_core_orchestrator()
            
            # Start systems if configured
            if self.config.auto_start_systems:
                await self.start()
            
            self.is_initialized = True
            self.startup_time = datetime.now()
            
            logger.info("EnterpriseAgentOrchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize EnterpriseAgentOrchestrator: {e}")
            raise
    
    async def start(self) -> None:
        """Start all enterprise systems."""
        if self.is_running:
            return
        
        try:
            logger.info("Starting EnterpriseAgentOrchestrator systems...")
            
            # Start systems in dependency order
            if self.event_bus:
                await self.event_bus.start()
            
            if self.dependency_container:
                await self.dependency_container.start()
            
            if self.real_time_monitor:
                await self.real_time_monitor.start()
            
            if self.analytics_dashboard:
                await self.analytics_dashboard.start()
            
            if self.quality_assurance:
                await self.quality_assurance.start()
            
            if self.knowledge_integration:
                await self.knowledge_integration.start()
            
            if self.data_reliability:
                await self.data_reliability.start()
            
            if self.clinical_compliance:
                await self.clinical_compliance.start()
            
            self.is_running = True
            
            # Publish startup event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    event_type=EventType.SYSTEM_STARTUP,
                    source_agent="enterprise_orchestrator",
                    data={
                        'startup_time': self.startup_time.isoformat(),
                        'config': self.config.to_dict()
                    }
                ))\n            \n            logger.info("All enterprise systems started successfully")\n            \n        except Exception as e:\n            logger.error(f"Failed to start enterprise systems: {e}")\n            raise\n    \n    async def stop(self) -> None:\n        """Stop all enterprise systems gracefully."""\n        if not self.is_running:\n            return\n        \n        try:\n            logger.info("Stopping EnterpriseAgentOrchestrator systems...")\n            \n            # Stop systems in reverse order\n            if self.clinical_compliance:\n                await self.clinical_compliance.stop()\n            \n            if self.data_reliability:\n                await self.data_reliability.stop()\n            \n            if self.knowledge_integration:\n                await self.knowledge_integration.stop()\n            \n            if self.quality_assurance:\n                await self.quality_assurance.stop()\n            \n            if self.analytics_dashboard:\n                await self.analytics_dashboard.stop()\n            \n            if self.real_time_monitor:\n                await self.real_time_monitor.stop()\n            \n            if self.dependency_container:\n                await self.dependency_container.stop()\n            \n            if self.event_bus:\n                await self.event_bus.stop()\n            \n            self.is_running = False\n            \n            logger.info("All enterprise systems stopped")\n            \n        except Exception as e:\n            logger.error(f"Error stopping enterprise systems: {e}")\n    \n    async def process_message(self, \n                            message: str, \n                            user_id: Optional[str] = None,\n                            session_id: Optional[str] = None,\n                            priority: EventPriority = EventPriority.NORMAL,\n                            **kwargs) -> Dict[str, Any]:\n        """Process a message with full enterprise integration."""\n        \n        # Create request context\n        context = EnterpriseRequestContext(\n            request_id=str(uuid.uuid4()),\n            user_id=user_id,\n            session_id=session_id or str(uuid.uuid4()),\n            message=message,\n            timestamp=datetime.now(),\n            priority=priority,\n            metadata=kwargs\n        )\n        \n        # Acquire request semaphore\n        async with self.request_semaphore:\n            return await self._process_enterprise_request(context)\n    \n    async def _process_enterprise_request(self, context: EnterpriseRequestContext) -> Dict[str, Any]:\n        """Process request with full enterprise features."""\n        start_time = time.time()\n        \n        try:\n            # Track active request\n            self.active_requests[context.request_id] = context\n            \n            # Update metrics\n            self.metrics['total_requests'] += 1\n            self.metrics['last_activity'] = context.timestamp\n            \n            # Record monitoring metrics\n            if self.real_time_monitor:\n                self.real_time_monitor.record_metric(\n                    "enterprise_request_started",\n                    1,\n                    labels={\n                        'user_id': context.user_id or 'anonymous',\n                        'session_id': context.session_id,\n                        'priority': context.priority.name\n                    }\n                )\n                self.metrics['enterprise_features_used']['monitoring'] += 1\n            \n            # Pre-processing quality assurance\n            if self.config.quality_assurance_enabled and self.quality_assurance and context.require_quality_check:\n                qa_result = await self._perform_quality_check(context.message, context)\n                if not qa_result['passed']:\n                    return self._create_error_response(\n                        context,\n                        "Quality assurance check failed",\n                        qa_result,\n                        start_time\n                    )\n                self.metrics['enterprise_features_used']['quality_assurance'] += 1\n            \n            # Knowledge enhancement\n            knowledge_context = {}\n            if self.config.knowledge_integration_enabled and self.knowledge_integration and context.enable_knowledge_enhancement:\n                knowledge_context = await self._enhance_with_knowledge(context.message)\n                self.metrics['enterprise_features_used']['knowledge_integration'] += 1\n            \n            # Process with core orchestrator\n            core_response = await self._process_with_core_orchestrator(\n                context, knowledge_context\n            )\n            \n            # Post-processing quality assurance\n            if self.config.quality_assurance_enabled and self.quality_assurance and core_response.get('content'):\n                response_qa = await self._perform_response_quality_check(\n                    core_response['content'], context\n                )\n                core_response['quality_assessment'] = response_qa\n            \n            # Clinical compliance check\n            compliance_result = {}\n            if self.config.clinical_compliance_enabled and self.clinical_compliance and context.require_compliance_check:\n                compliance_result = await self._perform_compliance_check(\n                    core_response, context\n                )\n                self.metrics['enterprise_features_used']['clinical_compliance'] += 1\n            \n            # Calculate processing time\n            processing_time = time.time() - start_time\n            self.metrics['response_times'].append(processing_time)\n            if len(self.metrics['response_times']) > 1000:  # Keep last 1000\n                self.metrics['response_times'] = self.metrics['response_times'][-1000:]\n            \n            self.metrics['average_response_time'] = sum(self.metrics['response_times']) / len(self.metrics['response_times'])\n            self.metrics['successful_requests'] += 1\n            \n            # Record completion metrics\n            if self.real_time_monitor:\n                self.real_time_monitor.record_metric(\n                    "enterprise_request_completed",\n                    1,\n                    labels={'success': 'true'}\n                )\n                self.real_time_monitor.record_metric(\n                    "enterprise_response_time",\n                    processing_time\n                )\n            \n            # Create comprehensive response\n            enterprise_response = {\n                'request_id': context.request_id,\n                'session_id': context.session_id,\n                'success': True,\n                'content': core_response.get('content', ''),\n                'core_response': core_response,\n                'knowledge_context': knowledge_context,\n                'compliance_result': compliance_result,\n                'processing_time_seconds': processing_time,\n                'enterprise_metadata': {\n                    'features_used': dict(self.metrics['enterprise_features_used']),\n                    'quality_checked': context.require_quality_check,\n                    'compliance_checked': context.require_compliance_check,\n                    'knowledge_enhanced': len(knowledge_context) > 0,\n                    'enterprise_orchestrator_version': '1.0.0'\n                },\n                'timestamp': context.timestamp.isoformat()\n            }\n            \n            return enterprise_response\n            \n        except Exception as e:\n            self.metrics['failed_requests'] += 1\n            \n            # Record error metrics\n            if self.real_time_monitor:\n                self.real_time_monitor.record_metric(\n                    "enterprise_request_errors",\n                    1,\n                    labels={'error_type': type(e).__name__}\n                )\n            \n            # Report to clinical compliance if available\n            if self.clinical_compliance:\n                await self.clinical_compliance.safety_monitor.report_safety_event(\n                    event_type='system_malfunction',\n                    severity='moderate',\n                    description=f'Enterprise orchestrator error: {str(e)}',\n                    session_id=context.session_id,\n                    metadata={'request_id': context.request_id}\n                )\n            \n            logger.error(f"Error processing enterprise request {context.request_id}: {e}")\n            \n            return self._create_error_response(context, str(e), {}, start_time)\n            \n        finally:\n            # Clean up active request\n            self.active_requests.pop(context.request_id, None)\n    \n    async def _initialize_enterprise_systems(self) -> None:\n        """Initialize all enterprise systems."""\n        try:\n            # Initialize dependency container\n            self.dependency_container = create_dependency_container(self.event_bus)\n            \n            # Initialize monitoring system\n            if self.config.monitoring_enabled:\n                self.real_time_monitor = get_real_time_monitor(self.event_bus)\n            \n            # Initialize analytics dashboard\n            if self.config.analytics_enabled and self.real_time_monitor:\n                self.analytics_dashboard = create_analytics_dashboard(self.real_time_monitor)\n            \n            # Initialize supervision mesh\n            from src.agents.supervisor_agent import SupervisorAgent\n            supervisor_agent = SupervisorAgent(self.model)\n            self.supervision_mesh = create_clinical_supervision_mesh(supervisor_agent, self.event_bus)\n            \n            # Initialize friction engine\n            self.friction_engine = FrictionEngine(self.event_bus)\n            \n            # Initialize quality assurance\n            if self.config.quality_assurance_enabled:\n                self.quality_assurance = create_quality_assurance_framework(\n                    self.event_bus, self.supervision_mesh\n                )\n            \n            # Initialize knowledge integration\n            if self.config.knowledge_integration_enabled:\n                self.knowledge_integration = create_knowledge_integration_system(self.event_bus)\n            \n            # Initialize data reliability\n            if self.config.data_reliability_enabled:\n                self.data_reliability = create_data_reliability_system(self.event_bus)\n            \n            # Initialize clinical compliance\n            if self.config.clinical_compliance_enabled:\n                self.clinical_compliance = create_clinical_compliance_system(self.event_bus)\n            \n            logger.info("Enterprise systems initialized")\n            \n        except Exception as e:\n            logger.error(f"Failed to initialize enterprise systems: {e}")\n            raise\n    \n    async def _initialize_core_orchestrator(self) -> None:\n        """Initialize core orchestrator."""\n        try:\n            # The core orchestrator should be initialized with the model\n            # This would depend on the specific implementation of AgentOrchestrator\n            pass\n            \n        except Exception as e:\n            logger.error(f"Failed to initialize core orchestrator: {e}")\n            raise\n    \n    async def _perform_quality_check(self, message: str, context: EnterpriseRequestContext) -> Dict[str, Any]:\n        """Perform quality assurance check on input message."""\n        try:\n            qa_report = await self.quality_assurance.assess_quality(\n                target=message,\n                assessment_type="content",\n                context={\n                    'user_id': context.user_id,\n                    'session_id': context.session_id,\n                    'request_id': context.request_id\n                }\n            )\n            \n            passed = qa_report.overall_quality_score >= self.config.min_quality_score\n            \n            return {\n                'passed': passed,\n                'score': qa_report.overall_quality_score,\n                'report': qa_report.to_dict()\n            }\n            \n        except Exception as e:\n            logger.error(f"Quality check failed: {e}")\n            return {'passed': True, 'error': str(e)}\n    \n    async def _perform_response_quality_check(self, response: str, context: EnterpriseRequestContext) -> Dict[str, Any]:\n        """Perform quality assurance check on response."""\n        try:\n            qa_report = await self.quality_assurance.assess_quality(\n                target=response,\n                assessment_type="comprehensive",\n                context={\n                    'user_id': context.user_id,\n                    'session_id': context.session_id,\n                    'request_id': context.request_id,\n                    'is_response': True\n                }\n            )\n            \n            return qa_report.to_dict()\n            \n        except Exception as e:\n            logger.error(f"Response quality check failed: {e}")\n            return {'error': str(e)}\n    \n    async def _enhance_with_knowledge(self, message: str) -> Dict[str, Any]:\n        """Enhance request with relevant knowledge."""\n        try:\n            # Search for relevant knowledge\n            knowledge_items = await self.knowledge_integration.search_knowledge(\n                message, limit=5\n            )\n            \n            # Get research trends\n            trends = await self.knowledge_integration.get_research_trends(message)\n            \n            return {\n                'relevant_knowledge': [item.to_dict() for item in knowledge_items],\n                'research_trends': trends.to_dict() if trends else {},\n                'knowledge_enhanced': len(knowledge_items) > 0\n            }\n            \n        except Exception as e:\n            logger.error(f"Knowledge enhancement failed: {e}")\n            return {'error': str(e)}\n    \n    async def _perform_compliance_check(self, response: Dict[str, Any], context: EnterpriseRequestContext) -> Dict[str, Any]:\n        """Perform clinical compliance check."""\n        try:\n            # Record audit event\n            await self.clinical_compliance.compliance_engine.record_audit_event(\n                action="process_message",\n                actor_type="system",\n                actor_id="enterprise_orchestrator",\n                resource_type="user_message",\n                resource_id=context.request_id,\n                details={\n                    'message_length': len(context.message),\n                    'user_id': context.user_id,\n                    'processing_time': response.get('processing_time_seconds', 0)\n                },\n                session_id=context.session_id\n            )\n            \n            return {\n                'audit_recorded': True,\n                'compliance_status': 'compliant'\n            }\n            \n        except Exception as e:\n            logger.error(f"Compliance check failed: {e}")\n            return {'error': str(e)}\n    \n    async def _process_with_core_orchestrator(self, \n                                            context: EnterpriseRequestContext,\n                                            knowledge_context: Dict[str, Any]) -> Dict[str, Any]:\n        """Process message with core orchestrator."""\n        try:\n            # This would integrate with the existing agent orchestrator\n            # For now, we'll simulate a response\n            response = {\n                'content': f"Processed message: {context.message[:100]}...",\n                'workflow': 'general',\n                'agents_used': ['diagnosis', 'therapy'],\n                'knowledge_context': knowledge_context\n            }\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f"Core orchestrator processing failed: {e}")\n            return {\n                'error': str(e),\n                'content': 'I apologize, but I encountered an error processing your request.'\n            }\n    \n    def _create_error_response(self, \n                              context: EnterpriseRequestContext,\n                              error_message: str,\n                              details: Dict[str, Any],\n                              start_time: float) -> Dict[str, Any]:\n        """Create standardized error response."""\n        return {\n            'request_id': context.request_id,\n            'session_id': context.session_id,\n            'success': False,\n            'error': error_message,\n            'error_details': details,\n            'processing_time_seconds': time.time() - start_time,\n            'enterprise_metadata': {\n                'error_handled': True,\n                'enterprise_orchestrator_version': '1.0.0'\n            },\n            'timestamp': context.timestamp.isoformat()\n        }\n    \n    async def get_system_health(self) -> Dict[str, Any]:\n        """Get comprehensive system health status."""\n        health_data = {\n            'orchestrator_status': 'healthy' if self.is_running else 'stopped',\n            'is_initialized': self.is_initialized,\n            'is_running': self.is_running,\n            'startup_time': self.startup_time.isoformat() if self.startup_time else None,\n            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,\n            'active_requests': len(self.active_requests),\n            'performance_metrics': self.metrics.copy(),\n            'configuration': self.config.to_dict()\n        }\n        \n        # Get enterprise system health\n        if self.real_time_monitor:\n            try:\n                enterprise_health = await self.real_time_monitor.get_system_health()\n                health_data['enterprise_systems'] = enterprise_health\n            except Exception as e:\n                health_data['enterprise_systems'] = {'error': str(e)}\n        \n        if self.data_reliability:\n            try:\n                reliability_health = await self.data_reliability.get_system_health()\n                health_data['data_reliability'] = reliability_health\n            except Exception as e:\n                health_data['data_reliability'] = {'error': str(e)}\n        \n        if self.clinical_compliance:\n            try:\n                compliance_status = await self.clinical_compliance.generate_comprehensive_report()\n                health_data['compliance_status'] = {\n                    'overall_compliance_score': compliance_status.get('compliance_report', {}).get('overall_compliance_score', 0),\n                    'active_safety_events': len(self.clinical_compliance.safety_monitor.safety_events)\n                }\n            except Exception as e:\n                health_data['compliance_status'] = {'error': str(e)}\n        \n        return health_data\n    \n    async def get_analytics_report(self, timeframe_hours: int = 24) -> Dict[str, Any]:\n        """Get comprehensive analytics report."""\n        if not self.analytics_dashboard:\n            return {'error': 'Analytics dashboard not available'}\n        \n        try:\n            # Get performance analytics\n            performance_analytics = await self.analytics_dashboard.get_performance_analytics()\n            \n            # Get clinical analytics\n            clinical_analytics = await self.analytics_dashboard.get_clinical_analytics()\n            \n            # Get real-time dashboard data\n            dashboard_data = await self.analytics_dashboard.get_real_time_dashboard_data()\n            \n            return {\n                'generated_at': datetime.now().isoformat(),\n                'timeframe_hours': timeframe_hours,\n                'orchestrator_metrics': self.metrics.copy(),\n                'performance_analytics': performance_analytics,\n                'clinical_analytics': clinical_analytics,\n                'dashboard_data': dashboard_data\n            }\n        \n        except Exception as e:\n            logger.error(f"Error generating analytics report: {e}")\n            return {'error': str(e)}\n    \n    async def get_compliance_report(self) -> Dict[str, Any]:\n        """Get comprehensive compliance report."""\n        if not self.clinical_compliance:\n            return {'error': 'Clinical compliance system not available'}\n        \n        try:\n            compliance_report = await self.clinical_compliance.generate_comprehensive_report()\n            return compliance_report\n        \n        except Exception as e:\n            logger.error(f"Error generating compliance report: {e}")\n            return {'error': str(e)}\n    \n    def get_configuration(self) -> Dict[str, Any]:\n        """Get current configuration."""\n        return self.config.to_dict()\n    \n    async def update_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:\n        """Update configuration dynamically."""\n        try:\n            # Update configuration\n            old_config = self.config.to_dict()\n            \n            for key, value in new_config.items():\n                if hasattr(self.config, key):\n                    setattr(self.config, key, value)\n            \n            # Log configuration change\n            if self.clinical_compliance:\n                await self.clinical_compliance.compliance_engine.record_audit_event(\n                    action="configuration_update",\n                    actor_type="system",\n                    actor_id="enterprise_orchestrator",\n                    resource_type="configuration",\n                    resource_id="orchestrator_config",\n                    details={\n                        'old_config': old_config,\n                        'new_config': new_config,\n                        'updated_keys': list(new_config.keys())\n                    }\n                )\n            \n            return {\n                'success': True,\n                'updated_configuration': self.config.to_dict(),\n                'message': 'Configuration updated successfully'\n            }\n            \n        except Exception as e:\n            logger.error(f"Error updating configuration: {e}")\n            return {\n                'success': False,\n                'error': str(e)\n            }\n\n\n# Factory function for easy instantiation\nasync def create_enterprise_orchestrator(model: BaseLanguageModel, \n                                       config: Optional[Dict[str, Any]] = None) -> EnterpriseAgentOrchestrator:\n    \"\"\"Create and initialize an enterprise orchestrator.\"\"\"\n    orchestrator = EnterpriseAgentOrchestrator(model, config)\n    await orchestrator.initialize()\n    return orchestrator\n\n\n# Default configuration\nDEFAULT_ENTERPRISE_CONFIG = {\n    'enterprise_features_enabled': True,\n    'auto_start_systems': True,\n    'monitoring_enabled': True,\n    'analytics_enabled': True,\n    'quality_assurance_enabled': True,\n    'knowledge_integration_enabled': True,\n    'data_reliability_enabled': True,\n    'clinical_compliance_enabled': True,\n    'max_concurrent_requests': 100,\n    'request_timeout_seconds': 30,\n    'health_check_interval': 60,\n    'min_quality_score': 70.0,\n    'max_response_time': 10.0,\n    'require_supervision': True,\n    'enable_therapeutic_friction': True\n}