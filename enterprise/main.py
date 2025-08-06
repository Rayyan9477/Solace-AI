"""
Enterprise Solace-AI Integration Main Entry Point
Initializes and coordinates all enterprise-level components
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import signal
import sys
import os
from pathlib import Path

# Enterprise components
from .architecture.microservices_registry import get_service_registry, get_discovery_client
from .memory.semantic_network import SemanticMemoryNetwork, PostgreSQLMemoryStore
from .memory.episodic_memory import EpisodicMemoryManager
from .research.literature_monitor import LiteratureMonitor
from .analytics.predictive_analytics import PredictiveAnalyticsEngine
from .analytics.population_health import PopulationHealthAnalyzer
from .security.hipaa_compliance import HIPAAComplianceManager
from .infrastructure.performance_optimization import PerformanceOptimizer
from .clinical.ehr_integration import EHRIntegrationManager
from .clinical.telehealth_integration import TelehealthIntegrationManager
from .testing.test_framework import ComprehensiveTestRunner, TestEnvironment
from .monitoring.comprehensive_monitoring import get_monitoring_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_solace_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class EnterpriseSolaceAI:
    """
    Main Enterprise Solace-AI Platform
    Coordinates all enterprise components for clinical deployment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.components = {}
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize core components
        self.service_registry = get_service_registry()
        self.discovery_client = get_discovery_client()
        self.monitoring_system = get_monitoring_system()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load enterprise configuration"""
        default_config = {
            "database": {
                "postgresql_url": "postgresql://localhost:5432/solace_ai_enterprise",
                "redis_url": "redis://localhost:6379"
            },
            "security": {
                "encryption_key_rotation_days": 90,
                "session_timeout_minutes": 15,
                "max_failed_login_attempts": 5
            },
            "monitoring": {
                "metrics_retention_days": 30,
                "alert_cooldown_minutes": 15,
                "health_check_interval_seconds": 60
            },
            "memory": {
                "consolidation_interval_hours": 24,
                "decay_factor": 0.95,
                "max_episodic_memory_days": 90
            },
            "research": {
                "literature_update_interval_hours": 6,
                "max_papers_per_search": 50,
                "research_email": "research@solace-ai.com"
            },
            "clinical": {
                "ehr_systems": [],
                "telehealth_platforms": [],
                "default_session_duration_minutes": 60
            },
            "analytics": {
                "prediction_batch_size": 100,
                "model_retrain_days": 30,
                "population_analysis_enabled": True
            },
            "performance": {
                "cache_size_mb": 512,
                "connection_pool_size": 20,
                "load_balancing_strategy": "least_connections"
            },
            "testing": {
                "auto_run_tests": False,
                "test_data_retention_days": 7,
                "performance_test_enabled": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
        
    async def initialize(self):
        """Initialize all enterprise components"""
        logger.info("Initializing Enterprise Solace-AI Platform...")
        
        try:
            # 1. Initialize Service Registry
            await self.service_registry.start()
            logger.info("✓ Service Registry initialized")
            
            # 2. Initialize Memory Systems
            await self._initialize_memory_systems()
            logger.info("✓ Memory Systems initialized")
            
            # 3. Initialize Security & Compliance
            await self._initialize_security_compliance()
            logger.info("✓ Security & Compliance initialized")
            
            # 4. Initialize Research Integration
            await self._initialize_research_integration()
            logger.info("✓ Research Integration initialized")
            
            # 5. Initialize Analytics
            await self._initialize_analytics()
            logger.info("✓ Analytics initialized")
            
            # 6. Initialize Clinical Integration
            await self._initialize_clinical_integration()
            logger.info("✓ Clinical Integration initialized")
            
            # 7. Initialize Performance Optimization
            await self._initialize_performance_optimization()
            logger.info("✓ Performance Optimization initialized")
            
            # 8. Initialize Testing Framework
            await self._initialize_testing_framework()
            logger.info("✓ Testing Framework initialized")
            
            # 9. Initialize Monitoring & Alerting
            await self._initialize_monitoring()
            logger.info("✓ Monitoring & Alerting initialized")
            
            # 10. Register all services
            await self._register_services()
            logger.info("✓ Services registered")
            
            logger.info("🚀 Enterprise Solace-AI Platform initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enterprise Platform: {e}")
            raise
            
    async def _initialize_memory_systems(self):
        """Initialize memory systems"""
        # Semantic Memory
        memory_store = PostgreSQLMemoryStore(
            self.config["database"]["postgresql_url"]
        )
        await memory_store.initialize()
        
        semantic_memory = SemanticMemoryNetwork(memory_store)
        await semantic_memory.initialize()
        self.components["semantic_memory"] = semantic_memory
        
        # Episodic Memory
        episodic_memory = EpisodicMemoryManager(
            self.config["database"]["redis_url"]
        )
        await episodic_memory.initialize()
        self.components["episodic_memory"] = episodic_memory
        
    async def _initialize_security_compliance(self):
        """Initialize security and compliance"""
        hipaa_manager = HIPAAComplianceManager()
        await hipaa_manager.initialize()
        self.components["hipaa_compliance"] = hipaa_manager
        
    async def _initialize_research_integration(self):
        """Initialize research integration"""
        literature_monitor = LiteratureMonitor(
            email=self.config["research"]["research_email"],
            update_interval=self.config["research"]["literature_update_interval_hours"] * 3600
        )
        self.components["literature_monitor"] = literature_monitor
        
    async def _initialize_analytics(self):
        """Initialize analytics systems"""
        # Predictive Analytics
        predictive_analytics = PredictiveAnalyticsEngine()
        self.components["predictive_analytics"] = predictive_analytics
        
        # Population Health Analytics
        population_analytics = PopulationHealthAnalyzer()
        self.components["population_analytics"] = population_analytics
        
    async def _initialize_clinical_integration(self):
        """Initialize clinical integration"""
        # EHR Integration
        ehr_manager = EHRIntegrationManager()
        self.components["ehr_integration"] = ehr_manager
        
        # Telehealth Integration
        telehealth_manager = TelehealthIntegrationManager()
        self.components["telehealth_integration"] = telehealth_manager
        
    async def _initialize_performance_optimization(self):
        """Initialize performance optimization"""
        performance_optimizer = PerformanceOptimizer()
        await performance_optimizer.initialize()
        self.components["performance_optimizer"] = performance_optimizer
        
    async def _initialize_testing_framework(self):
        """Initialize testing framework"""
        test_environment = TestEnvironment(
            environment_id="enterprise_env",
            name="Enterprise Environment",
            base_url="http://localhost:8000",
            database_config=self.config["database"],
            external_services={},
            test_data_path="./test_data"
        )
        
        test_runner = ComprehensiveTestRunner()
        await test_runner.initialize(test_environment)
        self.components["test_runner"] = test_runner
        
    async def _initialize_monitoring(self):
        """Initialize monitoring and alerting"""
        await self.monitoring_system.initialize()
        await self.monitoring_system.start()
        
    async def _register_services(self):
        """Register all services with service registry"""
        from .architecture.microservices_registry import ServiceRegistration, ServiceType, ServiceEndpoint, HealthCheck
        
        # Register core services
        services_to_register = [
            {
                "service_name": "memory_service",
                "service_type": ServiceType.MEMORY_SERVICE,
                "host": "localhost",
                "port": 8001,
                "endpoints": [
                    ServiceEndpoint("store_memory", "/api/memory/store", "POST"),
                    ServiceEndpoint("retrieve_memory", "/api/memory/retrieve", "GET"),
                    ServiceEndpoint("health", "/health", "GET")
                ]
            },
            {
                "service_name": "analytics_service",
                "service_type": ServiceType.ANALYTICS_SERVICE,
                "host": "localhost", 
                "port": 8002,
                "endpoints": [
                    ServiceEndpoint("predict", "/api/analytics/predict", "POST"),
                    ServiceEndpoint("population_analysis", "/api/analytics/population", "GET"),
                    ServiceEndpoint("health", "/health", "GET")
                ]
            },
            {
                "service_name": "clinical_service",
                "service_type": ServiceType.CLINICAL_SERVICE,
                "host": "localhost",
                "port": 8003,
                "endpoints": [
                    ServiceEndpoint("ehr_data", "/api/clinical/ehr", "GET"),
                    ServiceEndpoint("schedule_session", "/api/clinical/telehealth/schedule", "POST"),
                    ServiceEndpoint("health", "/health", "GET")
                ]
            }
        ]
        
        for service_config in services_to_register:
            registration = ServiceRegistration(
                service_id=str(datetime.utcnow().timestamp()),
                service_name=service_config["service_name"],
                service_type=service_config["service_type"],
                version="1.0.0",
                host=service_config["host"],
                port=service_config["port"],
                endpoints=service_config["endpoints"],
                health_check=HealthCheck(
                    endpoint="/health",
                    interval=60,
                    timeout=10
                ),
                metadata={"environment": "enterprise", "deployment": "production"}
            )
            
            await self.service_registry.register_service(registration)
            
    async def start(self):
        """Start the enterprise platform"""
        if self.running:
            logger.warning("Platform is already running")
            return
            
        logger.info("Starting Enterprise Solace-AI Platform...")
        self.running = True
        
        try:
            # Start literature monitoring
            if "literature_monitor" in self.components:
                asyncio.create_task(
                    self.components["literature_monitor"].start_monitoring()
                )
                
            # Start periodic tasks
            asyncio.create_task(self._periodic_maintenance())
            asyncio.create_task(self._health_monitoring())
            
            logger.info("🎯 Enterprise Solace-AI Platform is now running!")
            logger.info("Platform Status: OPERATIONAL")
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
        except Exception as e:
            logger.error(f"Failed to start platform: {e}")
            await self.stop()
            raise
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())
        
    async def stop(self):
        """Stop the enterprise platform"""
        if not self.running:
            return
            
        logger.info("Stopping Enterprise Solace-AI Platform...")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Stop monitoring first
            await self.monitoring_system.stop()
            
            # Stop literature monitoring
            if "literature_monitor" in self.components:
                await self.components["literature_monitor"].stop_monitoring()
                
            # Stop performance optimizer
            if "performance_optimizer" in self.components:
                await self.components["performance_optimizer"].shutdown()
                
            # Stop service registry
            await self.service_registry.stop()
            
            logger.info("✓ Enterprise Solace-AI Platform stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during platform shutdown: {e}")
            
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        while self.running:
            try:
                logger.info("Running periodic maintenance...")
                
                # Memory consolidation
                if "semantic_memory" in self.components:
                    await self.components["semantic_memory"].consolidate_memories()
                    
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Model retraining check
                await self._check_model_retraining()
                
                # Wait 24 hours before next maintenance
                await asyncio.sleep(24 * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic maintenance: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
                
    async def _health_monitoring(self):
        """Monitor overall platform health"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get system status
                status = await self.get_platform_status()
                
                if status["overall_health"] == "critical":
                    logger.error("CRITICAL: Platform health is critical!")
                elif status["overall_health"] == "unhealthy":
                    logger.warning("WARNING: Platform health is degraded")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                
    async def _cleanup_old_data(self):
        """Cleanup old data based on retention policies"""
        logger.info("Cleaning up old data...")
        
        # Cleanup test data
        if self.config["testing"]["test_data_retention_days"] > 0:
            # Would implement actual cleanup logic
            pass
            
        # Cleanup old metrics
        # This would be handled by the monitoring system
        
    async def _check_model_retraining(self):
        """Check if models need retraining"""
        retrain_interval = self.config["analytics"]["model_retrain_days"]
        
        if "predictive_analytics" in self.components:
            # Check if models need retraining
            # This would check model performance metrics and decide
            logger.info("Checking if models need retraining...")
            
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform_running": self.running,
            "uptime": "N/A",  # Would calculate actual uptime
            "version": "1.0.0",
            "environment": "enterprise",
            "components": {},
            "overall_health": "unknown",
            "active_alerts": 0,
            "total_users": 0,
            "active_sessions": 0
        }
        
        try:
            # Get monitoring system status
            monitoring_status = await self.monitoring_system.get_monitoring_status()
            status["monitoring"] = monitoring_status
            
            # Get service registry status  
            registry_status = await self.service_registry.get_registry_status()
            status["service_registry"] = registry_status
            
            # Get component statuses
            for name, component in self.components.items():
                if hasattr(component, 'get_status'):
                    status["components"][name] = await component.get_status()
                else:
                    status["components"][name] = {"status": "running"}
                    
            # Calculate overall health
            healthy_components = 0
            total_components = len(self.components)
            
            for comp_status in status["components"].values():
                if comp_status.get("status") == "healthy" or comp_status.get("status") == "running":
                    healthy_components += 1
                    
            if healthy_components == total_components:
                status["overall_health"] = "healthy"
            elif healthy_components >= total_components * 0.8:
                status["overall_health"] = "degraded"
            elif healthy_components >= total_components * 0.5:
                status["overall_health"] = "unhealthy" 
            else:
                status["overall_health"] = "critical"
                
            # Get active alerts count
            status["active_alerts"] = len(self.monitoring_system.alert_manager.active_alerts)
            
        except Exception as e:
            logger.error(f"Error getting platform status: {e}")
            status["error"] = str(e)
            
        return status
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive platform tests"""
        if "test_runner" not in self.components:
            return {"error": "Test runner not initialized"}
            
        logger.info("Running comprehensive platform tests...")
        
        test_runner = self.components["test_runner"]
        results = await test_runner.run_all_tests()
        
        logger.info(f"Test run completed: {results['overall_summary']['passed_tests']} passed, "
                   f"{results['overall_summary']['failed_tests']} failed")
        
        return results
        
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()


async def main():
    """Main entry point for Enterprise Solace-AI"""
    
    # Create and initialize the platform
    platform = EnterpriseSolaceAI()
    
    try:
        await platform.initialize()
        await platform.start()
        
        # Keep running until shutdown signal
        await platform.wait_for_shutdown()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Platform error: {e}")
    finally:
        await platform.stop()


if __name__ == "__main__":
    # Set up event loop policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    # Run the platform
    asyncio.run(main())