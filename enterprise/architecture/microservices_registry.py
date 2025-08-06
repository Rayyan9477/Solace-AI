"""
Enterprise Microservices Registry and Service Discovery
Manages all microservices in the Solace-AI enterprise platform
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import uuid
from datetime import datetime, timedelta
import aiohttp
import socket

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"


class ServiceType(Enum):
    API_GATEWAY = "api_gateway"
    MEMORY_SERVICE = "memory_service"
    RESEARCH_SERVICE = "research_service"
    ANALYTICS_SERVICE = "analytics_service"
    SECURITY_SERVICE = "security_service"
    CLINICAL_SERVICE = "clinical_service"
    NOTIFICATION_SERVICE = "notification_service"
    AUDIT_SERVICE = "audit_service"
    ML_SERVICE = "ml_service"
    STORAGE_SERVICE = "storage_service"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    method: str = "GET"
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str
    interval: int = 30
    timeout: int = 10
    retries: int = 3
    expected_status: int = 200


@dataclass
class ServiceRegistration:
    """Service registration information"""
    service_id: str
    service_name: str
    service_type: ServiceType
    version: str
    host: str
    port: int
    endpoints: List[ServiceEndpoint]
    health_check: HealthCheck
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    status: ServiceStatus = ServiceStatus.STARTING
    dependencies: List[str] = field(default_factory=list)


class ServiceRegistry:
    """
    Central service registry for microservices discovery and management
    """
    
    def __init__(self, cleanup_interval: int = 60):
        self.services: Dict[str, ServiceRegistration] = {}
        self.service_types: Dict[ServiceType, List[str]] = {}
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the service registry"""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Service registry started")
        
    async def stop(self):
        """Stop the service registry"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Service registry stopped")
        
    async def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a new service"""
        try:
            # Validate service
            if not await self._validate_service(registration):
                return False
                
            # Store service
            self.services[registration.service_id] = registration
            
            # Update service type mapping
            if registration.service_type not in self.service_types:
                self.service_types[registration.service_type] = []
            self.service_types[registration.service_type].append(registration.service_id)
            
            logger.info(f"Registered service: {registration.service_name} ({registration.service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {registration.service_name}: {e}")
            return False
            
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        try:
            if service_id not in self.services:
                return False
                
            service = self.services[service_id]
            
            # Remove from service type mapping
            if service.service_type in self.service_types:
                self.service_types[service.service_type].remove(service_id)
                if not self.service_types[service.service_type]:
                    del self.service_types[service.service_type]
                    
            # Remove service
            del self.services[service_id]
            
            logger.info(f"Deregistered service: {service.service_name} ({service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
            
    async def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat"""
        if service_id not in self.services:
            return False
            
        self.services[service_id].last_heartbeat = datetime.utcnow()
        return True
        
    async def get_service(self, service_id: str) -> Optional[ServiceRegistration]:
        """Get service by ID"""
        return self.services.get(service_id)
        
    async def get_services_by_type(self, service_type: ServiceType) -> List[ServiceRegistration]:
        """Get all services of a specific type"""
        service_ids = self.service_types.get(service_type, [])
        return [self.services[sid] for sid in service_ids if sid in self.services]
        
    async def get_healthy_services(self, service_type: ServiceType) -> List[ServiceRegistration]:
        """Get healthy services of a specific type"""
        services = await self.get_services_by_type(service_type)
        return [s for s in services if s.status == ServiceStatus.HEALTHY]
        
    async def discover_service(self, service_name: str) -> Optional[ServiceRegistration]:
        """Discover a service by name"""
        for service in self.services.values():
            if service.service_name == service_name and service.status == ServiceStatus.HEALTHY:
                return service
        return None
        
    async def health_check_service(self, service_id: str) -> bool:
        """Perform health check on a service"""
        if service_id not in self.services:
            return False
            
        service = self.services[service_id]
        
        try:
            health_url = f"http://{service.host}:{service.port}{service.health_check.endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=service.health_check.timeout)
                ) as response:
                    if response.status == service.health_check.expected_status:
                        service.status = ServiceStatus.HEALTHY
                        service.last_heartbeat = datetime.utcnow()
                        return True
                    else:
                        service.status = ServiceStatus.UNHEALTHY
                        return False
                        
        except Exception as e:
            logger.warning(f"Health check failed for {service.service_name}: {e}")
            service.status = ServiceStatus.UNHEALTHY
            return False
            
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all services"""
        results = {}
        tasks = []
        
        for service_id in self.services.keys():
            task = asyncio.create_task(self.health_check_service(service_id))
            tasks.append((service_id, task))
            
        for service_id, task in tasks:
            try:
                results[service_id] = await task
            except Exception as e:
                logger.error(f"Health check error for {service_id}: {e}")
                results[service_id] = False
                
        return results
        
    async def get_service_dependencies(self, service_id: str) -> List[ServiceRegistration]:
        """Get service dependencies"""
        if service_id not in self.services:
            return []
            
        service = self.services[service_id]
        dependencies = []
        
        for dep_id in service.dependencies:
            if dep_id in self.services:
                dependencies.append(self.services[dep_id])
                
        return dependencies
        
    async def check_dependencies(self, service_id: str) -> bool:
        """Check if all service dependencies are healthy"""
        dependencies = await self.get_service_dependencies(service_id)
        
        for dep in dependencies:
            if dep.status != ServiceStatus.HEALTHY:
                return False
                
        return True
        
    async def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status"""
        total_services = len(self.services)
        healthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
        
        service_types_count = {}
        for service_type, service_ids in self.service_types.items():
            service_types_count[service_type.value] = len(service_ids)
            
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "service_types": service_types_count,
            "registry_started": self._running,
            "last_cleanup": datetime.utcnow().isoformat()
        }
        
    async def _validate_service(self, registration: ServiceRegistration) -> bool:
        """Validate service registration"""
        # Check if port is available
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((registration.host, registration.port))
            sock.close()
            
            if result != 0:
                logger.warning(f"Service {registration.service_name} port {registration.port} not accessible")
                return False
                
        except Exception as e:
            logger.error(f"Port validation failed for {registration.service_name}: {e}")
            return False
            
        return True
        
    async def _cleanup_loop(self):
        """Cleanup stale services"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_services()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                
    async def _cleanup_stale_services(self):
        """Remove stale services"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        stale_services = []
        
        for service_id, service in self.services.items():
            if service.last_heartbeat < cutoff_time and service.status != ServiceStatus.MAINTENANCE:
                stale_services.append(service_id)
                
        for service_id in stale_services:
            logger.info(f"Removing stale service: {service_id}")
            await self.deregister_service(service_id)


class ServiceDiscoveryClient:
    """
    Client for service discovery and load balancing
    """
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.service_cache: Dict[str, List[ServiceRegistration]] = {}
        self.cache_ttl = 30  # seconds
        self.last_cache_update: Dict[str, datetime] = {}
        
    async def discover(self, service_type: ServiceType, refresh_cache: bool = False) -> List[ServiceRegistration]:
        """Discover services of a specific type"""
        cache_key = service_type.value
        
        # Check cache
        if not refresh_cache and cache_key in self.service_cache:
            last_update = self.last_cache_update.get(cache_key, datetime.min)
            if (datetime.utcnow() - last_update).seconds < self.cache_ttl:
                return self.service_cache[cache_key]
                
        # Fetch from registry
        services = await self.registry.get_healthy_services(service_type)
        
        # Update cache
        self.service_cache[cache_key] = services
        self.last_cache_update[cache_key] = datetime.utcnow()
        
        return services
        
    async def get_service_url(self, service_type: ServiceType, endpoint_name: str = "default") -> Optional[str]:
        """Get service URL with load balancing"""
        services = await self.discover(service_type)
        
        if not services:
            return None
            
        # Simple round-robin load balancing
        service = services[int(time.time()) % len(services)]
        
        # Find endpoint
        for endpoint in service.endpoints:
            if endpoint.name == endpoint_name:
                return f"http://{service.host}:{service.port}{endpoint.url}"
                
        # Default to first endpoint
        if service.endpoints:
            endpoint = service.endpoints[0]
            return f"http://{service.host}:{service.port}{endpoint.url}"
            
        return f"http://{service.host}:{service.port}"
        
    async def call_service(self, service_type: ServiceType, endpoint_name: str, 
                          method: str = "GET", data: Any = None, 
                          headers: Dict[str, str] = None) -> Any:
        """Make a call to a service"""
        url = await self.get_service_url(service_type, endpoint_name)
        
        if not url:
            raise Exception(f"No healthy services found for {service_type.value}")
            
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                headers=headers or {}
            ) as response:
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    return await response.text()


# Global registry instance
_global_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def get_discovery_client() -> ServiceDiscoveryClient:
    """Get the service discovery client"""
    registry = get_service_registry()
    return ServiceDiscoveryClient(registry)