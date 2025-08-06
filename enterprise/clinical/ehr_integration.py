"""
Electronic Health Record (EHR) Integration
FHIR-compliant integration with major EHR systems
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import aiohttp
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
import uuid
import base64
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class FHIRResourceType(Enum):
    PATIENT = "Patient"
    PRACTITIONER = "Practitioner"
    ENCOUNTER = "Encounter"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_REQUEST = "MedicationRequest"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    CARE_PLAN = "CarePlan"
    APPOINTMENT = "Appointment"
    DOCUMENT_REFERENCE = "DocumentReference"


class EHRSystem(Enum):
    EPIC = "epic"
    CERNER = "cerner"
    ALLSCRIPTS = "allscripts"
    ATHENA = "athena"
    ECLINICALWORKS = "eclinicalworks"
    GREENWAY = "greenway"
    GENERIC_FHIR = "generic_fhir"


@dataclass
class FHIRPatient:
    """FHIR Patient Resource"""
    id: str
    identifier: List[Dict[str, str]]
    active: bool
    name: List[Dict[str, Any]]
    telecom: List[Dict[str, str]]
    gender: str
    birth_date: Optional[date]
    deceased: Optional[Union[bool, datetime]]
    address: List[Dict[str, str]]
    marital_status: Optional[Dict[str, str]]
    contact: List[Dict[str, Any]]
    communication: List[Dict[str, Any]]
    general_practitioner: List[Dict[str, str]]
    managing_organization: Optional[Dict[str, str]]
    extension: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FHIRObservation:
    """FHIR Observation Resource"""
    id: str
    status: str
    category: List[Dict[str, Any]]
    code: Dict[str, Any]
    subject: Dict[str, str]
    encounter: Optional[Dict[str, str]]
    effective_datetime: Optional[datetime]
    issued: Optional[datetime]
    performer: List[Dict[str, str]]
    value: Optional[Union[Dict[str, Any], str, bool, int, float]]
    interpretation: List[Dict[str, Any]]
    note: List[Dict[str, str]]
    reference_range: List[Dict[str, Any]]
    component: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FHIRCondition:
    """FHIR Condition Resource"""
    id: str
    clinical_status: Dict[str, str]
    verification_status: Dict[str, str]
    category: List[Dict[str, Any]]
    severity: Optional[Dict[str, Any]]
    code: Dict[str, Any]
    subject: Dict[str, str]
    encounter: Optional[Dict[str, str]]
    onset: Optional[Union[datetime, str, Dict[str, Any]]]
    recorded_date: Optional[datetime]
    recorder: Optional[Dict[str, str]]
    asserter: Optional[Dict[str, str]]
    note: List[Dict[str, str]]
    evidence: List[Dict[str, Any]] = field(default_factory=list)


class FHIRClient(ABC):
    """Abstract FHIR client interface"""
    
    @abstractmethod
    async def get_patient(self, patient_id: str) -> Optional[FHIRPatient]:
        pass
    
    @abstractmethod
    async def search_patients(self, criteria: Dict[str, str]) -> List[FHIRPatient]:
        pass
    
    @abstractmethod
    async def get_observations(self, patient_id: str, 
                             category: Optional[str] = None) -> List[FHIRObservation]:
        pass
    
    @abstractmethod
    async def get_conditions(self, patient_id: str) -> List[FHIRCondition]:
        pass
    
    @abstractmethod
    async def create_observation(self, observation: FHIRObservation) -> str:
        pass
    
    @abstractmethod
    async def create_condition(self, condition: FHIRCondition) -> str:
        pass


class GenericFHIRClient(FHIRClient):
    """Generic FHIR R4 client implementation"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any]):
        self.base_url = base_url.rstrip('/')
        self.auth_config = auth_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize FHIR client"""
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        logger.info(f"Initialized FHIR client for {self.base_url}")
        
    async def _authenticate(self):
        """Authenticate with FHIR server"""
        if self.auth_config.get("type") == "oauth2":
            await self._oauth2_authenticate()
        elif self.auth_config.get("type") == "basic":
            # Basic auth doesn't need token refresh
            pass
        elif self.auth_config.get("type") == "api_key":
            # API key auth doesn't need token refresh
            pass
            
    async def _oauth2_authenticate(self):
        """OAuth2 authentication flow"""
        try:
            token_url = self.auth_config.get("token_url")
            client_id = self.auth_config.get("client_id")
            client_secret = self.auth_config.get("client_secret")
            scope = self.auth_config.get("scope", "system/*.read system/*.write")
            
            auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials",
                "scope": scope
            }
            
            async with self.session.post(token_url, headers=headers, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expires = datetime.utcnow().timestamp() + expires_in
                    logger.info("FHIR OAuth2 authentication successful")
                else:
                    error = await response.text()
                    raise Exception(f"OAuth2 authentication failed: {error}")
                    
        except Exception as e:
            logger.error(f"FHIR authentication error: {e}")
            raise
            
    async def _get_headers(self) -> Dict[str, str]:
        """Get authenticated request headers"""
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        }
        
        # Check if token needs refresh
        if (self.auth_config.get("type") == "oauth2" and 
            self.token_expires and 
            datetime.utcnow().timestamp() > self.token_expires - 300):  # Refresh 5 min early
            await self._authenticate()
            
        if self.auth_config.get("type") == "oauth2" and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif self.auth_config.get("type") == "basic":
            username = self.auth_config.get("username")
            password = self.auth_config.get("password")
            auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_string}"
        elif self.auth_config.get("type") == "api_key":
            api_key = self.auth_config.get("api_key")
            key_header = self.auth_config.get("key_header", "X-API-Key")
            headers[key_header] = api_key
            
        return headers
        
    async def get_patient(self, patient_id: str) -> Optional[FHIRPatient]:
        """Get patient by ID"""
        try:
            url = f"{self.base_url}/Patient/{patient_id}"
            headers = await self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_patient(data)
                elif response.status == 404:
                    return None
                else:
                    error = await response.text()
                    logger.error(f"Error getting patient {patient_id}: {error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting patient {patient_id}: {e}")
            return None
            
    async def search_patients(self, criteria: Dict[str, str]) -> List[FHIRPatient]:
        """Search patients with criteria"""
        try:
            url = f"{self.base_url}/Patient"
            headers = await self._get_headers()
            
            # Build query parameters
            params = {}
            for key, value in criteria.items():
                params[key] = value
                
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    patients = []
                    
                    if data.get("resourceType") == "Bundle":
                        for entry in data.get("entry", []):
                            if entry.get("resource", {}).get("resourceType") == "Patient":
                                patient = self._parse_patient(entry["resource"])
                                if patient:
                                    patients.append(patient)
                                    
                    return patients
                else:
                    error = await response.text()
                    logger.error(f"Error searching patients: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching patients: {e}")
            return []
            
    async def get_observations(self, patient_id: str, 
                             category: Optional[str] = None) -> List[FHIRObservation]:
        """Get observations for a patient"""
        try:
            url = f"{self.base_url}/Observation"
            headers = await self._get_headers()
            
            params = {"subject": f"Patient/{patient_id}"}
            if category:
                params["category"] = category
                
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = []
                    
                    if data.get("resourceType") == "Bundle":
                        for entry in data.get("entry", []):
                            if entry.get("resource", {}).get("resourceType") == "Observation":
                                observation = self._parse_observation(entry["resource"])
                                if observation:
                                    observations.append(observation)
                                    
                    return observations
                else:
                    error = await response.text()
                    logger.error(f"Error getting observations for {patient_id}: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting observations for {patient_id}: {e}")
            return []
            
    async def get_conditions(self, patient_id: str) -> List[FHIRCondition]:
        """Get conditions for a patient"""
        try:
            url = f"{self.base_url}/Condition"
            headers = await self._get_headers()
            
            params = {"subject": f"Patient/{patient_id}"}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    conditions = []
                    
                    if data.get("resourceType") == "Bundle":
                        for entry in data.get("entry", []):
                            if entry.get("resource", {}).get("resourceType") == "Condition":
                                condition = self._parse_condition(entry["resource"])
                                if condition:
                                    conditions.append(condition)
                                    
                    return conditions
                else:
                    error = await response.text()
                    logger.error(f"Error getting conditions for {patient_id}: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting conditions for {patient_id}: {e}")
            return []
            
    async def create_observation(self, observation: FHIRObservation) -> str:
        """Create a new observation"""
        try:
            url = f"{self.base_url}/Observation"
            headers = await self._get_headers()
            
            observation_data = self._observation_to_fhir(observation)
            
            async with self.session.post(url, headers=headers, json=observation_data) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    return data.get("id", "")
                else:
                    error = await response.text()
                    logger.error(f"Error creating observation: {error}")
                    raise Exception(f"Failed to create observation: {error}")
                    
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            raise
            
    async def create_condition(self, condition: FHIRCondition) -> str:
        """Create a new condition"""
        try:
            url = f"{self.base_url}/Condition"
            headers = await self._get_headers()
            
            condition_data = self._condition_to_fhir(condition)
            
            async with self.session.post(url, headers=headers, json=condition_data) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    return data.get("id", "")
                else:
                    error = await response.text()
                    logger.error(f"Error creating condition: {error}")
                    raise Exception(f"Failed to create condition: {error}")
                    
        except Exception as e:
            logger.error(f"Error creating condition: {e}")
            raise
            
    def _parse_patient(self, data: Dict[str, Any]) -> Optional[FHIRPatient]:
        """Parse FHIR Patient resource"""
        try:
            return FHIRPatient(
                id=data.get("id", ""),
                identifier=data.get("identifier", []),
                active=data.get("active", True),
                name=data.get("name", []),
                telecom=data.get("telecom", []),
                gender=data.get("gender", ""),
                birth_date=datetime.strptime(data["birthDate"], "%Y-%m-%d").date() 
                          if data.get("birthDate") else None,
                deceased=data.get("deceased"),
                address=data.get("address", []),
                marital_status=data.get("maritalStatus"),
                contact=data.get("contact", []),
                communication=data.get("communication", []),
                general_practitioner=data.get("generalPractitioner", []),
                managing_organization=data.get("managingOrganization"),
                extension=data.get("extension", [])
            )
        except Exception as e:
            logger.error(f"Error parsing patient data: {e}")
            return None
            
    def _parse_observation(self, data: Dict[str, Any]) -> Optional[FHIRObservation]:
        """Parse FHIR Observation resource"""
        try:
            return FHIRObservation(
                id=data.get("id", ""),
                status=data.get("status", ""),
                category=data.get("category", []),
                code=data.get("code", {}),
                subject=data.get("subject", {}),
                encounter=data.get("encounter"),
                effective_datetime=datetime.fromisoformat(data["effectiveDateTime"].replace("Z", "+00:00"))
                                 if data.get("effectiveDateTime") else None,
                issued=datetime.fromisoformat(data["issued"].replace("Z", "+00:00"))
                      if data.get("issued") else None,
                performer=data.get("performer", []),
                value=data.get("valueQuantity") or data.get("valueString") or 
                      data.get("valueBoolean") or data.get("valueCodeableConcept"),
                interpretation=data.get("interpretation", []),
                note=data.get("note", []),
                reference_range=data.get("referenceRange", []),
                component=data.get("component", [])
            )
        except Exception as e:
            logger.error(f"Error parsing observation data: {e}")
            return None
            
    def _parse_condition(self, data: Dict[str, Any]) -> Optional[FHIRCondition]:
        """Parse FHIR Condition resource"""
        try:
            return FHIRCondition(
                id=data.get("id", ""),
                clinical_status=data.get("clinicalStatus", {}),
                verification_status=data.get("verificationStatus", {}),
                category=data.get("category", []),
                severity=data.get("severity"),
                code=data.get("code", {}),
                subject=data.get("subject", {}),
                encounter=data.get("encounter"),
                onset=data.get("onsetDateTime") or data.get("onsetString") or data.get("onsetPeriod"),
                recorded_date=datetime.fromisoformat(data["recordedDate"].replace("Z", "+00:00"))
                            if data.get("recordedDate") else None,
                recorder=data.get("recorder"),
                asserter=data.get("asserter"),
                note=data.get("note", []),
                evidence=data.get("evidence", [])
            )
        except Exception as e:
            logger.error(f"Error parsing condition data: {e}")
            return None
            
    def _observation_to_fhir(self, observation: FHIRObservation) -> Dict[str, Any]:
        """Convert FHIRObservation to FHIR JSON"""
        data = {
            "resourceType": "Observation",
            "status": observation.status,
            "category": observation.category,
            "code": observation.code,
            "subject": observation.subject
        }
        
        if observation.encounter:
            data["encounter"] = observation.encounter
            
        if observation.effective_datetime:
            data["effectiveDateTime"] = observation.effective_datetime.isoformat()
            
        if observation.issued:
            data["issued"] = observation.issued.isoformat()
            
        if observation.performer:
            data["performer"] = observation.performer
            
        if observation.value:
            if isinstance(observation.value, dict):
                if "value" in observation.value and "unit" in observation.value:
                    data["valueQuantity"] = observation.value
                else:
                    data["valueCodeableConcept"] = observation.value
            elif isinstance(observation.value, str):
                data["valueString"] = observation.value
            elif isinstance(observation.value, bool):
                data["valueBoolean"] = observation.value
            elif isinstance(observation.value, (int, float)):
                data["valueQuantity"] = {"value": observation.value}
                
        if observation.interpretation:
            data["interpretation"] = observation.interpretation
            
        if observation.note:
            data["note"] = observation.note
            
        if observation.reference_range:
            data["referenceRange"] = observation.reference_range
            
        if observation.component:
            data["component"] = observation.component
            
        return data
        
    def _condition_to_fhir(self, condition: FHIRCondition) -> Dict[str, Any]:
        """Convert FHIRCondition to FHIR JSON"""
        data = {
            "resourceType": "Condition",
            "clinicalStatus": condition.clinical_status,
            "verificationStatus": condition.verification_status,
            "category": condition.category,
            "code": condition.code,
            "subject": condition.subject
        }
        
        if condition.severity:
            data["severity"] = condition.severity
            
        if condition.encounter:
            data["encounter"] = condition.encounter
            
        if condition.onset:
            if isinstance(condition.onset, datetime):
                data["onsetDateTime"] = condition.onset.isoformat()
            elif isinstance(condition.onset, str):
                data["onsetString"] = condition.onset
            elif isinstance(condition.onset, dict):
                data["onsetPeriod"] = condition.onset
                
        if condition.recorded_date:
            data["recordedDate"] = condition.recorded_date.isoformat()
            
        if condition.recorder:
            data["recorder"] = condition.recorder
            
        if condition.asserter:
            data["asserter"] = condition.asserter
            
        if condition.note:
            data["note"] = condition.note
            
        if condition.evidence:
            data["evidence"] = condition.evidence
            
        return data


class EpicFHIRClient(GenericFHIRClient):
    """Epic-specific FHIR client with custom extensions"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any]):
        super().__init__(base_url, auth_config)
        
    async def get_smart_on_fhir_metadata(self) -> Dict[str, Any]:
        """Get Epic SMART on FHIR metadata"""
        try:
            url = f"{self.base_url}/metadata"
            headers = {"Accept": "application/fhir+json"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting Epic metadata: {e}")
            return {}
            
    async def get_patient_everything(self, patient_id: str) -> Dict[str, Any]:
        """Get all data for a patient using Epic's $everything operation"""
        try:
            url = f"{self.base_url}/Patient/{patient_id}/$everything"
            headers = await self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    logger.error(f"Error getting patient everything for {patient_id}: {error}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting patient everything for {patient_id}: {e}")
            return {}


class CernerFHIRClient(GenericFHIRClient):
    """Cerner-specific FHIR client"""
    
    def __init__(self, base_url: str, auth_config: Dict[str, Any]):
        super().__init__(base_url, auth_config)
        
    async def get_patient_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Get enhanced patient demographics from Cerner"""
        try:
            # Cerner may have specific demographics endpoints
            patient = await self.get_patient(patient_id)
            if patient:
                return {
                    "name": patient.name,
                    "gender": patient.gender,
                    "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
                    "address": patient.address,
                    "telecom": patient.telecom
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting Cerner demographics for {patient_id}: {e}")
            return {}


class EHRIntegrationManager:
    """Manages multiple EHR system integrations"""
    
    def __init__(self):
        self.clients: Dict[str, FHIRClient] = {}
        self.system_mappings: Dict[str, EHRSystem] = {}
        
    async def add_ehr_system(self, system_id: str, ehr_type: EHRSystem,
                           base_url: str, auth_config: Dict[str, Any]):
        """Add EHR system integration"""
        try:
            if ehr_type == EHRSystem.EPIC:
                client = EpicFHIRClient(base_url, auth_config)
            elif ehr_type == EHRSystem.CERNER:
                client = CernerFHIRClient(base_url, auth_config)
            else:
                client = GenericFHIRClient(base_url, auth_config)
                
            await client.initialize()
            
            self.clients[system_id] = client
            self.system_mappings[system_id] = ehr_type
            
            logger.info(f"Added EHR system: {system_id} ({ehr_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to add EHR system {system_id}: {e}")
            raise
            
    async def get_patient_from_any_system(self, patient_identifier: str,
                                        identifier_system: Optional[str] = None) -> Optional[FHIRPatient]:
        """Find patient across all connected EHR systems"""
        for system_id, client in self.clients.items():
            try:
                # Try direct ID lookup first
                patient = await client.get_patient(patient_identifier)
                if patient:
                    return patient
                    
                # Try search by identifier
                search_criteria = {"identifier": patient_identifier}
                if identifier_system:
                    search_criteria["identifier"] = f"{identifier_system}|{patient_identifier}"
                    
                patients = await client.search_patients(search_criteria)
                if patients:
                    return patients[0]  # Return first match
                    
            except Exception as e:
                logger.warning(f"Error searching for patient in {system_id}: {e}")
                continue
                
        return None
        
    async def get_comprehensive_patient_data(self, patient_id: str,
                                           system_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive patient data from EHR"""
        if system_id and system_id in self.clients:
            clients_to_check = [self.clients[system_id]]
        else:
            clients_to_check = list(self.clients.values())
            
        comprehensive_data = {
            "patient": None,
            "observations": [],
            "conditions": [],
            "medications": [],
            "encounters": [],
            "source_systems": []
        }
        
        for client in clients_to_check:
            try:
                # Get patient
                if not comprehensive_data["patient"]:
                    patient = await client.get_patient(patient_id)
                    if patient:
                        comprehensive_data["patient"] = patient
                        
                # Get observations
                observations = await client.get_observations(patient_id)
                comprehensive_data["observations"].extend(observations)
                
                # Get conditions
                conditions = await client.get_conditions(patient_id)
                comprehensive_data["conditions"].extend(conditions)
                
                # Add source system info
                system_name = next((k for k, v in self.clients.items() if v == client), "unknown")
                comprehensive_data["source_systems"].append(system_name)
                
            except Exception as e:
                logger.error(f"Error getting comprehensive data from EHR: {e}")
                
        return comprehensive_data
        
    async def create_mental_health_observation(self, patient_id: str,
                                             assessment_type: str,
                                             score: float,
                                             system_id: Optional[str] = None) -> List[str]:
        """Create mental health assessment observation in EHR(s)"""
        # Define mental health observation codes
        assessment_codes = {
            "phq9": {
                "system": "http://loinc.org",
                "code": "44249-1",
                "display": "PHQ-9 quick depression assessment panel"
            },
            "gad7": {
                "system": "http://loinc.org", 
                "code": "69737-5",
                "display": "Generalized anxiety disorder 7 item (GAD-7)"
            },
            "dass21": {
                "system": "http://loinc.org",
                "code": "72133-2",
                "display": "Depression Anxiety Stress Scales 21 item"
            }
        }
        
        if assessment_type.lower() not in assessment_codes:
            raise ValueError(f"Unsupported assessment type: {assessment_type}")
            
        code_info = assessment_codes[assessment_type.lower()]
        
        observation = FHIRObservation(
            id="",  # Will be generated by EHR
            status="final",
            category=[{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }]
            }],
            code={
                "coding": [{
                    "system": code_info["system"],
                    "code": code_info["code"],
                    "display": code_info["display"]
                }]
            },
            subject={"reference": f"Patient/{patient_id}"},
            effective_datetime=datetime.utcnow(),
            issued=datetime.utcnow(),
            performer=[{"reference": "Organization/solace-ai"}],
            value={
                "value": score,
                "unit": "score",
                "system": "http://unitsofmeasure.org",
                "code": "{score}"
            },
            interpretation=[{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": self._get_interpretation_code(assessment_type, score),
                    "display": self._get_interpretation_display(assessment_type, score)
                }]
            }],
            note=[{
                "text": f"Mental health assessment conducted by Solace AI platform"
            }]
        )
        
        created_ids = []
        
        if system_id and system_id in self.clients:
            clients_to_update = [self.clients[system_id]]
        else:
            clients_to_update = list(self.clients.values())
            
        for client in clients_to_update:
            try:
                observation_id = await client.create_observation(observation)
                created_ids.append(observation_id)
                logger.info(f"Created mental health observation {observation_id} in EHR")
            except Exception as e:
                logger.error(f"Failed to create observation in EHR: {e}")
                
        return created_ids
        
    def _get_interpretation_code(self, assessment_type: str, score: float) -> str:
        """Get interpretation code based on assessment type and score"""
        if assessment_type.lower() == "phq9":
            if score >= 20:
                return "H"  # High
            elif score >= 15:
                return "H"  # High  
            elif score >= 10:
                return "M"  # Moderate
            elif score >= 5:
                return "L"  # Low
            else:
                return "N"  # Normal
        elif assessment_type.lower() == "gad7":
            if score >= 15:
                return "H"  # High
            elif score >= 10:
                return "M"  # Moderate
            elif score >= 5:
                return "L"  # Low
            else:
                return "N"  # Normal
        else:
            return "N"  # Default to normal
            
    def _get_interpretation_display(self, assessment_type: str, score: float) -> str:
        """Get interpretation display text"""
        if assessment_type.lower() == "phq9":
            if score >= 20:
                return "Severe depression"
            elif score >= 15:
                return "Moderately severe depression"
            elif score >= 10:
                return "Moderate depression"
            elif score >= 5:
                return "Mild depression"
            else:
                return "Minimal depression"
        elif assessment_type.lower() == "gad7":
            if score >= 15:
                return "Severe anxiety"
            elif score >= 10:
                return "Moderate anxiety"
            elif score >= 5:
                return "Mild anxiety"
            else:
                return "Minimal anxiety"
        else:
            return "Normal range"
            
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all EHR integrations"""
        status = {
            "total_systems": len(self.clients),
            "systems": {}
        }
        
        for system_id, client in self.clients.items():
            try:
                # Test connection with a simple metadata call
                if hasattr(client, '_get_headers'):
                    headers = await client._get_headers()
                    test_url = f"{client.base_url}/metadata"
                    
                    async with client.session.get(test_url, headers=headers) as response:
                        if response.status == 200:
                            system_status = "healthy"
                        else:
                            system_status = "error"
                else:
                    system_status = "unknown"
                    
                status["systems"][system_id] = {
                    "status": system_status,
                    "type": self.system_mappings.get(system_id, "unknown").value,
                    "base_url": client.base_url
                }
                
            except Exception as e:
                status["systems"][system_id] = {
                    "status": "error",
                    "error": str(e),
                    "type": self.system_mappings.get(system_id, "unknown").value
                }
                
        return status