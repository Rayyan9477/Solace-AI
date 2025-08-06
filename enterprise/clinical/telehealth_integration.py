"""
Telehealth Platform Integration
Integrates with major telehealth platforms for seamless virtual care
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import uuid
import hmac
import hashlib
import base64
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TelehealthPlatform(Enum):
    ZOOM_HEALTHCARE = "zoom_healthcare"
    WEBEX_HEALTHCARE = "webex_healthcare"
    TEAMS_HEALTHCARE = "teams_healthcare"
    DOXY_ME = "doxy_me"
    SIMPLE_PRACTICE = "simple_practice"
    AMWELL = "amwell"
    TELADOC = "teladoc"
    GENERIC_PLATFORM = "generic"


class SessionStatus(Enum):
    SCHEDULED = "scheduled"
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class ParticipantRole(Enum):
    PATIENT = "patient"
    PROVIDER = "provider"
    OBSERVER = "observer"
    RECORDER = "recorder"


@dataclass
class TelehealthParticipant:
    """Telehealth session participant"""
    participant_id: str
    name: str
    email: str
    role: ParticipantRole
    phone: Optional[str] = None
    join_url: Optional[str] = None
    join_time: Optional[datetime] = None
    leave_time: Optional[datetime] = None
    is_present: bool = False
    camera_enabled: bool = False
    microphone_enabled: bool = False


@dataclass
class TelehealthSession:
    """Telehealth session data"""
    session_id: str
    meeting_id: str
    topic: str
    start_time: datetime
    duration_minutes: int
    status: SessionStatus
    participants: List[TelehealthParticipant]
    host_key: Optional[str] = None
    password: Optional[str] = None
    join_url: Optional[str] = None
    recording_enabled: bool = False
    recording_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    platform: Optional[TelehealthPlatform] = None
    room_settings: Dict[str, Any] = field(default_factory=dict)
    session_notes: str = ""


@dataclass
class SessionAnalytics:
    """Telehealth session analytics"""
    session_id: str
    total_duration: int
    participant_count: int
    attendance_rate: float
    average_join_time: float
    connection_quality_score: float
    technical_issues: List[str]
    engagement_metrics: Dict[str, float]
    outcome_summary: str


class TelehealthClient(ABC):
    """Abstract telehealth platform client"""
    
    @abstractmethod
    async def create_session(self, session_data: Dict[str, Any]) -> TelehealthSession:
        pass
    
    @abstractmethod
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def cancel_session(self, session_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_session_status(self, session_id: str) -> Optional[TelehealthSession]:
        pass
    
    @abstractmethod
    async def start_recording(self, session_id: str) -> bool:
        pass
    
    @abstractmethod
    async def stop_recording(self, session_id: str) -> Optional[str]:
        pass


class ZoomHealthcareClient(TelehealthClient):
    """Zoom for Healthcare integration"""
    
    def __init__(self, api_key: str, api_secret: str, account_id: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_id = account_id
        self.base_url = "https://api.zoom.us/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize Zoom client"""
        self.session = aiohttp.ClientSession()
        await self._get_access_token()
        logger.info("Zoom Healthcare client initialized")
        
    async def _get_access_token(self):
        """Get OAuth access token"""
        try:
            token_url = "https://zoom.us/oauth/token"
            
            credentials = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
            
            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "account_credentials",
                "account_id": self.account_id
            }
            
            async with self.session.post(token_url, headers=headers, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
                else:
                    error = await response.text()
                    raise Exception(f"Failed to get Zoom access token: {error}")
                    
        except Exception as e:
            logger.error(f"Zoom authentication error: {e}")
            raise
            
    async def _get_headers(self) -> Dict[str, str]:
        """Get authenticated request headers"""
        # Check if token needs refresh
        if (self.token_expires and 
            datetime.utcnow() > self.token_expires - timedelta(minutes=5)):
            await self._get_access_token()
            
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
    async def create_session(self, session_data: Dict[str, Any]) -> TelehealthSession:
        """Create Zoom meeting for telehealth session"""
        try:
            url = f"{self.base_url}/users/me/meetings"
            headers = await self._get_headers()
            
            meeting_config = {
                "topic": session_data.get("topic", "Mental Health Session"),
                "type": 2,  # Scheduled meeting
                "start_time": session_data["start_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration": session_data.get("duration_minutes", 60),
                "timezone": "UTC",
                "password": session_data.get("password", self._generate_password()),
                "settings": {
                    "host_video": True,
                    "participant_video": True,
                    "cn_meeting": False,
                    "in_meeting": False,
                    "join_before_host": False,
                    "mute_upon_entry": True,
                    "watermark": True,
                    "use_pmi": False,
                    "approval_type": 1,  # Manually approve
                    "audio": "voip",
                    "auto_recording": "cloud" if session_data.get("recording_enabled") else "none",
                    "enforce_login": True,
                    "waiting_room": True,
                    "meeting_authentication": True
                }
            }
            
            async with self.session.post(url, headers=headers, json=meeting_config) as response:
                if response.status == 201:
                    meeting_data = await response.json()
                    
                    participants = []
                    for participant_data in session_data.get("participants", []):
                        participant = TelehealthParticipant(
                            participant_id=str(uuid.uuid4()),
                            name=participant_data["name"],
                            email=participant_data["email"],
                            role=ParticipantRole(participant_data.get("role", "patient")),
                            phone=participant_data.get("phone"),
                            join_url=meeting_data["join_url"]
                        )
                        participants.append(participant)
                        
                    session = TelehealthSession(
                        session_id=str(uuid.uuid4()),
                        meeting_id=str(meeting_data["id"]),
                        topic=meeting_data["topic"],
                        start_time=datetime.fromisoformat(meeting_data["start_time"].replace("Z", "+00:00")),
                        duration_minutes=meeting_data["duration"],
                        status=SessionStatus.SCHEDULED,
                        participants=participants,
                        host_key=meeting_data.get("host_key"),
                        password=meeting_data["password"],
                        join_url=meeting_data["join_url"],
                        recording_enabled=session_data.get("recording_enabled", False),
                        platform=TelehealthPlatform.ZOOM_HEALTHCARE,
                        room_settings={
                            "waiting_room": True,
                            "meeting_authentication": True,
                            "participant_video": True
                        }
                    )
                    
                    return session
                else:
                    error = await response.text()
                    raise Exception(f"Failed to create Zoom meeting: {error}")
                    
        except Exception as e:
            logger.error(f"Error creating Zoom session: {e}")
            raise
            
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update Zoom meeting"""
        try:
            # In real implementation, would need to store session_id -> meeting_id mapping
            meeting_id = updates.get("meeting_id")
            if not meeting_id:
                return False
                
            url = f"{self.base_url}/meetings/{meeting_id}"
            headers = await self._get_headers()
            
            update_data = {}
            if "topic" in updates:
                update_data["topic"] = updates["topic"]
            if "start_time" in updates:
                update_data["start_time"] = updates["start_time"].strftime("%Y-%m-%dT%H:%M:%SZ")
            if "duration_minutes" in updates:
                update_data["duration"] = updates["duration_minutes"]
                
            async with self.session.patch(url, headers=headers, json=update_data) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Error updating Zoom session: {e}")
            return False
            
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel Zoom meeting"""
        try:
            # Would need session_id -> meeting_id mapping
            meeting_id = "placeholder"  # Get from database
            
            url = f"{self.base_url}/meetings/{meeting_id}"
            headers = await self._get_headers()
            
            async with self.session.delete(url, headers=headers) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Error cancelling Zoom session: {e}")
            return False
            
    async def get_session_status(self, session_id: str) -> Optional[TelehealthSession]:
        """Get Zoom meeting status"""
        try:
            # Would need session_id -> meeting_id mapping
            meeting_id = "placeholder"  # Get from database
            
            url = f"{self.base_url}/meetings/{meeting_id}"
            headers = await self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    meeting_data = await response.json()
                    
                    # Convert to TelehealthSession (simplified)
                    session = TelehealthSession(
                        session_id=session_id,
                        meeting_id=str(meeting_data["id"]),
                        topic=meeting_data["topic"],
                        start_time=datetime.fromisoformat(meeting_data["start_time"].replace("Z", "+00:00")),
                        duration_minutes=meeting_data["duration"],
                        status=SessionStatus.SCHEDULED,  # Would need to determine actual status
                        participants=[],
                        platform=TelehealthPlatform.ZOOM_HEALTHCARE
                    )
                    
                    return session
                    
        except Exception as e:
            logger.error(f"Error getting Zoom session status: {e}")
            
        return None
        
    async def start_recording(self, session_id: str) -> bool:
        """Start Zoom meeting recording"""
        try:
            meeting_id = "placeholder"  # Get from database
            
            url = f"{self.base_url}/meetings/{meeting_id}/recordings"
            headers = await self._get_headers()
            
            data = {"action": "start"}
            
            async with self.session.patch(url, headers=headers, json=data) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Error starting Zoom recording: {e}")
            return False
            
    async def stop_recording(self, session_id: str) -> Optional[str]:
        """Stop Zoom meeting recording and return URL"""
        try:
            meeting_id = "placeholder"  # Get from database
            
            url = f"{self.base_url}/meetings/{meeting_id}/recordings"
            headers = await self._get_headers()
            
            data = {"action": "stop"}
            
            async with self.session.patch(url, headers=headers, json=data) as response:
                if response.status == 204:
                    # Get recording URL
                    await asyncio.sleep(5)  # Wait for processing
                    
                    recordings_url = f"{self.base_url}/meetings/{meeting_id}/recordings"
                    async with self.session.get(recordings_url, headers=headers) as rec_response:
                        if rec_response.status == 200:
                            recordings_data = await rec_response.json()
                            if recordings_data.get("recording_files"):
                                return recordings_data["recording_files"][0].get("download_url")
                                
        except Exception as e:
            logger.error(f"Error stopping Zoom recording: {e}")
            
        return None
        
    def _generate_password(self) -> str:
        """Generate secure meeting password"""
        import random
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))


class DoxyMeClient(TelehealthClient):
    """Doxy.me integration client"""
    
    def __init__(self, clinic_id: str, api_key: str):
        self.clinic_id = clinic_id
        self.api_key = api_key
        self.base_url = "https://api.doxy.me/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize Doxy.me client"""
        self.session = aiohttp.ClientSession()
        logger.info("Doxy.me client initialized")
        
    async def _get_headers(self) -> Dict[str, str]:
        """Get authenticated request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    async def create_session(self, session_data: Dict[str, Any]) -> TelehealthSession:
        """Create Doxy.me room"""
        try:
            url = f"{self.base_url}/rooms"
            headers = await self._get_headers()
            
            room_config = {
                "name": session_data.get("topic", "Mental Health Session"),
                "clinic_id": self.clinic_id,
                "start_time": session_data["start_time"].isoformat(),
                "duration": session_data.get("duration_minutes", 60),
                "require_password": True,
                "waiting_room": True,
                "recording_enabled": session_data.get("recording_enabled", False)
            }
            
            async with self.session.post(url, headers=headers, json=room_config) as response:
                if response.status == 201:
                    room_data = await response.json()
                    
                    participants = []
                    for participant_data in session_data.get("participants", []):
                        participant = TelehealthParticipant(
                            participant_id=str(uuid.uuid4()),
                            name=participant_data["name"],
                            email=participant_data["email"],
                            role=ParticipantRole(participant_data.get("role", "patient")),
                            join_url=room_data["join_url"]
                        )
                        participants.append(participant)
                        
                    session = TelehealthSession(
                        session_id=str(uuid.uuid4()),
                        meeting_id=room_data["room_id"],
                        topic=room_data["name"],
                        start_time=session_data["start_time"],
                        duration_minutes=session_data.get("duration_minutes", 60),
                        status=SessionStatus.SCHEDULED,
                        participants=participants,
                        join_url=room_data["join_url"],
                        password=room_data.get("password"),
                        recording_enabled=session_data.get("recording_enabled", False),
                        platform=TelehealthPlatform.DOXY_ME
                    )
                    
                    return session
                else:
                    error = await response.text()
                    raise Exception(f"Failed to create Doxy.me room: {error}")
                    
        except Exception as e:
            logger.error(f"Error creating Doxy.me session: {e}")
            raise
            
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update Doxy.me room"""
        # Implementation would depend on Doxy.me API capabilities
        return True
        
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel Doxy.me room"""
        # Implementation would depend on Doxy.me API capabilities
        return True
        
    async def get_session_status(self, session_id: str) -> Optional[TelehealthSession]:
        """Get Doxy.me room status"""
        # Implementation would depend on Doxy.me API capabilities
        return None
        
    async def start_recording(self, session_id: str) -> bool:
        """Start Doxy.me recording"""
        return True
        
    async def stop_recording(self, session_id: str) -> Optional[str]:
        """Stop Doxy.me recording"""
        return None


class TelehealthIntegrationManager:
    """Manages multiple telehealth platform integrations"""
    
    def __init__(self):
        self.clients: Dict[str, TelehealthClient] = {}
        self.sessions: Dict[str, TelehealthSession] = {}
        self.session_callbacks: Dict[str, List[Callable]] = {}
        self.default_platform: Optional[str] = None
        
    async def add_platform(self, platform_id: str, platform_type: TelehealthPlatform,
                          config: Dict[str, Any]):
        """Add telehealth platform integration"""
        try:
            if platform_type == TelehealthPlatform.ZOOM_HEALTHCARE:
                client = ZoomHealthcareClient(
                    api_key=config["api_key"],
                    api_secret=config["api_secret"],
                    account_id=config["account_id"]
                )
            elif platform_type == TelehealthPlatform.DOXY_ME:
                client = DoxyMeClient(
                    clinic_id=config["clinic_id"],
                    api_key=config["api_key"]
                )
            else:
                raise ValueError(f"Unsupported platform type: {platform_type}")
                
            await client.initialize()
            self.clients[platform_id] = client
            
            if not self.default_platform:
                self.default_platform = platform_id
                
            logger.info(f"Added telehealth platform: {platform_id} ({platform_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to add telehealth platform {platform_id}: {e}")
            raise
            
    async def schedule_session(self, session_data: Dict[str, Any],
                             platform_id: Optional[str] = None) -> TelehealthSession:
        """Schedule a telehealth session"""
        if platform_id and platform_id not in self.clients:
            raise ValueError(f"Platform {platform_id} not configured")
            
        client_id = platform_id or self.default_platform
        if not client_id:
            raise ValueError("No telehealth platform configured")
            
        client = self.clients[client_id]
        session = await client.create_session(session_data)
        
        # Store session
        self.sessions[session.session_id] = session
        
        # Schedule reminders and notifications
        await self._schedule_session_reminders(session)
        
        logger.info(f"Scheduled telehealth session: {session.session_id}")
        return session
        
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update a telehealth session"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        platform_client = None
        
        # Find the client for this session
        for client in self.clients.values():
            if hasattr(client, 'platform') and client.platform == session.platform:
                platform_client = client
                break
                
        if not platform_client:
            return False
            
        # Update in platform
        success = await platform_client.update_session(session_id, updates)
        
        if success:
            # Update local session data
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
                    
        return success
        
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a telehealth session"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Find and use appropriate client
        platform_client = None
        for client in self.clients.values():
            if hasattr(client, 'platform') and client.platform == session.platform:
                platform_client = client
                break
                
        if platform_client:
            success = await platform_client.cancel_session(session_id)
            if success:
                session.status = SessionStatus.CANCELLED
                await self._trigger_session_callbacks(session_id, "cancelled")
                return True
                
        return False
        
    async def start_session(self, session_id: str) -> bool:
        """Mark session as started"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        session.status = SessionStatus.IN_PROGRESS
        
        await self._trigger_session_callbacks(session_id, "started")
        
        # Auto-start recording if enabled
        if session.recording_enabled:
            await self.start_recording(session_id)
            
        logger.info(f"Started telehealth session: {session_id}")
        return True
        
    async def end_session(self, session_id: str, session_notes: str = "") -> bool:
        """End a telehealth session"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        session.status = SessionStatus.COMPLETED
        session.ended_at = datetime.utcnow()
        session.session_notes = session_notes
        
        # Stop recording if active
        if session.recording_enabled:
            recording_url = await self.stop_recording(session_id)
            if recording_url:
                session.recording_url = recording_url
                
        await self._trigger_session_callbacks(session_id, "ended")
        
        logger.info(f"Ended telehealth session: {session_id}")
        return True
        
    async def start_recording(self, session_id: str) -> bool:
        """Start recording a session"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Find appropriate client
        for client in self.clients.values():
            if hasattr(client, 'platform') and client.platform == session.platform:
                success = await client.start_recording(session_id)
                if success:
                    session.recording_enabled = True
                    logger.info(f"Started recording for session: {session_id}")
                return success
                
        return False
        
    async def stop_recording(self, session_id: str) -> Optional[str]:
        """Stop recording and return URL"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # Find appropriate client
        for client in self.clients.values():
            if hasattr(client, 'platform') and client.platform == session.platform:
                recording_url = await client.stop_recording(session_id)
                if recording_url:
                    session.recording_url = recording_url
                    logger.info(f"Stopped recording for session: {session_id}")
                return recording_url
                
        return None
        
    async def get_session_analytics(self, session_id: str) -> Optional[SessionAnalytics]:
        """Get session analytics"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        if session.status != SessionStatus.COMPLETED or not session.ended_at:
            return None
            
        # Calculate analytics
        total_duration = int((session.ended_at - session.start_time).total_seconds() / 60)
        participant_count = len(session.participants)
        
        # Calculate attendance rate
        present_participants = sum(1 for p in session.participants if p.is_present)
        attendance_rate = present_participants / participant_count if participant_count > 0 else 0
        
        # Calculate average join time (simplified)
        join_times = [p.join_time for p in session.participants if p.join_time]
        if join_times:
            session_start = session.start_time
            join_delays = [(jt - session_start).total_seconds() for jt in join_times]
            average_join_time = sum(join_delays) / len(join_delays)
        else:
            average_join_time = 0
            
        analytics = SessionAnalytics(
            session_id=session_id,
            total_duration=total_duration,
            participant_count=participant_count,
            attendance_rate=attendance_rate,
            average_join_time=average_join_time,
            connection_quality_score=8.5,  # Would be calculated from platform data
            technical_issues=[],  # Would be populated from session events
            engagement_metrics={
                "camera_usage_rate": 0.85,  # Percentage of time cameras were on
                "microphone_usage_rate": 0.92,  # Percentage of time microphones were active
                "interaction_rate": 0.78  # Based on chat, reactions, etc.
            },
            outcome_summary=session.session_notes[:200] if session.session_notes else ""
        )
        
        return analytics
        
    async def add_session_callback(self, session_id: str, callback: Callable):
        """Add callback for session events"""
        if session_id not in self.session_callbacks:
            self.session_callbacks[session_id] = []
        self.session_callbacks[session_id].append(callback)
        
    async def _schedule_session_reminders(self, session: TelehealthSession):
        """Schedule reminder notifications for session"""
        # This would integrate with notification system
        # For now, just log the reminder scheduling
        
        reminder_times = [
            session.start_time - timedelta(days=1),
            session.start_time - timedelta(hours=2),
            session.start_time - timedelta(minutes=15)
        ]
        
        for reminder_time in reminder_times:
            if reminder_time > datetime.utcnow():
                logger.info(f"Reminder scheduled for {session.session_id} at {reminder_time}")
                
    async def _trigger_session_callbacks(self, session_id: str, event_type: str):
        """Trigger callbacks for session events"""
        if session_id in self.session_callbacks:
            session = self.sessions[session_id]
            for callback in self.session_callbacks[session_id]:
                try:
                    await callback(session, event_type)
                except Exception as e:
                    logger.error(f"Error in session callback: {e}")
                    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get telehealth integration status"""
        status = {
            "total_platforms": len(self.clients),
            "active_sessions": len([s for s in self.sessions.values() 
                                  if s.status == SessionStatus.IN_PROGRESS]),
            "total_sessions": len(self.sessions),
            "platforms": {}
        }
        
        for platform_id, client in self.clients.items():
            try:
                # Test platform connectivity
                platform_status = "healthy"  # Would implement actual health check
                
                status["platforms"][platform_id] = {
                    "status": platform_status,
                    "type": getattr(client, 'platform', 'unknown'),
                    "sessions_today": len([s for s in self.sessions.values() 
                                         if s.platform and s.created_at.date() == datetime.now().date()])
                }
                
            except Exception as e:
                status["platforms"][platform_id] = {
                    "status": "error",
                    "error": str(e)
                }
                
        return status