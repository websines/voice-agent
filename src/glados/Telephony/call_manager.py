"""
CallManager handles active call sessions and coordinates audio streaming.
"""

from typing import Dict, Optional, Any
import asyncio
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, ConfigDict

from .telnyx_client import TelnyxClient
from .audio_bridge import AudioBridge


class CallSession(BaseModel):
    """Represents an active call session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    call_control_id: str
    call_leg_id: str
    start_time: datetime
    phone_number: str
    audio_bridge: Optional[AudioBridge] = None
    metadata: Dict[str, Any] = {}


class CallManager:
    """
    Manages active call sessions and coordinates audio streaming.
    
    This class keeps track of active calls, manages their state, and coordinates
    the audio streaming between Telnyx and GLaDOS's TTS/ASR components.
    """
    
    def __init__(self, telnyx_client: TelnyxClient):
        """
        Initialize the CallManager.
        
        Args:
            telnyx_client: Instance of TelnyxClient for call control
        """
        self.telnyx_client = telnyx_client
        self._active_sessions: Dict[str, CallSession] = {}
        self._setup_call_handlers()
        
    def _setup_call_handlers(self) -> None:
        """Register handlers for call events with the Telnyx client."""
        self.telnyx_client.register_call_handler("call.initiated", self._handle_call_initiated)
        self.telnyx_client.register_call_handler("call.answered", self._handle_call_answered)
        self.telnyx_client.register_call_handler("call.hangup", self._handle_call_hangup)
        self.telnyx_client.register_call_handler("call.speaking.started", self._handle_speaking_started)
        self.telnyx_client.register_call_handler("call.speaking.ended", self._handle_speaking_ended)
        
    async def start_call(self, phone_number: str) -> Optional[str]:
        """
        Start a new outbound call.
        
        Args:
            phone_number: The phone number to call in E.164 format
            
        Returns:
            call_control_id: The unique identifier for the call, or None if failed
        """
        call_control_id = await self.telnyx_client.make_call(phone_number)
        if call_control_id:
            session = CallSession(
                call_control_id=call_control_id,
                start_time=datetime.now(),
                phone_number=phone_number
            )
            self._active_sessions[call_control_id] = session
            return call_control_id
        return None
        
    async def end_call(self, call_control_id: str) -> bool:
        """
        End an active call.
        
        Args:
            call_control_id: The unique identifier for the call
            
        Returns:
            bool: True if call was ended successfully, False otherwise
        """
        if session := self._active_sessions.get(call_control_id):
            if self.telnyx_client.end_call(call_control_id):
                if session.audio_bridge:
                    await session.audio_bridge.stop()
                del self._active_sessions[call_control_id]
                return True
        return False
        
    async def _handle_call_initiated(self, payload: Dict[str, Any]) -> None:
        """Handle call.initiated event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        logger.info(f"Call initiated: {call_control_id}")
        
    async def _handle_call_answered(self, payload: Dict[str, Any]) -> None:
        """Handle call.answered event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        if session := self._active_sessions.get(call_control_id):
            session.audio_bridge = AudioBridge()
            await session.audio_bridge.start()
            logger.info(f"Call answered and audio bridge started: {call_control_id}")
            
    async def _handle_call_hangup(self, payload: Dict[str, Any]) -> None:
        """Handle call.hangup event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        await self.end_call(call_control_id)
        logger.info(f"Call ended: {call_control_id}")
        
    async def _handle_speaking_started(self, payload: Dict[str, Any]) -> None:
        """Handle call.speaking.started event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        if session := self._active_sessions.get(call_control_id):
            if session.audio_bridge:
                await session.audio_bridge.pause_tts()
                
    async def _handle_speaking_ended(self, payload: Dict[str, Any]) -> None:
        """Handle call.speaking.ended event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        if session := self._active_sessions.get(call_control_id):
            if session.audio_bridge:
                await session.audio_bridge.resume_tts() 