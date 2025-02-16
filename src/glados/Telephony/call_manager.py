"""
CallManager handles active call sessions and coordinates audio streaming.
"""

from typing import Dict, Optional, Any
import asyncio
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, ConfigDict
import numpy as np

from .telnyx_client import TelnyxClient
from .audio_bridge import AudioBridge
from ..ASR.asr import AudioTranscriber
from ..ASR.vad import VAD
from ..TTS.tts_glados import Synthesizer as GladosTTS
from ..utils.spoken_text_converter import SpokenTextConverter


class CallSession(BaseModel):
    """Represents an active call session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    call_control_id: str
    call_leg_id: Optional[str] = None
    start_time: datetime
    phone_number: str
    audio_bridge: Optional[AudioBridge] = None
    asr: Optional[AudioTranscriber] = None
    vad: Optional[VAD] = None
    tts: Optional[GladosTTS] = None
    text_converter: Optional[SpokenTextConverter] = None
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
            # Initialize ASR, VAD, and TTS components
            session = CallSession(
                call_control_id=call_control_id,
                start_time=datetime.now(),
                phone_number=phone_number,
                asr=AudioTranscriber(),
                vad=VAD(),
                tts=GladosTTS(),
                text_converter=SpokenTextConverter()
            )
            self._active_sessions[call_control_id] = session
            
            # Start audio processing task
            asyncio.create_task(self._process_audio(call_control_id))
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
        
    async def _process_audio(self, call_control_id: str) -> None:
        """Process audio for ASR and TTS."""
        session = self._active_sessions.get(call_control_id)
        if not session or not session.audio_bridge:
            return

        buffer = []
        is_speaking = False
        
        while call_control_id in self._active_sessions:
            try:
                # Get audio from the bridge
                audio_chunk = await session.audio_bridge.get_audio()
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                    continue

                # Check if someone is speaking using VAD
                if session.vad and not is_speaking:
                    is_speaking = session.vad.is_speech(audio_chunk)
                    if is_speaking:
                        logger.info(f"Speech detected in call {call_control_id}")
                        buffer = [audio_chunk]
                elif session.vad and is_speaking:
                    is_speaking = session.vad.is_speech(audio_chunk)
                    if is_speaking:
                        buffer.append(audio_chunk)
                    else:
                        # Speech ended, process the buffer
                        if buffer:
                            # Concatenate audio chunks
                            full_audio = np.concatenate(buffer)
                            
                            # Perform ASR
                            if session.asr:
                                text = session.asr.transcribe(full_audio)
                                if text:
                                    logger.info(f"Transcribed text from call {call_control_id}: {text}")
                                    
                                    # Generate GLaDOS response
                                    if session.text_converter and session.tts:
                                        spoken_text = session.text_converter.text_to_spoken(text)
                                        logger.info(f"GLaDOS response: {spoken_text}")
                                        
                                        # Generate and send audio response
                                        audio_response = session.tts.generate_speech_audio(spoken_text)
                                        await session.audio_bridge.send_audio(audio_response)
                            
                            buffer = []

            except Exception as e:
                logger.error(f"Error processing audio in call {call_control_id}: {e}")
                await asyncio.sleep(0.1)
        
    async def _handle_call_initiated(self, payload: Dict[str, Any]) -> None:
        """Handle call.initiated event."""
        data = payload.get("data", {}).get("payload", {})
        call_control_id = data.get("call_control_id")
        call_leg_id = data.get("call_leg_id")
        
        if session := self._active_sessions.get(call_control_id):
            session.call_leg_id = call_leg_id
            logger.info(f"Call initiated: {call_control_id} (leg: {call_leg_id})")
            
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
                logger.info(f"TTS paused - caller speaking in call {call_control_id}")
                
    async def _handle_speaking_ended(self, payload: Dict[str, Any]) -> None:
        """Handle call.speaking.ended event."""
        call_control_id = payload["data"]["payload"]["call_control_id"]
        if session := self._active_sessions.get(call_control_id):
            if session.audio_bridge:
                await session.audio_bridge.resume_tts()
                logger.info(f"TTS resumed - caller stopped speaking in call {call_control_id}") 