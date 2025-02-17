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
    """
    
    def __init__(self, telnyx_client: TelnyxClient):
        """Initialize the CallManager."""
        self.telnyx_client = telnyx_client
        self._active_sessions: Dict[str, CallSession] = {}
        self._setup_call_handlers()
        
    def _setup_call_handlers(self) -> None:
        """Register handlers for call events."""
        self.telnyx_client.register_call_handler("call.initiated", self._handle_call_initiated)
        self.telnyx_client.register_call_handler("call.answered", self._handle_call_answered)
        self.telnyx_client.register_call_handler("call.hangup", self._handle_call_hangup)
        
    async def start_call(self, phone_number: str) -> Optional[str]:
        """Start a new outbound call."""
        call_control_id = await self.telnyx_client.make_call(phone_number)
        if call_control_id:
            # Initialize session with GLaDOS components
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
            
            # Have GLaDOS say hello when the call is answered
            asyncio.create_task(self._say_hello(call_control_id))
            
            return call_control_id
        return None
        
    async def _say_hello(self, call_id: str) -> None:
        """Have GLaDOS say hello when the call is answered."""
        if session := self._active_sessions.get(call_id):
            try:
                # Wait a bit for the call to be fully established
                await asyncio.sleep(1)
                
                # Generate GLaDOS greeting
                if session.text_converter and session.tts:
                    greeting = "Hello, and welcome to the Aperture Science Enrichment Center."
                    spoken_text = session.text_converter.text_to_spoken(greeting)
                    audio = session.tts.generate_speech_audio(spoken_text)
                    
                    # Send audio to the call
                    await self.telnyx_client.send_audio(call_id, audio)
                    logger.info(f"Sent greeting to call {call_id}")
            except Exception as e:
                logger.error(f"Error sending greeting: {e}")
        
    async def end_call(self, call_control_id: str) -> bool:
        """End an active call."""
        if session := self._active_sessions.get(call_control_id):
            if await self.telnyx_client.end_call(call_control_id):
                if session.audio_bridge:
                    await session.audio_bridge.stop()
                del self._active_sessions[call_control_id]
                return True
        return False
        
    async def _handle_call_initiated(self, payload: dict) -> None:
        """Handle call initiation event."""
        data = payload.get('data', {})
        payload_data = data.get('payload', {})
        call_id = payload_data.get('call_control_id')
        
        if not call_id:
            logger.error(f"No call_control_id in payload: {payload}")
            return
            
        self._active_sessions[call_id] = CallSession(
            call_control_id=call_id,
            start_time=datetime.now(),
            phone_number=payload_data.get('to', '')
        )
        logger.info(f"Call initiated: {call_id}")
        
    async def _handle_call_answered(self, payload: Dict[str, Any]) -> None:
        """Handle call.answered event."""
        data = payload.get('data', {})
        payload_data = data.get('payload', {})
        call_id = payload_data.get('call_control_id')
        
        if not call_id:
            logger.error(f"No call_control_id in answered payload: {payload}")
            return
            
        if call_id in self._active_sessions:
            # Start audio processing for the call
            asyncio.create_task(self._process_audio(call_id))
            # Have GLaDOS say hello
            await self._say_hello(call_id)
            logger.info(f"Call answered and processing started: {call_id}")
        else:
            logger.warning(f"Call {call_id} not found in active sessions for answer")
        
    async def _handle_call_hangup(self, payload: Dict[str, Any]) -> None:
        """Handle call.hangup event."""
        data = payload.get('data', {})
        payload_data = data.get('payload', {})
        call_id = payload_data.get('call_control_id')
        
        if not call_id:
            logger.error(f"No call_control_id in hangup payload: {payload}")
            return
            
        if call_id in self._active_sessions:
            await self.end_call(call_id)
            logger.info(f"Call ended: {call_id}")
        else:
            logger.warning(f"Call {call_id} not found in active sessions for hangup")
        
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
                                        await self.telnyx_client.send_audio(call_control_id, audio_response)
                                        logger.info(f"Sent audio response to call {call_control_id}")
                            
                            buffer = []

            except Exception as e:
                logger.error(f"Error processing audio in call {call_control_id}: {e}")
                await asyncio.sleep(0.1) 